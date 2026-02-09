import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from skrample import common
from skrample.common import (
    DeltaPoint,
    DeltaUV,
    Point,
    Sample,
    SigmaTransform,
    Step,
    divf,
    ln,
    softmax,
    spowf,
)
from skrample.scheduling import SkrampleSchedule

from . import models, traits


@dataclass(frozen=True)
class SampleInput[T: Sample]:
    """Sampler result struct for easy management of multiple sampling stages.
    This should be accumulated in a list for the denoising loop in order to use higher order features"""

    sample: T
    "The model input"

    prediction: T
    "The model output"

    step: Step
    "Time at which to evaluate"

    noise: T | None
    "The extra stochastic noise"

    def delta_point(self, schedule: SkrampleSchedule) -> DeltaPoint:
        return DeltaPoint(*(Point(*p) for p in schedule.ipoints(self.step).tolist()))

    def delta_uv(self, schedule: SkrampleSchedule) -> DeltaUV:
        return self.delta_point(schedule).uv(schedule.sigma_transform)


@dataclass(frozen=True)
class SKSamples[T: Sample](SampleInput[T]):
    final: T
    "Final result. What you probably want"


@dataclass(frozen=True)
class StructuredSampler(ABC, traits.SamplingCommon):
    """Generic sampler structure with basic configurables and a stateless design.
    Abstract class not to be used directly.

    Unless otherwise specified, the Sample type is a stand-in that is
    type checked against torch.Tensor but should be generic enough to use with ndarrays or even raw floats"""

    @property
    def require_noise(self) -> bool:
        "Whether or not the sampler requires `noise: T` be passed"
        return False

    @property
    def require_previous(self) -> int:
        "How many prior samples the sampler needs in `previous: list[T]`"
        return 0

    @abstractmethod
    def sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]] = (),
    ) -> SKSamples[T]: ...

    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: Step | tuple[float, float],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        noise: T | None = None,
        previous: Sequence[SKSamples[T]] = (),
    ) -> SKSamples[T]:
        "Shorthand for `sample_packed`."
        return self.sample_packed(
            SampleInput(sample=sample, prediction=prediction, step=Step(*step), noise=noise),
            model_transform=model_transform,
            schedule=schedule,
            previous=previous,
        )

    def scale_input[T: Sample](self, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        """Some old samplers used to have different implmenetations,
        but now pretty for pretty much everything this is just a no-op."""
        return sample


@dataclass(frozen=True)
class StatedSampler(StructuredSampler):
    @abstractmethod
    def _sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
    ) -> T:
        "Must not modify or shadow `packed`"

    def sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]] = (),
    ) -> SKSamples[T]:
        return SKSamples(
            **(
                dataclasses.asdict(packed)
                | {
                    "final": self._sample_packed(
                        packed,
                        model_transform=model_transform,
                        schedule=schedule,
                        previous=previous,
                    )
                }
            )
        )


@dataclass(frozen=True)
class StructuredMultistep(StructuredSampler, traits.HigherOrder):
    """Samplers inheriting this trait support order > 1, and will require
    `prevous` be managed and passed to function accordingly."""

    @property
    def require_previous(self) -> int:
        return max(min(self.order, self.max_order()), self.min_order()) - 1

    def effective_order(self, step: Step, previous: Sequence[SKSamples]) -> int:
        "The order used in calculation given a step, schedule length, and previous sample count"
        return max(
            1,  # not min_order because previous may be < min. samplers should check effective >= min
            min(
                self.max_order(),
                round((position := step.position()) + 1),
                self.order,
                len(previous) + 1,
                round(step.amount() - position),  # lower for final is the default
                # len(schedule) - step,  # lower for final is the default
            ),
        )


@dataclass(frozen=True)
class StructuredStochastic(StructuredSampler):
    add_noise: bool = False
    "Flag for whether or not to add the given noise"

    @property
    def require_noise(self) -> bool:
        return self.add_noise


@dataclass(frozen=True)
class Euler(StatedSampler):
    """Basic sampler, the "safe" choice."""

    def _sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
    ) -> T:
        delta = packed.delta_point(schedule)
        return model_transform.forward(
            packed.sample,
            packed.prediction,
            delta.point_from.sigma,
            delta.point_to.sigma,
            schedule.sigma_transform,
        )


@dataclass(frozen=True)
class DPM(StatedSampler, StructuredMultistep, StructuredStochastic):
    """Good sampler, supports basically everything. Recommended default.

    https://arxiv.org/abs/2211.01095
    Page 4 Algo 2 for order=2
    Section 5 for SDE"""

    @staticmethod
    def max_order() -> int:
        return 3  # TODO(beinsezii): 3, 4+?

    def _sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
    ) -> T:
        delta = packed.delta_point(schedule)
        (sigma_u, sigma_v), (sigma_u_next, sigma_v_next) = delta.uv(schedule.sigma_transform)

        lambda_ = ln(divf(sigma_v, sigma_u))
        lambda_next = ln(divf(sigma_v_next, sigma_u_next))
        h = abs(lambda_next - lambda_)

        x_prediction = model_transform.to_x(
            packed.sample,
            packed.prediction,
            delta.point_from.sigma,
            schedule.sigma_transform,
        )

        if packed.noise is not None and self.add_noise:
            exp1 = math.exp(-h)
            hh = -2 * h
            noise_factor = sigma_u_next * math.sqrt(1 - math.exp(hh)) * packed.noise
        else:
            exp1 = 1
            hh = -h
            noise_factor = 0

        exp2 = math.expm1(hh)

        final = noise_factor + (sigma_u_next / sigma_u * exp1) * packed.sample

        # 1st order
        final -= (sigma_v_next * exp2) * x_prediction

        if (effective_order := self.effective_order(packed.step, previous)) >= 2:
            point_prev = schedule.ipoint(previous[-1].step.time_from)
            sigma_u_prev, sigma_v_prev = schedule.sigma_transform(point_prev.sigma)

            lambda_prev = ln(divf(sigma_v_prev, sigma_u_prev))
            h_prev = lambda_ - lambda_prev
            r = h_prev / h  # math people and their var names...

            # Calculate previous predicton from sample, output
            x_prediction_prev = model_transform.to_x(
                previous[-1].sample,
                previous[-1].prediction,
                point_prev.sigma,
                schedule.sigma_transform,
            )
            D1_0 = (1.0 / r) * (x_prediction - x_prediction_prev)

            if effective_order >= 3:
                point_prev2 = schedule.ipoint(previous[-2].step.time_from)
                sigma_u_prev2, sigma_v_prev2 = schedule.sigma_transform(point_prev2.sigma)

                lambda_prev2 = ln(divf(sigma_v_prev2, sigma_u_prev2))
                h_prev2 = lambda_prev - lambda_prev2
                r_prev2 = h_prev2 / h

                x_prediction_p2 = model_transform.to_x(
                    previous[-2].sample,
                    previous[-2].prediction,
                    point_prev2.sigma,
                    schedule.sigma_transform,
                )

                D1_1 = (1.0 / r_prev2) * (x_prediction_prev - x_prediction_p2)
                D1 = D1_0 + (r / (r + r_prev2)) * (D1_0 - D1_1)
                D2 = (1.0 / (r + r_prev2)) * (D1_0 - D1_1)

                final -= (sigma_v_next * (exp2 / hh - 1.0)) * D1
                final -= (sigma_v_next * ((exp2 - hh) / hh**2 - 0.5)) * D2

            else:  # 2nd order. using this in O3 produces valid images but not going to risk correctness
                final -= (0.5 * sigma_v_next * exp2) * D1_0

        return final  # type: ignore


@dataclass(frozen=True)
class Adams(StatedSampler, StructuredMultistep, traits.DerivativeTransform):
    "Higher order extension to Euler using the Adams-Bashforth coefficients on the model prediction"

    @staticmethod
    def max_order() -> int:
        return 9

    def _sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
    ) -> T:
        effective_order = self.effective_order(packed.step, previous)
        delta = packed.delta_point(schedule)

        if self.derivative_transform:
            convert = models.ModelConvert(model_transform, self.derivative_transform)
            predictions = [
                convert.output_to(packed.sample, packed.prediction, delta.point_from.sigma, schedule.sigma_transform),
                *reversed(
                    [
                        convert.output_to(
                            p.sample,
                            p.prediction,
                            p.delta_point(schedule).point_from.sigma,
                            schedule.sigma_transform,
                        )
                        for p in previous[-effective_order + 1 :]
                    ]
                ),
            ]
            model_transform = convert.transform_to
        else:
            predictions = [packed.prediction, *reversed([p.prediction for p in previous[-effective_order + 1 :]])]

        weighted_prediction: T = math.sumprod(
            predictions[:effective_order],  # type: ignore
            common.bashforth(effective_order),
        )

        return model_transform.forward(
            packed.sample,
            weighted_prediction,
            delta.point_from.sigma,
            delta.point_to.sigma,
            schedule.sigma_transform,
        )


@dataclass(frozen=True)
class UniP(StatedSampler, StructuredMultistep):
    "Just the solver from UniPC without any correction stages."

    fast_solve: bool = False
    "Skip matrix solve for UniP-2 and UniC-1"

    @staticmethod
    def max_order() -> int:
        # TODO(beinsezii): seems more stable after converting to python scalars
        # 4-6 is mostly stable now, 7-9 depends on the model. What ranges are actually useful..?
        return 9

    def unisolve[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
        x_prediction_next: Sample | None = None,
    ) -> T:
        "Passing `prediction_next` is equivalent to UniC, otherwise behaves as UniP"

        delta = packed.delta_point(schedule)
        (sigma_u, sigma_v), (sigma_u_next, sigma_v_next) = delta.uv(schedule.sigma_transform)

        lambda_ = ln(divf(sigma_v, sigma_u))
        lambda_next = ln(divf(sigma_v_next, sigma_u_next))
        h = abs(lambda_next - lambda_)

        # hh = -h if self.predict_x0 else h
        hh_X = -h
        h_phi_1 = math.expm1(hh_X)  # h\phi_1(h) = e^h - 1

        # # bh1
        # B_h = hh
        # bh2
        B_h = h_phi_1

        x_prediction = model_transform.to_x(
            packed.sample,
            packed.prediction,
            delta.point_from.sigma,
            schedule.sigma_transform,
        )

        rks: list[float] = []
        D1s: list[Sample] = []
        effective_order = self.effective_order(packed.step, previous)
        for n in range(1, effective_order):
            sigma_prev_N = previous[-n].delta_point(schedule).point_from.sigma
            x_prediction_prev_N = model_transform.to_x(
                previous[-n].sample,
                previous[-n].prediction,
                sigma_prev_N,
                schedule.sigma_transform,
            )

            sigma_u_prev_N, sigma_v_prev_N = schedule.sigma_transform(sigma_prev_N)
            lambda_pO = ln(divf(sigma_v_prev_N, sigma_u_prev_N))
            rk = (lambda_pO - lambda_) / h
            if math.isfinite(rk):  # for subnormal
                rks.append(rk)
            else:
                rks.append(0)  # TODO(beinsezii): proper value?
            D1s.append((x_prediction_prev_N - x_prediction) / rk)

        # INFO(beinsezii): Fast solve from F.1 in paper
        if x_prediction_next is not None:
            rks.append(1.0)
            order_check: int = 1
        else:
            order_check = 2

        if not rks or (effective_order == order_check and self.fast_solve):
            rhos: list[float] = [0.5]
        else:
            h_phi_k = h_phi_1 / hh_X - 1
            R: list[list[float]] = []
            b: list[float] = []

            for n in range(1, len(rks) + 1):
                R.append([math.pow(v, n - 1) for v in rks])
                b.append(h_phi_k * math.factorial(n) / B_h)
                h_phi_k = h_phi_k / hh_X - 1 / math.factorial(n + 1)

            # small array order x order, fast to do it in just np
            rhos = np.linalg.solve(R, b).tolist()

        result = math.sumprod(rhos[: len(D1s)], D1s)  # type: ignore  # Float

        # if self.predict_x0:
        x_t_ = sigma_u_next / sigma_u * packed.sample - sigma_v_next * h_phi_1 * x_prediction

        if x_prediction_next is not None:
            D1_t = x_prediction_next - x_prediction
            final = x_t_ - sigma_v_next * B_h * (result + rhos[-1] * D1_t)
        else:
            final = x_t_ - sigma_v_next * B_h * result

        # else:
        #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        #     x_t = x_t_ - sigma_t * B_h * pred_res

        return final  # type: ignore

    def _sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
    ) -> T:
        return self.unisolve(packed, model_transform, schedule, previous)


@dataclass(frozen=True)
class UniPC(UniP, traits.DerivativeTransform):
    """Unique sampler that can correct other samplers or its own prediction function.
    The additional correction essentially adds +1 order on top of what is set.
    https://arxiv.org/abs/2302.04867"""

    solver: StructuredSampler | None = None
    "If not set, defaults to `UniSolver(order=self.order)`"

    @staticmethod
    def max_order() -> int:
        # TODO(beinsezii): seems more stable after converting to python scalars
        # 4-6 is mostly stable now, 7-9 depends on the model. What ranges are actually useful..?
        return 9

    @property
    def require_noise(self) -> bool:
        return self.solver.require_noise if self.solver else False

    @property
    def require_previous(self) -> int:
        # +1 for correction
        return max(super().require_previous + 1, self.solver.require_previous if self.solver else 0)

    def sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]] = (),
    ) -> SKSamples[T]:
        delta = packed.delta_point(schedule)

        if self.derivative_transform:
            convert = models.ModelConvert(model_transform, self.derivative_transform)
            packed = replace(
                packed,
                prediction=convert.output_to(
                    packed.sample,
                    packed.prediction,
                    delta.point_from.sigma,
                    schedule.sigma_transform,
                ),
            )
            model_transform = convert.transform_to

        if previous:
            packed = replace(
                packed,
                sample=self.unisolve(
                    previous[-1],
                    model_transform,
                    schedule,
                    previous[:-1],
                    x_prediction_next=model_transform.to_x(
                        packed.sample,
                        packed.prediction,
                        delta.point_from.sigma,
                        schedule.sigma_transform,
                    ),
                ),
            )

        return (self.solver or super()).sample_packed(packed, model_transform, schedule, previous)


@dataclass(frozen=True)
class SPC(StructuredSampler, traits.DerivativeTransform):
    """Simple predictor-corrector.
    Uses basic blended correction against the previous sample."""

    predictor: StructuredSampler = Euler()
    "Sampler for the current step"
    corrector: StructuredSampler = Adams(order=4)
    "Sampler to correct the previous step"

    bias: float = 0
    "Lower for more prediction, higher for more correction"
    power: float = 1
    "Scale the predicted and corrected samples before blending"
    adaptive: bool = True
    "Weight the predcition/correction ratio based on the sigma schedule"
    invert: bool = False
    "Invert the prediction/correction ratios"

    @property
    def require_noise(self) -> bool:
        return self.predictor.require_noise or self.corrector.require_noise

    @property
    def require_previous(self) -> int:
        return max(self.predictor.require_previous, self.corrector.require_previous + 1)

    def sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]] = (),
    ) -> SKSamples[T]:
        delta = packed.delta_point(schedule)

        if self.derivative_transform:
            convert = models.ModelConvert(model_transform, self.derivative_transform)
            packed = replace(
                packed,
                prediction=convert.output_to(
                    packed.sample,
                    packed.prediction,
                    delta.point_from.sigma,
                    schedule.sigma_transform,
                ),
            )
            model_transform = convert.transform_to

        if previous:
            offset_previous: list[SKSamples[T]] = [
                replace(p, prediction=pred)
                for p, pred in zip(previous, (*(p.prediction for p in previous[1:]), packed.prediction), strict=True)
            ]

            corrected = self.corrector.sample_packed(
                offset_previous.pop(),
                model_transform,
                schedule,
                offset_previous,
            ).final

            if self.adaptive:
                p, c = delta.uv(schedule.sigma_transform).uv_from
            else:
                p, c = 0, 0

            p, c = softmax((p - self.bias, c + self.bias))

            if self.invert:
                p, c = c, p

            if abs(self.power - 1) > 1e-8:  # short circuit because spowf is expensive
                sample = spowf(
                    spowf(packed.sample, self.power) * p + spowf(corrected, self.power) * c,
                    1 / self.power,
                )
            else:
                sample = packed.sample * p + corrected * c

            packed = replace(packed, sample=sample)

        return self.predictor.sample_packed(packed, model_transform, schedule, previous)
