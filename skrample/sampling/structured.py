import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from skrample import common
from skrample.common import DeltaPoint, Point, Sample, Step, divf, ln, softmax, spowf
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

    def delta_uv(self, schedule: SkrampleSchedule) -> DeltaPoint:
        return self.delta_point(schedule)


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

    def scale_input[T: Sample](self, sample: T, point: Point) -> T:
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
            **(  # ty: ignore # ???
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
class StructuredMultistep(traits.HigherOrder, StructuredSampler):
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
class StructuredStochastic(traits.Stochastic, StructuredSampler):
    @property
    def require_noise(self) -> bool:
        return abs(self.stochasticity) > 1e-8


@dataclass(frozen=True)
class StructuredUnified(traits.UnifiedModelling, StructuredStochastic, StructuredMultistep): ...


@dataclass(frozen=True)
class Euler(StructuredStochastic, StatedSampler):
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
            delta.point_from,
            delta.point_to,
            packed.noise,
            self.stochasticity,
        )


@dataclass(frozen=True)
class DPM(StructuredUnified, StatedSampler):
    """Good sampler, supports basically everything. Recommended default.

    https://arxiv.org/abs/2211.01095
    Page 4 Algo 2 for order=2
    Section 5 for SDE"""

    @staticmethod
    def max_order() -> int:
        return 3

    def _sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
    ) -> T:
        delta = packed.delta_point(schedule)

        effective_order = self.effective_order(packed.step, previous)

        # Convert prediction to derivative space
        if self.derivative_transform:
            convert = models.ModelConvert(model_transform, self.derivative_transform)
            predictions = [
                convert.output_to(packed.sample, packed.prediction, delta.point_from),
                *reversed(
                    [
                        convert.output_to(p.sample, p.prediction, p.delta_point(schedule).point_from)
                        for p in previous[-effective_order + 1 :]
                    ]
                ),
            ]
            model_transform = convert.transform_to
        else:
            predictions = [packed.prediction, *reversed([p.prediction for p in previous[-effective_order + 1 :]])]

        prediction = predictions.pop(0)

        # Initial implementation by diffusers.DPMSolverMultiStepScheduler,
        # rewritten for skrample by myself,
        # adjusted for DiffusionModel by Qwen 3.5,
        # everything since once again by myself
        if effective_order >= 2:
            # T0
            (_t0, sigma_u, sigma_v), (_t1, sigma_u_next, sigma_v_next) = delta

            lambda_ = ln(divf(sigma_v, sigma_u))
            lambda_next = ln(divf(sigma_v_next, sigma_u_next))
            h = abs(lambda_next - lambda_)

            # T-1
            point_prev = schedule.ipoint(previous[-1].step.time_from)
            _t_prev, sigma_u_prev, sigma_v_prev = point_prev
            lambda_prev = ln(divf(sigma_v_prev, sigma_u_prev))
            h_prev = lambda_ - lambda_prev
            r = h_prev / h

            prediction_prev = predictions.pop(0)
            D1_0 = (1.0 / r) * (prediction - prediction_prev)

            if effective_order >= 3:
                # T-2
                point_prev2 = schedule.ipoint(previous[-2].step.time_from)
                _t_prev2, sigma_u_prev2, sigma_v_prev2 = point_prev2
                lambda_prev2 = ln(divf(sigma_v_prev2, sigma_u_prev2))
                h_prev2 = lambda_prev - lambda_prev2
                r_prev2 = h_prev2 / h

                prediction_p2 = predictions.pop(0)

                D1_1 = (1.0 / r_prev2) * (prediction_prev - prediction_p2)
                D1 = D1_0 + (r / (r + r_prev2)) * (D1_0 - D1_1)
                D2 = (1.0 / (r + r_prev2)) * (D1_0 - D1_1)

                # Absorb corrections into prediction
                # Original: final -= sigma_v_next * exp2 * prediction
                #           final -= sigma_v_next * (exp2/hh - 1) * D1
                #           final -= sigma_v_next * ((exp2-hh)/hh^2 - 0.5) * D2
                # To absorb: correction = (exp2/hh - 1)/exp2 * D1 + ((exp2-hh)/hh^2 - 0.5)/exp2 * D2
                hh = -h
                exp2 = math.expm1(hh)
                c1 = (exp2 / hh - 1.0) / exp2 if exp2 != 0 else 0
                c2 = ((exp2 - hh) / hh**2 - 0.5) / exp2 if exp2 != 0 else 0
                prediction: T = prediction + c1 * D1 + c2 * D2  # pyright: ignore [reportAssignmentType] # float RHS is always T
            else:  # 2nd order
                # Absorb correction into prediction
                # Original: final -= sigma_v_next * exp2 * prediction
                #           final -= sigma_v_next * 0.5 * exp2 * D1_0
                # To absorb: correction = 0.5 * D1_0
                prediction: T = prediction + 0.5 * D1_0  # pyright: ignore [reportAssignmentType] # float RHS is always T

        return model_transform.forward(
            packed.sample,
            prediction,
            delta.point_from,
            delta.point_to,
            packed.noise,
            eta=self.stochasticity,
        )


@dataclass(frozen=True)
class Adams(StructuredUnified, StatedSampler):
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
                convert.output_to(packed.sample, packed.prediction, delta.point_from),
                *reversed(
                    [
                        convert.output_to(p.sample, p.prediction, p.delta_point(schedule).point_from)
                        for p in previous[-effective_order + 1 :]
                    ]
                ),
            ]
            model_transform = convert.transform_to
        else:
            predictions = [packed.prediction, *reversed([p.prediction for p in previous[-effective_order + 1 :]])]

        weighted_prediction: T = math.sumprod(  # ty: ignore # sumprod is T
            predictions[:effective_order],  # pyright: ignore # sumprod is T
            common.bashforth(effective_order),
        )

        return model_transform.forward(
            packed.sample,
            weighted_prediction,
            delta.point_from,
            delta.point_to,
            packed.noise,
            self.stochasticity,
        )


@dataclass(frozen=True)
class UniP(StructuredUnified, StatedSampler):
    "Just the solver from UniPC without any correction stages."

    fast_solve: bool = False
    "Skip matrix solve for UniP-2 and UniC-1"

    @staticmethod
    def max_order() -> int:
        return 9

    def unisolve[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
        prediction_next: Sample | None = None,
    ) -> T:
        "Passing `prediction_next` is equivalent to UniC, otherwise behaves as UniP"
        delta = packed.delta_point(schedule)

        effective_order = self.effective_order(packed.step, previous)
        if self.derivative_transform:
            convert = models.ModelConvert(model_transform, self.derivative_transform)
            predictions = [
                convert.output_to(packed.sample, packed.prediction, delta.point_from),
                *reversed(
                    [
                        convert.output_to(p.sample, p.prediction, p.delta_point(schedule).point_from)
                        for p in previous[-effective_order + 1 :]
                    ]
                ),
            ]
            if prediction_next is not None:
                prediction_next = convert.output_to(packed.sample, prediction_next, delta.point_from)
            model_transform = convert.transform_to
        else:
            predictions = [packed.prediction, *reversed([p.prediction for p in previous[-effective_order + 1 :]])]

        prediction = predictions.pop(0)

        # Initial implementation by diffusers.UniPCMultistepScheduler,
        # rewritten for skrample by myself,
        # adjusted for DiffusionModel by Qwen 3.5,
        # everything since once again by myself

        (_t0, sigma_u, sigma_v), (_t1, sigma_u_next, sigma_v_next) = delta

        lambda_ = ln(divf(sigma_v, sigma_u))
        lambda_next = ln(divf(sigma_v_next, sigma_u_next))
        h = abs(lambda_next - lambda_)

        hh_X = -h
        h_phi_1 = math.expm1(hh_X)
        B_h = h_phi_1

        rks: list[float] = []
        D1s: list[Sample] = []
        for n in range(1, effective_order):
            prediction_prev_N = predictions.pop(0)
            _tN, sigma_u_prev_N, sigma_v_prev_N = previous[-n].delta_point(schedule).point_from
            lambda_pO = ln(divf(sigma_v_prev_N, sigma_u_prev_N))
            rk = (lambda_pO - lambda_) / h
            if math.isfinite(rk):
                rks.append(rk)
            else:
                rks.append(0)
            D1s.append((prediction_prev_N - prediction) / rk)

        # Handle UniC correction term
        if prediction_next is not None:
            rks.append(1.0)
            order_check: int = 1
            D1s.append(prediction_next - prediction)
        else:
            order_check = 2

        # Fast solve from paper
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

            rhos = np.linalg.solve(R, b).tolist()

        result = math.sumprod(rhos[: len(D1s)], D1s)  # type: ignore  # Float

        prediction: T = prediction + result  # pyright: ignore [reportAssignmentType] # float RHS is always T

        return model_transform.forward(
            packed.sample,
            prediction,
            delta.point_from,
            delta.point_to,
            packed.noise,
            eta=self.stochasticity,
        )

    def _sample_packed[T: Sample](
        self,
        packed: SampleInput[T],
        model_transform: models.DiffusionModel,
        schedule: SkrampleSchedule,
        previous: Sequence[SKSamples[T]],
    ) -> T:
        return self.unisolve(packed, model_transform, schedule, previous)


@dataclass(frozen=True)
class UniPC(UniP):
    """Unique sampler that can correct other samplers or its own prediction function.
    The additional correction essentially adds +1 order on top of what is set.
    https://arxiv.org/abs/2302.04867"""

    solver: StructuredSampler | None = None
    "If not set, defaults to `UniSolver(order=self.order)`"

    @staticmethod
    def max_order() -> int:
        return 9

    @property
    def require_noise(self) -> bool:
        return super().require_noise or (self.solver.require_noise if self.solver else False)

    @property
    def require_previous(self) -> int:
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
                prediction=convert.output_to(packed.sample, packed.prediction, delta.point_from),
            )
            model_transform = convert.transform_to

        if previous:
            # Correct previous step using current prediction (UniC)
            corrected_sample = self.unisolve(
                previous[-1],
                model_transform,
                schedule,
                previous[:-1],
                prediction_next=packed.prediction,
            )
            packed = replace(packed, sample=corrected_sample)

        return (self.solver or super()).sample_packed(packed, model_transform, schedule, previous)


@dataclass(frozen=True)
class SPC(traits.DerivativeTransform, StructuredSampler):
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
                prediction=convert.output_to(packed.sample, packed.prediction, delta.point_from),
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
                _t, p, c = delta.point_from
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
