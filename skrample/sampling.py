import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from skrample.common import Sample, SigmaTransform, bashforth, safe_log, softmax, spowf


@dataclass(frozen=True)
class SKSamples[T: Sample]:
    """Sampler result struct for easy management of multiple sampling stages.
    This should be accumulated in a list for the denoising loop in order to use higher order features"""

    final: T
    "Final result. What you probably want"

    prediction: T
    "The model prediction"

    sample: T
    "The model input"

    noise: T | None = None
    "The extra stochastic noise"


@dataclass(frozen=True)
class SkrampleSampler(ABC):
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

    @staticmethod
    def get_sigma(step: int, sigma_schedule: NDArray) -> float:
        "Just returns zero if step > len"
        return sigma_schedule[step].item() if step < len(sigma_schedule) else 0

    @abstractmethod
    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        """sigma_schedule is just the sigmas, IE SkrampleSchedule()[:, 1].

        `sigma_transform` is a function for mapping an arbitrary sigma to more normalized coordinates.
        Typically this is `sigma_complement` for flow models, othersie `sigma_polar`.
        All SkrampleSchedules contain a `.sigma_transform` property with this defined.

        `noise` is noise specific to this step for StochasticSampler or other schedulers that compute against noise.
        This is NOT the input noise, which is added directly into the sample with `merge_noise()`

        """

    def scale_input[T: Sample](self, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return sample

    def merge_noise[T: Sample](self, sample: T, noise: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_u, sigma_v = sigma_transform(sigma)
        return sample * sigma_v + noise * sigma_u  # type: ignore

    def __call__[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        return self.sample(
            sample=sample,
            prediction=prediction,
            step=step,
            sigma_schedule=sigma_schedule,
            sigma_transform=sigma_transform,
            noise=noise,
            previous=previous,
        )


@dataclass(frozen=True)
class HighOrderSampler(SkrampleSampler):
    """Samplers inheriting this trait support order > 1, and will require
    `prevous` be managed and passed to function accordingly."""

    order: int = 1

    @staticmethod
    def min_order() -> int:
        return 1

    @staticmethod
    @abstractmethod
    def max_order() -> int:
        pass

    @property
    def require_previous(self) -> int:
        return max(min(self.order, self.max_order()), self.min_order()) - 1

    def effective_order(self, step: int, schedule: NDArray, previous: tuple[SKSamples, ...]) -> int:
        "The order used in calculation given a step, schedule length, and previous sample count"
        return max(
            1,  # not min_order because previous may be < min. samplers should check effective >= min
            min(
                self.max_order(),
                step + 1,
                self.order,
                len(previous) + 1,
                len(schedule) - step,  # lower for final is the default
            ),
        )


@dataclass(frozen=True)
class StochasticSampler(SkrampleSampler):
    add_noise: bool = False
    "Flag for whether or not to add the given noise"

    @property
    def require_noise(self) -> bool:
        return self.add_noise


@dataclass(frozen=True)
class Euler(SkrampleSampler):
    """Basic sampler, the "safe" choice."""

    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_next = self.get_sigma(step + 1, sigma_schedule)

        sigma_u, sigma_v = sigma_transform(sigma)
        sigma_u_next, sigma_v_next = sigma_transform(sigma_next)

        scale = sigma_u_next / sigma_u
        delta = sigma_v_next - sigma_v * scale  # aka `h` or `dt`
        final = sample * scale + prediction * delta

        return SKSamples(  # type: ignore
            final=final,
            prediction=prediction,
            sample=sample,
        )


@dataclass(frozen=True)
class DPM(HighOrderSampler, StochasticSampler):
    """Good sampler, supports basically everything. Recommended default.

    https://arxiv.org/abs/2211.01095
    Page 4 Algo 2 for order=2
    Section 5 for SDE"""

    @staticmethod
    def max_order() -> int:
        return 3  # TODO(beinsezii): 3, 4+?

    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_next = self.get_sigma(step + 1, sigma_schedule)

        sigma_u, sigma_v = sigma_transform(sigma)
        sigma_u_next, sigma_v_next = sigma_transform(sigma_next)

        lambda_ = safe_log(sigma_v) - safe_log(sigma_u)
        lambda_next = safe_log(sigma_v_next) - safe_log(sigma_u_next)
        h = abs(lambda_next - lambda_)

        if noise is not None and self.add_noise:
            exp1 = math.exp(-h)
            hh = -2 * h
            noise_factor = sigma_u_next * math.sqrt(1 - math.exp(hh)) * noise
        else:
            exp1 = 1
            hh = -h
            noise_factor = 0

        exp2 = math.expm1(hh)

        final = noise_factor + (sigma_u_next / sigma_u * exp1) * sample

        # 1st order
        final -= (sigma_v_next * exp2) * prediction

        effective_order = self.effective_order(step, sigma_schedule, previous)

        if effective_order >= 2:
            sigma_prev = self.get_sigma(step - 1, sigma_schedule)
            sigma_u_prev, sigma_v_prev = sigma_transform(sigma_prev)

            lambda_prev = safe_log(sigma_v_prev) - safe_log(sigma_u_prev)
            h_prev = lambda_ - lambda_prev
            r = h_prev / h  # math people and their var names...

            # Calculate previous predicton from sample, output
            prediction_prev = previous[-1].prediction
            D1_0 = (1.0 / r) * (prediction - prediction_prev)

            if effective_order >= 3:
                sigma_prev2 = self.get_sigma(step - 2, sigma_schedule)
                sigma_u_prev2, sigma_v_prev2 = sigma_transform(sigma_prev2)
                lambda_prev2 = safe_log(sigma_v_prev2) - safe_log(sigma_u_prev2)
                h_prev2 = lambda_prev - lambda_prev2
                r_prev2 = h_prev2 / h

                prediction_p2 = previous[-2].prediction

                D1_1 = (1.0 / r_prev2) * (prediction_prev - prediction_p2)
                D1 = D1_0 + (r / (r + r_prev2)) * (D1_0 - D1_1)
                D2 = (1.0 / (r + r_prev2)) * (D1_0 - D1_1)

                final -= (sigma_v_next * (exp2 / hh - 1.0)) * D1
                final -= (sigma_v_next * ((exp2 - hh) / hh**2 - 0.5)) * D2

            else:  # 2nd order. using this in O3 produces valid images but not going to risk correctness
                final -= (0.5 * sigma_v_next * exp2) * D1_0

        return SKSamples(  # type: ignore
            final=final,
            prediction=prediction,
            sample=sample,
        )


@dataclass(frozen=True)
class Adams(HighOrderSampler, Euler):
    "Higher order extension to Euler using the Adams-Bashforth coefficients on the model prediction"

    order: int = 2

    @staticmethod
    def max_order() -> int:
        return 9

    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        effective_order = self.effective_order(step, sigma_schedule, previous)

        predictions = [prediction, *reversed([p.prediction for p in previous[-effective_order + 1 :]])]
        weighted_prediction: T = math.sumprod(
            predictions[:effective_order],  # type: ignore
            bashforth(effective_order),
        )

        return replace(
            super().sample(sample, weighted_prediction, step, sigma_schedule, sigma_transform, noise, previous),
            prediction=prediction,
        )


@dataclass(frozen=True)
class UniP(HighOrderSampler):
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
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
        prediction_next: Sample | None = None,
    ) -> T:
        "Passing `prediction_next` is equivalent to UniC, otherwise behaves as UniP"
        sigma = self.get_sigma(step, sigma_schedule)

        sigma = self.get_sigma(step, sigma_schedule)
        sigma_u, sigma_v = sigma_transform(sigma)
        lambda_ = safe_log(sigma_v) - safe_log(sigma_u)

        effective_order = self.effective_order(step, sigma_schedule, previous)

        sigma_next = self.get_sigma(step + 1, sigma_schedule)
        sigma_u_next, sigma_v_next = sigma_transform(sigma_next)
        lambda_next = safe_log(sigma_v_next) - safe_log(sigma_u_next)
        h = abs(lambda_next - lambda_)

        # hh = -h if self.predict_x0 else h
        hh_X = -h
        h_phi_1 = math.expm1(hh_X)  # h\phi_1(h) = e^h - 1

        # # bh1
        # B_h = hh
        # bh2
        B_h = h_phi_1

        rks: list[float] = []
        D1s: list[Sample] = []
        for n in range(1, effective_order):
            step_prev_N = step - n
            prediction_prev_N = previous[-n].prediction
            sigma_u_prev_N, sigma_v_prev_N = sigma_transform(self.get_sigma(step_prev_N, sigma_schedule))
            lambda_pO = safe_log(sigma_v_prev_N) - safe_log(sigma_u_prev_N)
            rk = (lambda_pO - lambda_) / h
            if math.isfinite(rk):  # for subnormal
                rks.append(rk)
            else:
                rks.append(0)  # TODO(beinsezii): proper value?
            D1s.append((prediction_prev_N - prediction) / rk)

        # INFO(beinsezii): Fast solve from F.1 in paper
        if prediction_next is not None:
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
        x_t_ = sigma_u_next / sigma_u * sample - sigma_v_next * h_phi_1 * prediction

        if prediction_next is not None:
            D1_t = prediction_next - prediction
            final = x_t_ - sigma_v_next * B_h * (result + rhos[-1] * D1_t)
        else:
            final = x_t_ - sigma_v_next * B_h * result

        # else:
        #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        #     x_t = x_t_ - sigma_t * B_h * pred_res

        return final  # type: ignore

    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        return SKSamples(  # type: ignore
            final=self.unisolve(sample, prediction, step, sigma_schedule, sigma_transform, noise, previous),
            prediction=prediction,
            sample=sample,
        )


@dataclass(frozen=True)
class UniPC(UniP):
    """Unique sampler that can correct other samplers or its own prediction function.
    The additional correction essentially adds +1 order on top of what is set.
    https://arxiv.org/abs/2302.04867"""

    solver: SkrampleSampler | None = None
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

    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        if previous:
            sample = self.unisolve(
                previous[-1].sample,
                previous[-1].prediction,
                step - 1,
                sigma_schedule,
                sigma_transform,
                noise,
                previous[:-1],
                prediction_next=prediction,
            )

        return (self.solver if self.solver else super()).sample(
            sample,
            prediction,
            step,
            sigma_schedule,
            sigma_transform,
            noise,
            previous,
        )


@dataclass(frozen=True)
class SPC(SkrampleSampler):
    """Simple predictor-corrector.
    Uses basic blended correction against the previous sample."""

    predictor: SkrampleSampler = Euler()
    "Sampler for the current step"
    corrector: SkrampleSampler = Adams(order=4)
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

    def sample[T: Sample](
        self,
        sample: T,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: tuple[SKSamples[T], ...] = (),
    ) -> SKSamples[T]:
        if previous:
            offset_previous = tuple(
                replace(p, prediction=pred)
                for p, pred in zip(previous, (*(p.prediction for p in previous[1:]), prediction), strict=True)
            )
            prior = offset_previous[-1]

            corrected = self.corrector.sample(
                prior.sample,
                prior.prediction,
                step - 1,
                sigma_schedule,
                sigma_transform,
                prior.noise,
                offset_previous[:-1],
            ).final

            if self.adaptive:
                p, c = sigma_transform(self.get_sigma(step, sigma_schedule))
            else:
                p, c = 0, 0

            p, c = softmax((p - self.bias, c + self.bias))

            if self.invert:
                p, c = c, p

            if abs(self.power - 1) > 1e-8:  # short circuit because spowf is expensive
                sample = spowf(
                    spowf(sample, self.power) * p + spowf(corrected, self.power) * c,
                    1 / self.power,
                )  # type: ignore
            else:
                sample = sample * p + corrected * c  # type: ignore

        return replace(
            self.predictor.sample(sample, prediction, step, sigma_schedule, sigma_transform, noise, previous),
            noise=noise,  # the corrector may or may not need noise so we always store
        )
