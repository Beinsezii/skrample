import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skrample.common import Predictor, Sample, SigmaTransform, safe_log

# Just hardcode 5 because >=6 is 100% unusable
ADAMS_BASHFORTH_COEFFICIENTS: tuple[tuple[float, ...], ...] = (
    (1,),
    (3 / 2, -1 / 2),
    (23 / 12, -4 / 3, 5 / 12),
    (55 / 24, -59 / 24, 37 / 24, -3 / 8),
    (1901 / 720, -1387 / 360, 109 / 30, -637 / 360, 251 / 720),
)


def EPSILON[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "If a model does not specify, this is usually what it needs."
    u, v = sigma_transform(sigma)
    return (sample - u * output) / v  # type: ignore


def SAMPLE[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "No prediction. Only for single step afaik."
    return output


def VELOCITY[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "Rare, models will usually explicitly say they require velocity/vpred/zero terminal SNR"
    u, v = sigma_transform(sigma)
    return v * sample - u * output  # type: ignore


def FLOW[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "Flow matching models use this, notably FLUX.1 and SD3"
    # TODO(beinsezii): this might need to be u * output. Don't trust diffusers
    # Our tests will fail if we do so, leaving here for now.
    return sample - sigma * output  # type: ignore


@dataclass(frozen=True)
class SKSamples[T: Sample]:
    """Sampler result struct for easy management of multiple sampling stages.
    This should be accumulated in a list for the denoising loop in order to use higher order features"""

    final: T
    "Final result. What you probably want"

    prediction: T
    "Just the prediction from SkrampleSampler.predictor if it's used"

    sample: T
    "The unmodified model input"


@dataclass(frozen=True)
class SkrampleSampler(ABC):
    """Generic sampler structure with basic configurables and a stateless design.
    Abstract class not to be used directly.

    Unless otherwise specified, the Sample type is a stand-in that is
    type checked against torch.Tensor but should be generic enough to use with ndarrays or even raw floats"""

    predictor: Predictor = EPSILON
    "Predictor function. Most models are EPSILON, FLUX/SD3 are FLOW, VELOCITY and SAMPLE are rare."

    @staticmethod
    def get_sigma(step: int, sigma_schedule: NDArray) -> float:
        "Just returns zero if step > len"
        return sigma_schedule[step].item() if step < len(sigma_schedule) else 0

    @abstractmethod
    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
    ) -> SKSamples[T]:
        """sigma_schedule is just the sigmas, IE SkrampleSchedule()[:, 1].

        `noise` is noise specific to this step for StochasticSampler or other schedulers that compute against noise.
        This is NOT the input noise, which is added directly into the sample with `merge_noise()`

        `subnormal` is whether or not the noise schedule is all <= 1.0, IE Flow.
        All SkrampleSchedules contain a `.subnormal` property with this defined.
        """

    def scale_input[T: Sample](self, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return sample

    def merge_noise[T: Sample](self, sample: T, noise: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        u, v = sigma_transform(sigma)
        return sample * v + noise * u  # type: ignore

    def __call__[T: Sample](
        self,
        sample: T,
        output: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
    ) -> SKSamples[T]:
        return self.sample(
            sample=sample,
            output=output,
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

    @property
    def min_order(self) -> int:
        return 1

    @property
    @abstractmethod
    def max_order(self) -> int:
        pass

    def effective_order(self, step: int, schedule: NDArray, previous: list[SKSamples]) -> int:
        "The order used in calculation given a step, schedule length, and previous sample count"
        return max(
            self.min_order,
            min(
                self.max_order,
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


@dataclass(frozen=True)
class Euler(SkrampleSampler):
    """Basic sampler, the "safe" choice."""

    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_next = self.get_sigma(step + 1, sigma_schedule)

        u, v = sigma_transform(sigma)
        u_next, v_next = sigma_transform(sigma_next)

        prediction: T = self.predictor(sample, output, sigma, sigma_transform)  # type: ignore

        try:
            ratio = u_next / sigma_next
        except ZeroDivisionError:
            ratio = 1

        # thx Qwen
        term1 = (sample * sigma) / u
        term2 = (term1 - prediction) * (sigma_next / sigma - 1)
        sampled = (term1 + term2) * ratio

        return SKSamples(  # type: ignore
            final=sampled,
            prediction=prediction,
            sample=sample,
        )


@dataclass(frozen=True)
class DPM(HighOrderSampler, StochasticSampler):
    """Good sampler, supports basically everything. Recommended default.

    https://arxiv.org/abs/2211.01095
    Page 4 Algo 2 for order=2
    Section 5 for SDE"""

    @property
    def max_order(self) -> int:
        return 3  # TODO(beinsezii): 3, 4+?

    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_next = self.get_sigma(step + 1, sigma_schedule)

        u, v = sigma_transform(sigma)
        u_next, v_next = sigma_transform(sigma_next)

        lambda_ = safe_log(v) - safe_log(u)
        lambda_next = safe_log(v_next) - safe_log(u_next)
        h = abs(lambda_next - lambda_)

        if noise is not None and self.add_noise:
            exp1 = math.exp(-h)
            hh = -2 * h
            noise_factor = u_next * math.sqrt(1 - math.exp(hh)) * noise
        else:
            exp1 = 1
            hh = -h
            noise_factor = 0

        exp2 = math.expm1(hh)

        prediction: T = self.predictor(sample, output, sigma, sigma_transform)  # type: ignore

        sampled = noise_factor + (u_next / u * exp1) * sample

        # 1st order
        sampled -= (v_next * exp2) * prediction

        effective_order = self.effective_order(step, sigma_schedule, previous)

        if effective_order >= 2:
            sigma_prev = self.get_sigma(step - 1, sigma_schedule)
            u_prev, v_prev = sigma_transform(sigma_prev)

            lambda_prev = safe_log(v_prev) - safe_log(u_prev)
            h_prev = lambda_ - lambda_prev
            r = h_prev / h  # math people and their var names...

            # Calculate previous predicton from sample, output
            prediction_prev = previous[-1].prediction
            D1_0 = (1.0 / r) * (prediction - prediction_prev)

            if effective_order >= 3:
                sigma_prev2 = self.get_sigma(step - 2, sigma_schedule)
                u_prev2, v_prev2 = sigma_transform(sigma_prev2)
                lambda_prev2 = safe_log(v_prev2) - safe_log(u_prev2)
                h_prev2 = lambda_prev - lambda_prev2
                r_prev2 = h_prev2 / h

                prediction_p2 = previous[-2].prediction

                D1_1 = (1.0 / r_prev2) * (prediction_prev - prediction_p2)
                D1 = D1_0 + (r / (r + r_prev2)) * (D1_0 - D1_1)
                D2 = (1.0 / (r + r_prev2)) * (D1_0 - D1_1)

                sampled -= (v_next * (exp2 / hh - 1.0)) * D1
                sampled -= (v_next * ((exp2 - hh) / hh**2 - 0.5)) * D2

            else:  # 2nd order. using this in O3 produces valid images but not going to risk correctness
                sampled -= (0.5 * v_next * exp2) * D1_0

        return SKSamples(  # type: ignore
            final=sampled,
            prediction=prediction,
            sample=sample,
        )


@dataclass(frozen=True)
class Adams(HighOrderSampler):
    "Higher order extension to Euler using the Adams-Bashforth coefficients on the model prediction"

    order: int = 2

    @property
    def max_order(self) -> int:
        return len(ADAMS_BASHFORTH_COEFFICIENTS)

    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_next = self.get_sigma(step + 1, sigma_schedule)

        u, v = sigma_transform(sigma)
        u_next, v_next = sigma_transform(sigma_next)

        effective_order = self.effective_order(step, sigma_schedule, previous)
        prediction: T = self.predictor(sample, output, self.get_sigma(step, sigma_schedule), sigma_transform)  # type: ignore

        predictions = [prediction, *reversed([p.prediction for p in previous[-effective_order + 1 :]])]
        weighted_prediction: T = math.sumprod(
            predictions[:effective_order],  # type: ignore
            ADAMS_BASHFORTH_COEFFICIENTS[effective_order - 1],
        )

        # Plain Euler from here out
        try:
            ratio = u_next / sigma_next
        except ZeroDivisionError:
            ratio = 1

        term1 = (sample * sigma) / u
        term2 = (term1 - weighted_prediction) * (sigma_next / sigma - 1)
        sampled = (term1 + term2) * ratio

        return SKSamples(
            final=sampled,
            prediction=prediction,
            sample=sample,
        )  # type: ignore


@dataclass(frozen=True)
class UniPC(HighOrderSampler):
    """Unique sampler that can correct other samplers or its own prediction function.
    The additional correction essentially adds +1 order on top of what is set."""

    solver: SkrampleSampler | None = None
    """If set, will use another sampler then perform its own correction.
    May break, particularly if the solver uses different scaling for noise or input."""

    @property
    def max_order(self) -> int:
        # TODO(beinsezii): seems more stable after converting to python scalars
        # 4-6 is mostly stable now, 7-9 depends on the model. What ranges are actually useful..?
        return 9

    def _uni_p_c_prelude[T: Sample](
        self,
        prediction: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        previous: list[SKSamples[T]],
        lambda_X: float,
        h_X: float,
        order: int,
        prior: bool,
    ) -> tuple[float, list[float], float | T, float]:
        "B_h, rhos, result, h_phi_1_X"
        # hh = -h if self.predict_x0 else h
        hh_X = -h_X
        h_phi_1_X = math.expm1(hh_X)  # h\phi_1(h) = e^h - 1

        # # bh1
        # B_h = hh
        # bh2
        B_h = h_phi_1_X

        rks: list[float] = []
        D1s: list[Sample] = []
        for n in range(1 + prior, order + prior):
            step_prev_N = step - n
            prediction_prev_N = previous[-n].prediction
            u_prev_N, v_prev_N = sigma_transform(self.get_sigma(step_prev_N, sigma_schedule))
            lambda_pO = safe_log(v_prev_N) - safe_log(u_prev_N)
            rk = (lambda_pO - lambda_X) / h_X
            if math.isfinite(rk):  # for subnormal
                rks.append(rk)
            else:
                rks.append(0)  # TODO(beinsezii): proper value?
            D1s.append((prediction_prev_N - prediction) / rk)

        if prior:
            rks.append(1.0)

        R: list[list[float]] = []
        b: list[float] = []

        h_phi_k = h_phi_1_X / hh_X - 1

        for n in range(1, order + 1):
            R.append([math.pow(v, n - 1) for v in rks])
            b.append(h_phi_k * math.factorial(n) / B_h)
            h_phi_k = h_phi_k / hh_X - 1 / math.factorial(n + 1)

        if order <= 2 - prior:
            rhos: list[float] = [0.5]
        else:
            # small array order x order, fast to do it in just np
            n = len(rks)
            rhos = np.linalg.solve(R[:n], b[:n]).tolist()  # type: ignore

        uni_res = math.sumprod(rhos[: len(D1s)], D1s)  # type: ignore  # Float

        return B_h, rhos, uni_res, h_phi_1_X

    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        step: int,
        sigma_schedule: NDArray,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        prediction: T = self.predictor(sample, output, sigma, sigma_transform)  # type: ignore

        sigma = self.get_sigma(step, sigma_schedule)
        u, v = sigma_transform(sigma)
        lambda_ = safe_log(v) - safe_log(u)

        if previous:
            # -1 step since it effectively corrects the prior step before the next prediction
            effective_order = self.effective_order(step - 1, sigma_schedule, previous[:-1])

            sigma_prev = self.get_sigma(step - 1, sigma_schedule)
            u_prev, v_prev = sigma_transform(sigma_prev)
            lambda_prev = safe_log(v_prev) - safe_log(u_prev)
            h_prev = abs(lambda_ - lambda_prev)

            prediction_prev = previous[-1].prediction
            sample_prev = previous[-1].sample

            B_h_prev, rhos_c, uni_c_res, h_phi_1_prev = self._uni_p_c_prelude(
                prediction_prev,
                step,
                sigma_schedule,
                sigma_transform,
                previous,
                lambda_prev,
                h_prev,
                effective_order,
                True,
            )

            # if self.predict_x0:
            x_t_ = u / u_prev * sample_prev - v * h_phi_1_prev * prediction_prev
            D1_t = prediction - prediction_prev
            sample = x_t_ - v * B_h_prev * (uni_c_res + rhos_c[-1] * D1_t)  # type: ignore
            # else:
            #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            #     D1_t = model_t - m0
            #     x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        if self.solver:
            sampled = self.solver.sample(sample, output, step, sigma_schedule, sigma_transform, noise, previous).final
        else:
            effective_order = self.effective_order(step, sigma_schedule, previous)

            sigma_next = self.get_sigma(step + 1, sigma_schedule)
            u_next, v_next = sigma_transform(sigma_next)
            lambda_next = safe_log(v_next) - safe_log(u_next)
            h = abs(lambda_next - lambda_)

            B_h, _, uni_p_res, h_phi_1 = self._uni_p_c_prelude(
                prediction, step, sigma_schedule, sigma_transform, previous, lambda_, h, effective_order, False
            )

            # if self.predict_x0:
            x_t_ = u_next / u * sample - v_next * h_phi_1 * prediction
            sampled = x_t_ - v_next * B_h * uni_p_res
            # else:
            #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            #     x_t = x_t_ - sigma_t * B_h * pred_res

        return SKSamples(  # type: ignore
            final=sampled,
            prediction=prediction,
            sample=sample,
        )
