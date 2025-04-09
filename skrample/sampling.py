import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from skrample.common import safe_log

if TYPE_CHECKING:
    from torch.types import Tensor

    Sample = float | NDArray[np.floating] | Tensor
else:
    # Avoid pulling all of torch as the code doesn't explicitly depend on it.
    Sample = float | NDArray[np.floating]


PREDICTOR = Callable[[Sample, Sample, float, bool], Sample]
"sample, output, sigma, subnormal"


def sigma_normal(sigma: float, subnormal: bool = False) -> tuple[float, float]:
    if subnormal:
        return sigma, 1 - sigma
    else:
        theta = math.atan(sigma)
        return math.sin(theta), math.cos(theta)


def EPSILON[T: Sample](sample: T, output: T, sigma: float, subnormal: bool = False) -> T:
    "If a model does not specify, this is usually what it needs."
    sigma, alpha = sigma_normal(sigma, subnormal)
    return (sample - sigma * output) / alpha  # type: ignore


def SAMPLE[T: Sample](sample: T, output: T, sigma: float, subnormal: bool = False) -> T:
    "No prediction. Only for single step afaik."
    return output


def VELOCITY[T: Sample](sample: T, output: T, sigma: float, subnormal: bool = False) -> T:
    "Rare, models will usually explicitly say they require velocity/vpred/zero terminal SNR"
    sigma, alpha = sigma_normal(sigma, subnormal)
    return alpha * sample - sigma * output  # type: ignore


def FLOW[T: Sample](sample: T, output: T, sigma: float, subnormal: bool = False) -> T:
    "Flow matching models use this, notably FLUX.1 and SD3"
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
    "An intermediate sample stage or input samples. Mostly for internal use by advanced samplers"


@dataclass
class SkrampleSampler(ABC):
    """Generic sampler structure with basic configurables and a stateless design.
    Abstract class not to be used directly.

    Unless otherwise specified, the Sample type is a stand-in that is
    type checked against torch.Tensor but should be generic enough to use with ndarrays or even raw floats"""

    predictor: PREDICTOR = EPSILON
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
        sigma_schedule: NDArray,
        step: int,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
        subnormal: bool = False,
    ) -> SKSamples[T]:
        """sigma_schedule is just the sigmas, IE SkrampleSchedule()[:, 1].

        `noise` is noise specific to this step for StochasticSampler or other schedulers that compute against noise.
        This is NOT the input noise, which is added directly into the sample with `merge_noise()`

        `subnormal` is whether or not the noise schedule is all <= 1.0, IE Flow.
        All SkrampleSchedules contain a `.subnormal` property with this defined.
        """

    def scale_input[T: Sample](self, sample: T, sigma: float, subnormal: bool = False) -> T:
        return sample

    def merge_noise[T: Sample](self, sample: T, noise: T, sigma: float, subnormal: bool = False) -> T:
        sigma, alpha = sigma_normal(sigma, subnormal)
        return sample * alpha + noise * sigma  # type: ignore

    def __call__[T: Sample](
        self,
        sample: T,
        output: T,
        sigma_schedule: NDArray,
        step: int,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
        subnormal: bool = False,
    ) -> SKSamples[T]:
        return self.sample(
            sample=sample,
            output=output,
            sigma_schedule=sigma_schedule,
            step=step,
            noise=noise,
            previous=previous,
            subnormal=subnormal,
        )


@dataclass
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


@dataclass
class StochasticSampler(SkrampleSampler):
    add_noise: bool = False
    "Flag for whether or not to add the given noise"


@dataclass
class Euler(StochasticSampler):
    """Basic sampler, the "safe" choice.
    Add noise for ancestral sampling."""

    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        sigma_schedule: NDArray,
        step: int,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
        subnormal: bool = False,
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_n1 = self.get_sigma(step + 1, sigma_schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        if self.add_noise and noise is not None:
            if subnormal:
                # TODO(beinsezii): correct values for Euler?
                # DPM is mathematically close enough that this is fine enough,
                # but ideally we wouldn't need this.
                # Maybe Euler in general should just alias DPM...
                return DPM(
                    predictor=self.predictor,
                    add_noise=self.add_noise,
                    order=1,
                ).sample(
                    sample=sample,
                    output=output,
                    sigma_schedule=sigma_schedule,
                    step=step,
                    noise=noise,
                    previous=previous,
                    subnormal=subnormal,
                )
            else:
                sigma_up = sigma / 2 * math.sin(math.asin(sigma_n1 / sigma) * 2)
                sigma_down = sigma_n1**2 / sigma

            noise_factor = noise * sigma_up
            sigma_factor = sigma_down
        else:
            noise_factor = 0
            sigma_factor = sigma_n1

        prediction: T = self.predictor(sample, output, sigma, subnormal)  # type: ignore

        try:
            ratio = signorm_n1 / sigma_n1
        except ZeroDivisionError:
            ratio = 1

        # thx Qwen
        term1 = (sample * sigma) / signorm
        term2 = (term1 - prediction) * (sigma_factor / sigma - 1)
        sampled = (term1 + term2 + noise_factor) * ratio

        return SKSamples(  # type: ignore
            final=sampled,
            prediction=prediction,
            sample=sample,
        )


@dataclass
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
        sigma_schedule: NDArray,
        step: int,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
        subnormal: bool = False,
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_n1 = self.get_sigma(step + 1, sigma_schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        lambda_ = safe_log(alpha) - safe_log(signorm)
        lambda_n1 = safe_log(alpha_n1) - safe_log(signorm_n1)
        h = abs(lambda_n1 - lambda_)

        if noise is not None and self.add_noise:
            exp1 = math.exp(-h)
            hh = -2 * h
            noise_factor = signorm_n1 * math.sqrt(1 - math.exp(hh)) * noise
        else:
            exp1 = 1
            hh = -h
            noise_factor = 0

        exp2 = math.expm1(hh)

        prediction: T = self.predictor(sample, output, sigma, subnormal)  # type: ignore

        sampled = noise_factor + (signorm_n1 / signorm * exp1) * sample

        # 1st order
        sampled -= (alpha_n1 * exp2) * prediction

        effective_order = self.effective_order(step, sigma_schedule, previous)

        if effective_order >= 2:
            sigma_p1 = self.get_sigma(step - 1, sigma_schedule)
            signorm_p1, alpha_p1 = sigma_normal(sigma_p1, subnormal)

            lambda_p1 = safe_log(alpha_p1) - safe_log(signorm_p1)
            h_p1 = lambda_ - lambda_p1
            r = h_p1 / h  # math people and their var names...

            # Calculate previous predicton from sample, output
            prediction_p1 = previous[-1].prediction
            D1_0 = (1.0 / r) * (prediction - prediction_p1)

            if effective_order >= 3:
                sigma_p2 = self.get_sigma(step - 2, sigma_schedule)
                signorm_p2, alpha_p2 = sigma_normal(sigma_p2, subnormal)
                lambda_p2 = safe_log(alpha_p2) - safe_log(signorm_p2)
                h_p2 = lambda_p1 - lambda_p2
                r_p2 = h_p2 / h

                prediction_p2 = previous[-2].prediction

                D1_1 = (1.0 / r_p2) * (prediction_p1 - prediction_p2)
                D1 = D1_0 + (r / (r + r_p2)) * (D1_0 - D1_1)
                D2 = (1.0 / (r + r_p2)) * (D1_0 - D1_1)

                sampled -= (alpha_n1 * (exp2 / hh - 1.0)) * D1
                sampled -= (alpha_n1 * ((exp2 - hh) / hh**2 - 0.5)) * D2

            else:  # 2nd order. using this in O3 produces valid images but not going to risk correctness
                sampled -= (0.5 * alpha_n1 * exp2) * D1_0

        return SKSamples(  # type: ignore
            final=sampled,
            prediction=prediction,
            sample=sample,
        )


@dataclass
class IPNDM(HighOrderSampler, Euler):
    """Higher order extension to Euler.
    Requires 4th order for optimal effect."""

    order: int = 4

    @property
    def max_order(self) -> int:
        return 4

    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        sigma_schedule: NDArray,
        step: int,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
        subnormal: bool = False,
    ) -> SKSamples[T]:
        effective_order = self.effective_order(step, sigma_schedule, previous)

        if effective_order >= 4:
            eps = (55 / 24 * output - 59 / 24 * previous[-1].sample) + (
                37 / 24 * previous[-2].sample - 9 / 24 * previous[-3].sample
            )
        elif effective_order >= 3:
            eps = 23 / 12 * output - 16 / 12 * previous[-1].sample + 5 / 12 * previous[-2].sample
        elif effective_order >= 2:
            eps = 3 / 2 * output - 1 / 2 * previous[-1].sample
        else:
            eps = output

        result = super().sample(sample, eps, sigma_schedule, step, noise, previous, subnormal)  # type: ignore
        return SKSamples(final=result.final, prediction=result.prediction, sample=output)  # type: ignore


@dataclass
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
        previous: list[SKSamples[T]],
        subnormal: bool,
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
        for i in range(1 + prior, order + prior):
            step_pO = step - i
            prediction_pO = previous[-i].prediction
            sigma_pO, alpha_pO = sigma_normal(self.get_sigma(step_pO, sigma_schedule), subnormal)
            lambda_pO = safe_log(alpha_pO) - safe_log(sigma_pO)
            rk = (lambda_pO - lambda_X) / h_X
            if math.isfinite(rk):  # for subnormal
                rks.append(rk)
            else:
                rks.append(0)  # TODO(beinsezii): proper value?
            D1s.append((prediction_pO - prediction) / rk)

        if prior:
            rks.append(1.0)

        R: list[list[float]] = []
        b: list[float] = []

        h_phi_k = h_phi_1_X / hh_X - 1

        for i in range(1, order + 1):
            R.append([math.pow(v, i - 1) for v in rks])
            b.append(h_phi_k * math.factorial(i) / B_h)
            h_phi_k = h_phi_k / hh_X - 1 / math.factorial(i + 1)

        if order <= 2 - prior:
            rhos: list[float] = [0.5]
        else:
            # small array order x order, fast to do it in just np
            i = len(rks)
            rhos = np.linalg.solve(R[:i], b[:i]).tolist()  # type: ignore

        uni_res = math.sumprod(rhos[: len(D1s)], D1s)  # type: ignore  # Float

        return B_h, rhos, uni_res, h_phi_1_X

    def sample[T: Sample](
        self,
        sample: T,
        output: T,
        sigma_schedule: NDArray,
        step: int,
        noise: T | None = None,
        previous: list[SKSamples[T]] = [],
        subnormal: bool = False,
    ) -> SKSamples[T]:
        sigma = self.get_sigma(step, sigma_schedule)
        prediction: T = self.predictor(sample, output, sigma, subnormal)  # type: ignore

        sigma = self.get_sigma(step, sigma_schedule)
        signorm, alpha = sigma_normal(sigma, subnormal)
        lambda_ = safe_log(alpha) - safe_log(signorm)

        if previous:
            # -1 step since it effectively corrects the prior step before the next prediction
            effective_order = self.effective_order(step - 1, sigma_schedule, previous[:-1])

            sigma_p1 = self.get_sigma(step - 1, sigma_schedule)
            signorm_p1, alpha_p1 = sigma_normal(sigma_p1, subnormal)
            lambda_p1 = safe_log(alpha_p1) - safe_log(signorm_p1)
            h_p1 = abs(lambda_ - lambda_p1)

            prediction_p1 = previous[-1].prediction
            sample_p1 = previous[-1].sample

            B_h_p1, rhos_c, uni_c_res, h_phi_1_p1 = self._uni_p_c_prelude(
                prediction_p1, step, sigma_schedule, previous, subnormal, lambda_p1, h_p1, effective_order, True
            )

            # if self.predict_x0:
            x_t_ = signorm / signorm_p1 * sample_p1 - alpha * h_phi_1_p1 * prediction_p1
            D1_t = prediction - prediction_p1
            sample = x_t_ - alpha * B_h_p1 * (uni_c_res + rhos_c[-1] * D1_t)  # type: ignore
            # else:
            #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            #     D1_t = model_t - m0
            #     x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        if self.solver:
            sampled = self.solver.sample(sample, output, sigma_schedule, step, noise, previous, subnormal).final
        else:
            effective_order = self.effective_order(step, sigma_schedule, previous)

            sigma_n1 = self.get_sigma(step + 1, sigma_schedule)
            signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)
            lambda_n1 = safe_log(alpha_n1) - safe_log(signorm_n1)
            h = abs(lambda_n1 - lambda_)

            B_h, _, uni_p_res, h_phi_1 = self._uni_p_c_prelude(
                prediction, step, sigma_schedule, previous, subnormal, lambda_, h, effective_order, False
            )

            # if self.predict_x0:
            x_t_ = signorm_n1 / signorm * sample - alpha_n1 * h_phi_1 * prediction
            sampled = x_t_ - alpha_n1 * B_h * uni_p_res
            # else:
            #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            #     x_t = x_t_ - sigma_t * B_h * pred_res

        return SKSamples(  # type: ignore
            final=sampled,
            prediction=prediction,
            sample=sample,
        )
