import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

    Sample = Tensor
else:
    # Avoid pulling all of torch as the code doesn't explicitly depend on it.
    Sample = float


PREDICTOR = Callable[[Sample, Sample, float, bool], Sample]


def safe_log(x: float) -> float:
    try:
        return math.log(x)
    except ValueError:
        return math.inf


def sigma_normal(sigma: float, subnormal: bool = False) -> tuple[float, float]:
    if subnormal:
        return sigma, 1 - sigma
    else:
        alpha = 1 / ((sigma**2 + 1) ** 0.5)
        return sigma * alpha, alpha


def EPSILON(sample: Sample, output: Sample, sigma: float, subnormal: bool = False) -> Sample:
    "If a model does not specify, this is usually what it needs."
    sigma, alpha = sigma_normal(sigma, subnormal)
    return (sample - sigma * output) / alpha


def SAMPLE(sample: Sample, output: Sample, sigma: float, subnormal: bool = False) -> Sample:
    "No prediction. Only for single step afaik."
    return output


def VELOCITY(sample: Sample, output: Sample, sigma: float, subnormal: bool = False) -> Sample:
    "Rare, models will usually explicitly say they require velocity/vpred/zero terminal SNR"
    sigma, alpha = sigma_normal(sigma, subnormal)
    return alpha * sample - sigma * output


def FLOW(sample: Sample, output: Sample, sigma: float, subnormal: bool = False) -> Sample:
    "Flow matching models use this, notably FLUX.1 and SD3"
    return sample - sigma * output


@dataclass(frozen=True)
class SKSamples:
    """Sampler result struct for easy management of multiple sampling stages.
    This should be accumulated in a list for the denoising loop in order to use higher order features"""

    final: Sample
    "Final result. What you probably want"

    prediction: Sample
    "Just the prediction from SkrampleSampler.predictor if it's used"

    sample: Sample
    "An intermediate sample stage or input samples. Mostly for internal use by advanced samplers"


@dataclass
class SkrampleSampler(ABC):
    """Generic sampler structure with basic configurables and a stateless design.
    Abstract class not to be used directly.

    Unless otherwise specified, the Sample type is a stand-in that is
    type checked against torch.Tensor but should be generic enough to use with ndarrays or even raw floats"""

    predictor: Callable[[Sample, Sample, float, bool], Sample] = EPSILON
    "Predictor function. Most models are EPSILON, FLUX/SD3 are FLOW, VELOCITY and SAMPLE are rare."

    @staticmethod
    def get_sigma(step: int, sigma_schedule: NDArray) -> float:
        "Just returns zero if step > len"
        return sigma_schedule[step].item() if step < len(sigma_schedule) else 0

    @abstractmethod
    def sample(
        self,
        sample: Sample,
        output: Sample,
        sigma_schedule: NDArray,
        step: int,
        noise: Sample | None = None,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        """sigma_schedule is just the sigmas, IE SkrampleSchedule()[:, 1].

        `noise` is noise specific to this step for StochasticSampler or other schedulers that compute against noise.
        This is NOT the input noise, which is added directly into the sample with `merge_noise()`

        `subnormal` is whether or not the noise schedule is all <= 1.0, IE Flow.
        All SkrampleSchedules contain a `.subnormal` property with this defined.
        """
        pass

    def scale_input(self, sample: Sample, sigma: float, subnormal: bool = False) -> Sample:
        return sample

    def merge_noise(self, sample: Sample, noise: Sample, sigma: float, subnormal: bool = False) -> Sample:
        sigma, alpha = sigma_normal(sigma, subnormal)
        return sample * alpha + noise * sigma

    def __call__(
        self,
        sample: Sample,
        output: Sample,
        sigma_schedule: NDArray,
        step: int,
        noise: Sample | None = None,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
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
class Euler(SkrampleSampler):
    """Basic sampler, the "safe" choice."""

    def sample(
        self,
        sample: Sample,
        output: Sample,
        sigma_schedule: NDArray,
        step: int,
        noise: Sample | None = None,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_n1 = self.get_sigma(step + 1, sigma_schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        prediction = self.predictor(sample, output, sigma, subnormal)

        # Dual branch *works* but blows up if sigma sigma_schedule is exactly `[1.0, 0.0]`
        # DPM works anyways so I'm not sure it's worth having a separate EulerFlow sampler just for that edge case
        if sigma_n1 == 0:  # get_sigma returns exact zero on +1 index
            # More accurate to how diffusers does it. / 0 on leading
            sampled = (sample + ((sample - prediction * alpha) / sigma) * (sigma_n1 - sigma)) * (alpha_n1 / alpha)
        else:
            # Moved / to signorm instead so / 0 is on trailing
            # Works but result is very slightly less accurate. Like +- 1e-14
            # thx Qwen
            sample = (sample * sigma) / signorm
            term2 = (sample - prediction) * (sigma_n1 / sigma - 1)
            sampled = (sample + term2) * (signorm_n1 / sigma_n1)

        return SKSamples(
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
        return 2  # TODO: 3, 4+?

    def sample(
        self,
        sample: Sample,
        output: Sample,
        sigma_schedule: NDArray,
        step: int,
        noise: Sample | None = None,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        sigma = self.get_sigma(step, sigma_schedule)
        sigma_n1 = self.get_sigma(step + 1, sigma_schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        lambda_ = safe_log(alpha) - safe_log(signorm)
        lambda_n1 = safe_log(alpha_n1) - safe_log(signorm_n1)
        h = abs(lambda_n1 - lambda_)

        if noise is not None and self.add_noise:
            exp1 = math.exp(-h)
            exp2 = 1.0 - math.exp(-2 * h)
            noise_factor = signorm_n1 * math.sqrt(exp2) * noise
        else:
            exp1 = 1
            exp2 = 1.0 - math.exp(-h)
            noise_factor = 0

        prediction = self.predictor(sample, output, sigma, subnormal)

        sampled = noise_factor + (signorm_n1 / signorm * exp1) * sample
        # 1st order
        sampled += (alpha_n1 * exp2) * prediction

        effective_order = self.effective_order(step, sigma_schedule, previous)

        if effective_order >= 2:
            sigma_p1 = self.get_sigma(step - 1, sigma_schedule)
            signorm_p1, alpha_p1 = sigma_normal(sigma_p1, subnormal)

            lambda_p1 = safe_log(alpha_p1) - safe_log(signorm_p1)
            h_p1 = lambda_ - lambda_p1
            r = h_p1 / h  # math people and their var names...

            # Calculate previous predicton from sample, output
            prediction_p1 = previous[-1].prediction
            prediction_p1 = (1.0 / r) * (prediction - prediction_p1)

            # 2nd order
            sampled += (0.5 * alpha_n1 * exp2) * prediction_p1

        return SKSamples(
            final=sampled,
            prediction=prediction,
            sample=sample,
        )


@dataclass
class UniPC(HighOrderSampler):
    """Unique sampler that can correct other samplers or its own prediction function.
    The additional correction essentially adds +1 order on top of what is set.

    Requires `torch`, inputs MUST be `torch.Tensor`"""

    solver: SkrampleSampler | None = None
    """If set, will use another sampler then perform its own correction.
    May break, particularly if the solver uses different scaling for noise or input."""

    @property
    def max_order(self) -> int:
        # TODO: seems more stable after converting to python scalars
        # 4-6 is mostly stable now, 7-9 depends on the model. What ranges are actually useful..?
        return 9

    # diffusers.sigma_schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.multistep_uni_c_bh_update
    def unified_corrector(
        self,
        sample: Sample,
        prediction: Sample,
        sigma_schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> Sample:
        import torch  # einsum operates on the full sample, so numpy would involve lots of data movement.

        # -1 step since it effectively corrects the prior step before the next prediction
        effective_order = self.effective_order(step - 1, sigma_schedule, previous[:-1])

        sigma = self.get_sigma(step, sigma_schedule)
        sigma_p1 = self.get_sigma(step - 1, sigma_schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_p1, alpha_p1 = sigma_normal(sigma_p1, subnormal)

        prediction_p1 = previous[-1].prediction
        sample_p1 = previous[-1].sample

        lambda_ = safe_log(alpha) - safe_log(signorm)
        lambda_p1 = safe_log(alpha_p1) - safe_log(signorm_p1)

        h = abs(lambda_ - lambda_p1)

        rks: list[float] = []
        D1s: list[Sample] = []
        for i in range(1, effective_order):
            step_pO1 = step - (i + 1)
            prediction_pO1 = previous[-(i + 1)].prediction
            sigma_pO1, alpha_pO1 = sigma_normal(self.get_sigma(step_pO1, sigma_schedule), subnormal)
            lambda_pO1 = safe_log(alpha_pO1) - safe_log(sigma_pO1)
            rk = (lambda_pO1 - lambda_p1) / h
            if math.isfinite(rk):  # for subnormal
                rks.append(rk)
            else:
                rks.append(0)  # TODO: proper value?

            D1s.append((prediction_pO1 - prediction_p1) / rk)

        rks.append(1.0)

        R: list[list[float]] = []
        b: list[float] = []

        # hh = -h if self.predict_x0 else h
        hh = -h
        h_phi_1 = math.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        # # BH1
        # B_h = hh
        # BH2
        B_h = math.expm1(hh)

        factorial_i = 1
        for i in range(1, effective_order + 1):
            R.append([math.pow(v, i - 1) for v in rks])
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        # for order 1, we use a simplified version
        if effective_order == 1:
            rhos_c = torch.tensor([0.5], dtype=sample_p1.dtype, device=sample.device)
        else:
            rhos_c = torch.linalg.solve(
                torch.tensor(R, dtype=sample.dtype, device=sample.device),
                torch.tensor(b, dtype=sample.dtype, device=sample.device),
            )

        if D1s:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], torch.stack(D1s, dim=1))
        else:
            corr_res = 0

        # if self.predict_x0:
        x_t_ = signorm / signorm_p1 * sample_p1 - alpha * h_phi_1 * prediction_p1
        D1_t = prediction - prediction_p1
        sample = x_t_ - alpha * B_h * (corr_res + rhos_c[-1] * D1_t)
        # else:
        #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        #     D1_t = model_t - m0
        #     x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return sample

    # diffusers.sigma_schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.multistep_uni_p_bh_update
    def unified_predictor(
        self,
        sample: Sample,
        prediction: Sample,
        sigma_schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> Sample:
        import torch  # einsum operates on the full sample, so numpy would involve lots of data movement.

        effective_order = self.effective_order(step, sigma_schedule, previous)

        sigma = self.get_sigma(step, sigma_schedule)
        sigma_n1 = self.get_sigma(step + 1, sigma_schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        lambda_n1 = safe_log(alpha_n1) - safe_log(signorm_n1)
        lambda_ = safe_log(alpha) - safe_log(signorm)

        h = abs(lambda_n1 - lambda_)

        rks: list[float] = []
        D1s: list[Sample] = []
        for i in range(1, effective_order):
            step_pO = step - i
            prediction_pO = previous[-i].prediction
            sigma_pO, alpha_pO = sigma_normal(self.get_sigma(step_pO, sigma_schedule), subnormal)
            lambda_pO = safe_log(alpha_pO) - safe_log(sigma_pO)
            rk = (lambda_pO - lambda_) / h
            if math.isfinite(rk):  # for subnormal
                rks.append(rk)
            else:
                rks.append(0)  # TODO: proper value?
            D1s.append((prediction_pO - prediction) / rk)

        rks.append(1.0)

        R: list[list[float]] = []
        b: list[float] = []

        # hh = -h if self.predict_x0 else h
        hh = -h
        h_phi_1 = math.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        # # bh1
        # B_h = hh
        # bh2
        B_h = math.expm1(hh)

        factorial_i = 1
        for i in range(1, effective_order + 1):
            R.append([math.pow(v, i - 1) for v in rks[:-1]])
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        if D1s:
            # for order 2, we use a simplified version
            if effective_order == 2:
                rhos_p = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
            else:
                rhos_p = torch.linalg.solve(
                    torch.tensor(R[:-1], device=sample.device, dtype=sample.dtype),
                    torch.tensor(b[:-1], device=sample.device, dtype=sample.dtype),
                )

            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, torch.stack(D1s, dim=1))
        else:
            pred_res = 0

        # if self.predict_x0:
        x_t_ = signorm_n1 / signorm * sample - alpha_n1 * h_phi_1 * prediction
        x_t = x_t_ - alpha_n1 * B_h * pred_res
        # else:
        #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        #     x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t

    def sample(
        self,
        sample: Sample,
        output: Sample,
        sigma_schedule: NDArray,
        step: int,
        noise: Sample | None = None,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        sigma = self.get_sigma(step, sigma_schedule)
        prediction = self.predictor(sample, output, sigma, subnormal)

        if previous:
            sample = self.unified_corrector(sample, prediction, sigma_schedule, step, previous, subnormal)

        if self.solver:
            sampled = self.solver.sample(sample, output, sigma_schedule, step, noise, previous, subnormal).final
        else:
            sampled = self.unified_predictor(
                sample,
                prediction,
                sigma_schedule,
                step,
                previous,
                subnormal,
            )

        return SKSamples(
            final=sampled,
            prediction=prediction,
            sample=sample,
        )
