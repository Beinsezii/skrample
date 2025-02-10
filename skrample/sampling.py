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
    sigma, alpha = sigma_normal(sigma, subnormal)
    return (sample - sigma * output) / alpha


def SAMPLE(sample: Sample, output: Sample, sigma: float, subnormal: bool = False) -> Sample:
    return output


def VELOCITY(sample: Sample, output: Sample, sigma: float, subnormal: bool = False) -> Sample:
    sigma, alpha = sigma_normal(sigma, subnormal)
    return alpha * sample - sigma * output


def FLOW(sample: Sample, output: Sample, sigma: float, subnormal: bool = False) -> Sample:
    return sample - sigma * output


@dataclass(frozen=True)
class SKSamples:
    sampled: Sample
    prediction: Sample
    sample: Sample


@dataclass
class SkrampleSampler(ABC):
    predictor: Callable[[Sample, Sample, float, bool], Sample] = EPSILON

    @staticmethod
    def get_sigma(step: int, schedule: NDArray) -> float:
        return schedule[step, 1].item() if step < len(schedule) else 0

    @abstractmethod
    def sample(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        pass

    def scale_input(self, sample: Sample, sigma: float, subnormal: bool = False) -> Sample:
        return sample

    def merge_noise(self, sample: Sample, noise: Sample, sigma: float, subnormal: bool = False) -> Sample:
        sigma, alpha = sigma_normal(sigma, subnormal)
        return sample * alpha + noise * sigma


@dataclass
class HighOrderSampler(SkrampleSampler):
    order: int = 1

    @property
    def min_order(self) -> int:
        return 1

    @property
    @abstractmethod
    def max_order(self) -> int:
        pass

    def effective_order(self, step: int, schedule: NDArray, previous: list[SKSamples]) -> int:
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
class Euler(SkrampleSampler):
    def sample(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        prediction = self.predictor(sample, output, sigma, subnormal)

        # Dual branch *works* but blows up if sigma schedule is exactly `[1.0, 0.0]`
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
            sampled=sampled,
            prediction=prediction,
            sample=sample,
        )


@dataclass
class DPM(HighOrderSampler):
    """https://arxiv.org/abs/2211.01095
    Page 4 Algo 2 for order=2
    Section 5 for SDE"""

    @property
    def max_order(self) -> int:
        return 2  # TODO: 3, 4+?

    def sample(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        lambda_ = safe_log(alpha) - safe_log(signorm)
        lambda_n1 = safe_log(alpha_n1) - safe_log(signorm_n1)
        h = abs(lambda_n1 - lambda_)

        prediction = self.predictor(sample, output, sigma, subnormal)
        # 1st order
        sampled = (signorm_n1 / signorm) * sample - (alpha_n1 * (math.exp(-h) - 1.0)) * prediction

        effective_order = self.effective_order(step, schedule, previous)

        if effective_order >= 2:
            sigma_p1 = self.get_sigma(step - 1, schedule)
            signorm_p1, alpha_p1 = sigma_normal(sigma_p1, subnormal)

            lambda_p1 = safe_log(alpha_p1) - safe_log(signorm_p1)
            h_p1 = lambda_ - lambda_p1
            r = h_p1 / h  # math people and their var names...

            # Calculate previous predicton from sample, output
            prediction_p1 = previous[-1].prediction
            prediction_p1 = (1.0 / r) * (prediction - prediction_p1)

            # 2nd order
            sampled -= 0.5 * (alpha_n1 * (math.exp(-h) - 1.0)) * prediction_p1

        return SKSamples(
            sampled=sampled,
            prediction=prediction,
            sample=sample,
        )


@dataclass
class UniPC(HighOrderSampler):
    # TODO: Euler bork.
    # Either need to convert Euler to use same scaling fns
    # or override UniPC's to use the solver's scalers.
    solver: SkrampleSampler | None = None

    @property
    def max_order(self) -> int:
        # TODO: seems more stable after converting to python scalars
        # 4-6 is mostly stable now, 7-9 depends on the model. What ranges are actually useful..?
        return 9

    # diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.multistep_uni_c_bh_update
    def unified_corrector(
        self,
        sample: Sample,
        prediction: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> Sample:
        import torch  # einsum operates on the full sample, so numpy would involve lots of data movement.

        # -1 step since it effectively corrects the prior step before the next prediction
        effective_order = self.effective_order(step - 1, schedule, previous[:-1])

        sigma = self.get_sigma(step, schedule)
        sigma_p1 = self.get_sigma(step - 1, schedule)

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
            sigma_pO1, alpha_pO1 = sigma_normal(schedule[step_pO1, 1].item(), subnormal)
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

    # diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.multistep_uni_p_bh_update
    def unified_predictor(
        self,
        sample: Sample,
        prediction: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> Sample:
        import torch  # einsum operates on the full sample, so numpy would involve lots of data movement.

        effective_order = self.effective_order(step, schedule, previous)

        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

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
            sigma_pO, alpha_pO = sigma_normal(schedule[step_pO, 1].item(), subnormal)
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
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> SKSamples:
        sigma = self.get_sigma(step, schedule)
        prediction = self.predictor(sample, output, sigma, subnormal)

        if previous:
            sample = self.unified_corrector(sample, prediction, schedule, step, previous, subnormal)

        if self.solver:
            sampled = self.solver.sample(sample, output, schedule, step, previous, subnormal).sampled
        else:
            sampled = self.unified_predictor(
                sample,
                prediction,
                schedule,
                step,
                previous,
                subnormal,
            )

        return SKSamples(
            sampled=sampled,
            prediction=prediction,
            sample=sample,
        )
