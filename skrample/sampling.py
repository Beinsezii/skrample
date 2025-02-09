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
        return float("inf")


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

        prediction = self.predictor(self.scale_input(sample, sigma), output, sigma, subnormal)

        return SKSamples(
            sampled=sample + ((sample - prediction) / sigma) * (sigma_n1 - sigma),
            prediction=prediction,
            sample=sample,
        )

    def scale_input(self, sample: Sample, sigma: float, subnormal: bool = False) -> Sample:
        return sample / ((sigma**2 + 1) ** 0.5)

    def merge_noise(self, sample: Sample, noise: Sample, sigma: float, subnormal: bool = False) -> Sample:
        return sample + noise * sigma


@dataclass
class EulerFlow(SkrampleSampler):
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

        return SKSamples(
            sampled=sample + (sigma_n1 - sigma) * output,
            prediction=output,
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
    # TODO: custom solvers?
    # solver: SkrampleSampler | None = None

    @property
    def max_order(self) -> int:
        return 9  # TODO: 4+ is super unstable. Probably either workaround or clamp

    # diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.multistep_uni_c_bh_update
    def unified_corrector(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> Sample:
        import torch

        # -1 step since it effectively corrects the prior step before the next prediction
        effective_order = self.effective_order(step - 1, schedule, previous[:-1])  # remove extra sample

        sigma = self.get_sigma(step, schedule)
        sigma_p1 = self.get_sigma(step - 1, schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_p1, alpha_p1 = sigma_normal(sigma_p1, subnormal)

        this_model_output = output
        this_sample = sample

        model_output_list = previous
        order = effective_order

        m0 = model_output_list[-1].prediction
        # x = last_sample
        x = model_output_list[-1].sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = torch.tensor(sigma), torch.tensor(sigma_p1)
        alpha_t, sigma_t = torch.tensor(alpha), torch.tensor(signorm)
        alpha_s0, sigma_s0 = torch.tensor(alpha_p1), torch.tensor(signorm_p1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = step - (i + 1)
            mi = model_output_list[-(i + 1)].prediction
            sigma_si, alpha_si = sigma_normal(torch.tensor(schedule[si, 1]), subnormal)
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        # hh = -h if self.predict_x0 else h
        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        # # BH1
        # B_h = hh
        # BH2
        B_h = torch.expm1(hh)

        for i in range(1, effective_order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        # if self.predict_x0:
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        # else:
        #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        #     if D1s is not None:
        #         corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        #     else:
        #         corr_res = 0
        #     D1_t = model_t - m0
        #     x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    # diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.multistep_uni_p_bh_update
    def unified_predictor(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
        subnormal: bool = False,
    ) -> Sample:
        import torch

        effective_order = min(
            step + 1,
            self.order,
            len(previous),
            len(schedule) - step,  # lower for final is the default
        )

        effective_order = self.effective_order(step, schedule, previous[:-1])  # remove extra sample

        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        signorm, alpha = sigma_normal(sigma, subnormal)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1, subnormal)

        model_output_list = previous
        order = effective_order

        m0 = model_output_list[-1].prediction
        x = sample

        # TODO: custom solvers?
        # if self.solver:
        #     return self.solver.sample(sample, output, schedule, step, previous, subnormal).sampled

        # if self.solver_p:
        #     s0 = self.timestep_list[-1]
        #     x_t = self.solver_p.step(model_output, s0, x).prev_sample
        #     return x_t

        sigma_t, sigma_s0 = torch.tensor(sigma_n1), torch.tensor(sigma)
        alpha_t, sigma_t = torch.tensor(alpha_n1), torch.tensor(signorm_n1)
        alpha_s0, sigma_s0 = torch.tensor(alpha), torch.tensor(signorm)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = step - i
            mi = model_output_list[-(i + 1)].prediction
            sigma_si, alpha_si = sigma_normal(torch.tensor(schedule[si, 1]), subnormal)
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        # hh = -h if self.predict_x0 else h
        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        # # bh1
        # B_h = hh
        # bh2
        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if effective_order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        # if self.predict_x0:
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
        # else:
        #     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        #     if D1s is not None:
        #         pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        #     else:
        #         pred_res = 0
        #     x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
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

        sampled = self.unified_predictor(
            sample,
            output,
            schedule,
            step,
            previous + [SKSamples(prediction=prediction, sampled=sample, sample=sample)],
            subnormal,
        )

        return SKSamples(
            sampled=sampled,
            prediction=prediction,
            sample=sample,
        )
