import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skrample.common import normalize, regularize, sigmoid


@dataclass
class SkrampleSchedule(ABC):
    @property
    def subnormal(self) -> bool:
        """Whether or not the sigma values all fall within 0..1.
        Needs alternative sampling strategies."""
        return False

    @abstractmethod
    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def schedule(self, steps: int) -> NDArray[np.float64]:
        "Return the full noise schedule, timesteps stacked on top of sigmas."

    def timesteps(self, steps: int) -> NDArray[np.float64]:
        return self.schedule(steps)[:, 0]

    def sigmas(self, steps: int) -> NDArray[np.float64]:
        return self.schedule(steps)[:, 1]

    def __call__(self, steps: int) -> NDArray[np.float64]:
        return self.schedule(steps)


@dataclass
class ScheduleCommon(SkrampleSchedule):
    # keep diffusers names for now
    base_timesteps: int = 1000


@dataclass
class Scaled(ScheduleCommon):
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_scale: float = 2

    # Let's name this "uniform" instead of trailing since it basically just avoids the truncation.
    # Think that's what ComfyUI does
    uniform: bool = True
    "Equivalent to spacing='trailing' in diffusers"

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        # it uses full distribution pre-interp
        scaled_sigmas = self.scaled_sigmas(self.alphas_cumprod(self.betas()))
        log_sigmas = np.log(scaled_sigmas)

        # below here just a copy of diffusers' _sigma_to_t

        # get log sigma
        log_sigma = np.log(np.maximum(sigmas, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        return t

    def timesteps(self, steps: int) -> NDArray[np.float64]:
        # # https://arxiv.org/abs/2305.08891 Table 2
        if self.uniform:
            return np.linspace(self.base_timesteps - 1, 0, steps + 1, dtype=np.float64).round()[:-1]
        else:
            # They use a truncated ratio for ...reasons?
            return np.flip(np.arange(0, steps, dtype=np.float64) * (self.base_timesteps // steps)).round()

    def betas(self) -> NDArray[np.float64]:
        return (
            np.linspace(
                self.beta_start ** (1 / self.beta_scale),
                self.beta_end ** (1 / self.beta_scale),
                self.base_timesteps,
                dtype=np.float64,
            )
            ** self.beta_scale
        )

    def alphas_cumprod(self, betas: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.cumprod(1 - betas, axis=0)

    def scaled_sigmas(self, alphas_cumprod: NDArray[np.float64]) -> NDArray[np.float64]:
        return ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

    def schedule(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.scaled_sigmas(self.alphas_cumprod(self.betas()))
        timesteps = self.timesteps(steps)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        return np.stack([timesteps, sigmas], axis=1)


@dataclass
class ZSNR(Scaled):
    # Just some funny number I made up when working on the diffusers PR that worked well. F32 smallest subnormal
    epsilon: float = 2**-24
    "Amount to shift the zero value by to keep calculations finite."

    uniform: bool = True
    "ZSNR should always uniform/trailing"

    def alphas_cumprod(self, betas: NDArray[np.float64]) -> NDArray[np.float64]:
        ### from https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        # Convert betas to alphas_bar_sqrt
        alphas_bar_sqrt = np.cumprod(1 - betas, axis=0) ** 0.5

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].item()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].item()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_cumprod = alphas_bar_sqrt**2  # Revert sqrt

        alphas_cumprod[-1] = self.epsilon  # Epsilon to avoid inf
        return alphas_cumprod


@dataclass
class Linear(ScheduleCommon):
    sigma_start: float = 1
    # TODO(beinsezii): don't assume zero
    # Whole library needs to be updates for this
    # sigma_end: float = 0
    present_subnormal: bool | None = None
    """If set to a bool, will be used as the value for `self.subnormal`
    Otherwise, subnormal will be False for sigmas_start > 1 and True for <= 1"""

    @property
    def subnormal(self) -> bool:
        if self.present_subnormal is None:
            return self.sigma_start <= 1
        else:
            return self.present_subnormal

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        return normalize(sigmas, start=self.sigma_start) * self.base_timesteps

    def sigmas(self, steps: int) -> NDArray[np.float64]:
        return np.linspace(self.sigma_start, 0, steps, endpoint=False, dtype=np.float64)

    def schedule(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.sigmas(steps)
        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps, sigmas], axis=1)


@dataclass
class SigmoidCDF(Linear):
    cdf_scale: float = 3

    def sigmas(self, steps: int) -> NDArray[np.float64]:
        from scipy.stats import norm

        step_peak = 1 / (steps * math.pi / 2)
        probabilities = np.linspace(step_peak, 1 - step_peak, steps, dtype=np.float64)
        sigmas = sigmoid(norm.ppf(probabilities) * self.cdf_scale)
        return regularize(sigmas, start=self.sigma_start)


@dataclass
class ScheduleModifier(SkrampleSchedule):
    base: SkrampleSchedule

    @property
    def subnormal(self) -> bool:
        return self.base.subnormal

    @property
    def all(self) -> list[SkrampleSchedule]:
        bases: list[SkrampleSchedule] = [self]
        last = self.base
        while isinstance(last, ScheduleModifier):
            bases.append(last)
            last = last.base
        bases.append(last)

        return bases

    @property
    def lowest(self) -> SkrampleSchedule:
        return self.all[-1]

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.base.sigmas_to_timesteps(sigmas)

    def find[T: SkrampleSchedule](self, skrample_schedule: type[T]) -> T | None:
        for schedule in self.all:
            if type(schedule) is skrample_schedule:
                return schedule


@dataclass
class NoMod(ScheduleModifier):
    "Does nothing. For generic programming against ScheduleModifier"

    def schedule(self, steps: int) -> NDArray[np.float64]:
        return self.base.schedule(steps)


@dataclass
class FlowShift(ScheduleModifier):
    mu: float | None = None
    shift: float = 3.0

    def schedule(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.base.sigmas(steps)

        # Compute flow match in 0-1 scale
        # TODO(beinsezii): maybe the shift itself should be rewritten to accomodate start/end?
        start, end = sigmas.max(), 0
        sigmas = normalize(sigmas, start=start, end=end)

        if self.mu is not None:  # dynamic
            sigmas = math.exp(self.mu) / (math.exp(self.mu) + (1 / sigmas - 1))
        else:  # non-dynamic
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        sigmas = regularize(sigmas, start=start, end=end)
        timesteps = self.sigmas_to_timesteps(sigmas)
        return np.stack([timesteps.flatten(), sigmas], axis=1)


@dataclass
class Karras(ScheduleModifier):
    rho: float = 7.0

    def schedule(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.base.sigmas(steps)

        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()

        ramp = np.linspace(0, 1, steps, dtype=np.float64)
        min_inv_rho = sigma_min ** (1 / self.rho)
        max_inv_rho = sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps.flatten(), sigmas], axis=1)


@dataclass
class Exponential(ScheduleModifier):
    "Also known as 'polyexponential' when rho != 1"

    rho: float = 1.0
    "Ramp power"

    def schedule(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.base.sigmas(steps)
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()

        ramp = np.linspace(1, 0, steps, dtype=np.float64) ** self.rho
        sigmas = np.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))

        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps, sigmas], axis=1)


@dataclass
class Beta(ScheduleModifier):
    alpha: float = 0.6
    beta: float = 0.6

    def schedule(self, steps: int) -> NDArray[np.float64]:
        import scipy

        sigmas = self.base.sigmas(steps)

        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()

        pparr = scipy.stats.beta.ppf(1 - np.linspace(0, 1, steps, dtype=np.float64), self.alpha, self.beta)
        sigmas = sigma_min + (pparr * (sigma_max - sigma_min))

        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps.flatten(), sigmas], axis=1)
