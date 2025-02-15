import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
        pass

    def timesteps(self, steps: int) -> NDArray[np.float64]:
        return self.schedule(steps)[:, 0]

    def sigmas(self, steps: int) -> NDArray[np.float64]:
        return self.schedule(steps)[:, 1]

    def __call__(self, steps: int) -> NDArray[np.float64]:
        return self.schedule(steps)


@dataclass
class ScheduleCommon(SkrampleSchedule):
    # keep diffusers names for now
    num_train_timesteps: int = 1000


@dataclass
class Scaled(ScheduleCommon):
    beta_start: float = 0.00085
    beta_end: float = 0.012
    scale: float = 2

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
            return np.linspace(self.num_train_timesteps - 1, 0, steps + 1, dtype=np.float64).round()[:-1]
        else:
            # They use a truncated ratio for ...reasons?
            return np.flip(np.arange(0, steps, dtype=np.float64) * (self.num_train_timesteps // steps)).round()

    def betas(self) -> NDArray[np.float64]:
        return (
            np.linspace(
                self.beta_start ** (1 / self.scale),
                self.beta_end ** (1 / self.scale),
                self.num_train_timesteps,
                dtype=np.float64,
            )
            ** self.scale
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
class Flow(ScheduleCommon):
    mu: float | None = None
    shift: float = 3.0
    # base_image_seq_len: int = 256
    # max_image_seq_len: float = 4096
    # base_shift: float = 0.5
    # max_shift: float = 1.15
    # use_dynamic_shifting: bool = True

    @property
    def subnormal(self) -> bool:
        return True

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        return sigmas * self.num_train_timesteps

    def sigmas(self, steps: int) -> NDArray[np.float64]:
        # # # The actual schedule code
        #
        # # Strange it's 1000 -> 1 instead of 999 -> 0?
        # sigma_start, sigma_end = 1, 1 / self.num_train_timesteps
        #
        # if self.mu is None:
        #     sigma_start = self.shift * sigma_start / (1 + (self.shift - 1) * sigma_start)
        #     sigma_end = self.shift * sigma_end / (1 + (self.shift - 1) * sigma_end)
        #
        # sigmas = np.linspace(sigma_start, sigma_end, steps + 1, dtype=np.float64)[:-1]

        # What the flux pipeline overrides it to. Seems more correct?
        sigmas = np.linspace(1, 1 / steps, steps, dtype=np.float64)

        if self.mu is not None:  # dynamic
            sigmas = math.exp(self.mu) / (math.exp(self.mu) + (1 / sigmas - 1))
        else:  # non-dynamic
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        return sigmas

    def schedule(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.sigmas(steps)
        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps, sigmas], axis=1)


@dataclass
class ScheduleModifier(SkrampleSchedule):
    base: SkrampleSchedule

    @property
    def subnormal(self) -> bool:
        return self.base.subnormal

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.base.sigmas_to_timesteps(sigmas)


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
