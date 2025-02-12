import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SkrampleSchedule(ABC):
    # keep diffusers names for now
    num_train_timesteps: int = 1000

    @property
    def subnormal(self) -> bool:
        """Whether or not the sigma values all fall within 0..1.
        Needs alternative sampling strategies."""
        return False

    @abstractmethod
    def schedule(self, steps: int) -> NDArray[np.float32]:
        "Return the full noise schedule, timesteps stacked on top of sigmas."
        pass

    def timesteps(self, steps: int) -> NDArray[np.float32]:
        return self.schedule(steps)[:, 0]

    def sigmas(self, steps: int) -> NDArray[np.float32]:
        return self.schedule(steps)[:, 1]

    def __call__(self, steps: int) -> NDArray[np.float32]:
        return self.schedule(steps)


@dataclass
class Scaled(SkrampleSchedule):
    beta_start: float = 0.00085
    beta_end: float = 0.012
    scale: float = 2

    # Let's name this "uniform" instead of trailing since it basically just avoids the truncation.
    # Think that's what ComfyUI does
    uniform: bool = True

    def schedule(self, steps: int) -> NDArray[np.float32]:
        # # https://arxiv.org/abs/2305.08891 Table 2
        if self.uniform:
            timesteps = np.linspace(self.num_train_timesteps - 1, 0, steps + 1, dtype=np.float32).round()[:-1]
        else:
            # They use a truncated ratio for ...reasons?
            timesteps = np.flip(np.arange(0, steps, dtype=np.float32) * (self.num_train_timesteps // steps)).round()

        betas = (
            np.linspace(
                self.beta_start ** (1 / self.scale),
                self.beta_end ** (1 / self.scale),
                self.num_train_timesteps,
                dtype=np.float32,
            )
            ** self.scale
        )
        alphas_cumprod = np.cumprod(1 - betas, axis=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas).astype(np.float32)

        return np.stack([timesteps, sigmas], axis=1)


@dataclass
class ZSNR(Scaled):
    # Just some funny number I made up when working on the diffusers PR that worked well. F32 smallest subnormal
    epsilon = 2**-24
    "Amount to shift the zero value by to keep calculations finite."

    # ZSNR should always uniform/trailing
    uniform: bool = True

    def schedule(self, steps: int) -> NDArray[np.float32]:
        # from super()
        if self.uniform:
            timesteps = np.linspace(self.num_train_timesteps - 1, 0, steps + 1, dtype=np.float32).round()[:-1]
        else:
            timesteps = np.flip(np.arange(0, steps, dtype=np.float32) * (self.num_train_timesteps // steps)).round()

        betas = (
            np.linspace(
                self.beta_start ** (1 / self.scale),
                self.beta_end ** (1 / self.scale),
                self.num_train_timesteps,
                dtype=np.float32,
            )
            ** self.scale
        )

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
        ###

        # from super()
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas).astype(np.float32)

        return np.stack([timesteps, sigmas], axis=1)


@dataclass
class Flow(SkrampleSchedule):
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

    def schedule(self, steps: int) -> NDArray[np.float32]:
        # # # The actual schedule code
        #
        # # Strange it's 1000 -> 1 instead of 999 -> 0?
        # sigma_start, sigma_end = 1, 1 / self.num_train_timesteps
        #
        # if mu is None:
        #     sigma_start = self.shift * sigma_start / (1 + (self.shift - 1) * sigma_start)
        #     sigma_end = self.shift * sigma_end / (1 + (self.shift - 1) * sigma_end)
        #
        # sigmas = np.linspace(sigma_start, sigma_end, steps, dtype=np.float32)
        # sigmas = np.linspace(sigma_start, sigma_end, steps + 1, dtype=np.float32)[:-1]

        # What the flux pipeline overrides it to. Seems more correct?
        sigmas = np.linspace(1, 1 / steps, steps, dtype=np.float32)

        if self.mu is not None:  # dynamic
            sigmas = math.exp(self.mu) / (math.exp(self.mu) + (1 / sigmas - 1))
        else:  # non-dynamic
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        timesteps = sigmas * self.num_train_timesteps

        return np.stack([timesteps, sigmas], axis=1)
