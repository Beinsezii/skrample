import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SkrampleSchedule(ABC):
    num_train_timesteps: int = 1000

    @abstractmethod
    def __call__(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        pass

    def timesteps(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        return self(steps, mu)[:, 0]

    def sigmas(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        return self(steps, mu)[:, 1]


@dataclass
class Scaled(SkrampleSchedule):
    # keep diffusers names for now
    beta_start: float = 0.00085
    beta_end: float = 0.012

    # Let's name this "uniform" instead of trailing since it basically just avoids the truncation.
    # Think that's what ComfyUI does
    uniform: bool = True

    def __call__(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        # # https://arxiv.org/abs/2305.08891 Table 2
        if self.uniform:
            timesteps = np.linspace(self.num_train_timesteps - 1, 0, steps + 1, dtype=np.float32).round()[:-1]
        else:
            # They use a truncated ratio for ...reasons?
            timesteps = np.flip(np.arange(0, steps, dtype=np.float32) * (self.num_train_timesteps // steps)).round()

        betas = np.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps, dtype=np.float32) ** 2
        alphas_cumprod = np.cumprod(1 - betas, axis=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas).astype(np.float32)

        return np.stack([timesteps, sigmas], axis=1)


@dataclass
class Flow(SkrampleSchedule):
    shift: float = 3.0
    # base_image_seq_len: int = 256
    # max_image_seq_len: float = 4096
    # base_shift: float = 0.5
    # max_shift: float = 1.15
    # use_dynamic_shifting: bool = True

    def __call__(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
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

        if mu is not None:  # dynamic
            sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        else:  # non-dynamic
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        timesteps = sigmas * self.num_train_timesteps

        return np.stack([timesteps, sigmas], axis=1)
