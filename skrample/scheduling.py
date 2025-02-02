import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ScheduleTrait(ABC):
    num_train_timesteps: int = 1000

    @abstractmethod
    def __call__(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        pass

    def timesteps(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        return self(steps, mu)[:, 0]

    def sigmas(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        return self(steps, mu)[:, 1]

    # diffusers copy for now
    def time_shift(self, mu: float, sigma: float, schedule: NDArray[np.float32]) -> NDArray[np.float32]:
        return math.exp(mu) / (math.exp(mu) + (1 / schedule - 1) ** sigma)


@dataclass
class ScaledSchedule(ScheduleTrait):
    # keep diffusers names for now
    beta_start: float = 0.00085
    beta_end: float = 0.012
    trailing: bool = False

    def __call__(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        ratio = self.num_train_timesteps / steps

        # # https://arxiv.org/abs/2305.08891 Table 2
        if self.trailing:  # # Trailing
            new_timesteps = np.arange(steps, 0, -ratio, dtype=np.float32).round().copy() - 1
        else:  # # Leading
            new_timesteps = (np.arange(0, steps, dtype=np.float32) * ratio).round()[::-1].copy()

        # # Linspace. Useful?
        # new_timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.float64)[::-1].copy()

        # Step offset for SD
        new_timesteps += 1

        # Default scaled sigma schedule
        betas = np.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps, dtype=np.float32) ** 2
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = np.interp(new_timesteps, np.arange(0, len(sigmas)), sigmas).astype(np.float32)

        return np.stack([new_timesteps, sigmas], axis=1)


@dataclass
class FlowSchedule(ScheduleTrait):
    # keep diffusers names for now
    base_image_seq_len: int = 256
    max_image_seq_len: float = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15
    shift: float = 3.0
    use_dynamic_shifting: bool = True

    def __call__(self, steps: int, mu: float | None = None) -> NDArray[np.float32]:
        sigmas = np.linspace(1, 0, steps + 1, dtype=np.float32)[:-1]

        if mu is not None:  # dynamic
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:  # non-dynamic
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        timesteps = sigmas * self.num_train_timesteps

        return np.stack([timesteps, sigmas], axis=1)
