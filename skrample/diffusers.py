import collections
import dataclasses
import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from skrample.sampling import SKSamples, SkrampleSampler
from skrample.scheduling import SkrampleSchedule


class SkrampleWrapperScheduler:
    sampler: SkrampleSampler
    schedule: SkrampleSchedule

    _steps: int
    _mu: float | None = None
    _device: torch.device = torch.device("cpu")
    _previous: list[SKSamples] = []

    def __init__(self, sampler: SkrampleSampler, schedule: SkrampleSchedule):
        self.sampler = sampler
        self.schedule = schedule

        self._steps = schedule.num_train_timesteps

    @property
    def schedule_np(self) -> NDArray[np.float32]:
        return self.schedule(steps=self._steps, mu=self._mu)

    @property
    def schedule_pt(self) -> Tensor:
        return torch.from_numpy(self.schedule(steps=self._steps, mu=self._mu)).to(self._device)

    @property
    def timesteps(self) -> Tensor:
        return self.schedule_pt[:, 0]

    @property
    def sigmas(self) -> Tensor:
        sigmas = self.schedule_pt[:, 1]
        # diffusers expects the extra zero
        return torch.cat([sigmas, torch.zeros([1], device=sigmas.device, dtype=sigmas.dtype)])

    @property
    def init_noise_sigma(self) -> float:
        return 1

    @property
    def order(self) -> int:
        return getattr(self.sampler, "order", 1)

    @property
    def config(self):
        # Since we use diffusers names this will just work™
        # Eventually when we use prettier names this will need a LUT
        fake_config = dataclasses.asdict(self.sampler) | dataclasses.asdict(self.schedule)
        return collections.namedtuple("FrozenDict", field_names=fake_config.keys())(**fake_config)

    def time_shift(self, mu: float, sigma: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        timesteps: Tensor | list[int] | None = None,
        sigmas: Tensor | list[float] | None = None,
        mu: float | None = None,
    ):
        if num_inference_steps is None:
            if timesteps is not None:
                num_inference_steps = len(timesteps)
            elif sigmas is not None:
                num_inference_steps = len(sigmas)
            else:
                return

        self._steps = num_inference_steps
        self._mu = mu
        self._previous = []

        if device is not None:
            self._device = torch.device(device)

    def scale_noise(self, sample: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep.item())
        sigma = schedule[step, 1].item()
        return self.sampler.merge_noise(sample, noise, sigma)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.scale_noise(original_samples, timesteps[0], noise)

    def scale_model_input(self, sample: Tensor, timestep: float | Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int | float)) else timestep.item())
        sigma = schedule[step, 1].item()
        return self.sampler.scale_input(sample, sigma)

    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> tuple[Tensor, Tensor]:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int, float)) else timestep.item())

        if return_dict:
            raise ValueError
        else:
            sampled = self.sampler.sample(
                sample=sample,
                output=model_output,
                schedule=schedule,
                step=step,
                previous=self._previous,
            )
            self._previous.append(sampled)
            return (sampled.sampled, sampled.prediction)
