import collections
import dataclasses
import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from skrample.scheduling import ScheduleTrait


class SkrampleScheduler:
    schedule: ScheduleTrait
    flow: bool  # Hack for before we split

    _steps: int
    _mu: float | None = None
    _device: torch.device = torch.device("cpu")

    def __init__(self, schedule: ScheduleTrait, flow: bool = False):
        self.schedule = schedule
        self.flow = flow

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
    def order(self) -> int:
        return 1

    @property
    def config(self):
        # Since we use diffusers names this will just work™
        # Eventually when we use prettier names this will need a LUT
        fake_config = dataclasses.asdict(self.schedule)
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

        if device is not None:
            self._device = torch.device(device)

    # Non-Flow
    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timesteps[0].item())
        sigma = schedule[step, 1]
        return original_samples + noise * sigma

    # FlowMatch
    def scale_noise(self, sample: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep.item())
        sigma = schedule[step, 1]
        return sigma * noise + (1.0 - sigma) * sample

    # Only called on FlowMatch afaik
    def scale_model_input(self, sample: Tensor, timestep: float | Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int | float)) else timestep.item())
        sigma = schedule[step, 1]
        return sample / ((sigma**2 + 1) ** 0.5)

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

        sigma = schedule[step, 1]
        sigma_next = 0 if step + 1 >= len(schedule) else schedule[step + 1, 1]

        if self.flow:
            # Euler Flow
            prev_sample = sample.to(torch.float32) + (sigma_next - sigma) * model_output.to(torch.float32)
            pred_original_sample = sample.clone().to(torch.float32)
        else:
            # # Sample
            # pred_original_sample = model_output
            # Epsilon
            pred_original_sample = sample.to(torch.float32) - sigma * model_output.to(torch.float32)
            # # denoised = model_output * c_out + input * c_skip
            # # Velocity
            # pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))

            # Euler
            derivative = (sample - pred_original_sample) / sigma
            prev_sample = sample + derivative * (sigma_next - sigma)

        if return_dict:
            raise ValueError
        else:
            return (
                prev_sample.to(model_output.dtype),
                pred_original_sample.to(model_output.dtype),
            )
