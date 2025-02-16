import dataclasses
import math
from collections import OrderedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from skrample.sampling import SkrampleSampler, SKSamples, StochasticSampler
from skrample.scheduling import Flow, SkrampleSchedule


class SkrampleWrapperScheduler:
    sampler: SkrampleSampler
    schedule: SkrampleSchedule
    compute_scale: torch.dtype | None

    _steps: int = 50
    _device: torch.device = torch.device("cpu")
    _previous: list[SKSamples] = []

    def __init__(
        self,
        sampler: SkrampleSampler,
        schedule: SkrampleSchedule,
        compute_scale: torch.dtype | None = torch.float32,
    ):
        self.sampler = sampler
        self.schedule = schedule
        self.compute_scale = compute_scale

    @property
    def schedule_np(self) -> NDArray[np.float64]:
        return self.schedule(steps=self._steps)

    @property
    def schedule_pt(self) -> Tensor:
        return torch.from_numpy(self.schedule_np).to(self._device)

    @property
    def timesteps(self) -> Tensor:
        return torch.from_numpy(self.schedule.timesteps(steps=self._steps)).to(self._device)

    @property
    def sigmas(self) -> Tensor:
        sigmas = torch.from_numpy(self.schedule.sigmas(steps=self._steps)).to(self._device)
        # diffusers expects the extra zero
        return torch.cat([sigmas, torch.zeros([1], device=sigmas.device, dtype=sigmas.dtype)])

    @property
    def init_noise_sigma(self) -> float:
        # idk why tf diffusers uses this instead of add_noise() for some shit
        return self.sampler.merge_noise(0, 1, self.schedule_np[0, 1].item(), subnormal=self.schedule.subnormal)  # type: ignore

    @property
    def order(self) -> int:
        return 1  # for multistep this is always 1

    @property
    def config(self):
        fake_config = (
            {
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "max_image_seq_len": 4096,
                "max_shift": 1.15,
                "num_train_timesteps": 1000,
                "shift": 3.0,
                "use_dynamic_shifting": True,
            }
            # Since we use diffusers names this will just work™
            # Eventually when we use prettier names this will need a LUT
            | dataclasses.asdict(self.sampler)
            | dataclasses.asdict(self.schedule)
        )
        fake_config_object = OrderedDict(**fake_config)
        for k, v in fake_config_object.items():
            setattr(fake_config_object, k, v)
        return fake_config_object

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
        if isinstance(self.schedule, Flow):
            self.schedule.mu = mu
        self._previous = []

        if device is not None:
            self._device = torch.device(device)

    def scale_noise(self, sample: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep.item())
        sigma = schedule[step, 1].item()
        return self.sampler.merge_noise(sample, noise, sigma, subnormal=self.schedule.subnormal)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.scale_noise(original_samples, timesteps[0], noise)

    def scale_model_input(self, sample: Tensor, timestep: float | Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int | float)) else timestep.item())
        sigma = schedule[step, 1].item()
        return self.sampler.scale_input(sample, sigma, subnormal=self.schedule.subnormal)

    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
    ) -> tuple[Tensor, Tensor]:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int, float)) else timestep.item())

        if isinstance(self.sampler, StochasticSampler) and self.sampler.add_noise:
            if isinstance(generator, list) and len(generator) == sample.shape[0]:
                seeds = generator
            elif isinstance(generator, torch.Generator) and sample.shape[0] == 1:
                seeds = [generator]
            else:
                # use median element +4 decimals as seed for a balance of determinism without lacking variety
                # multiply by step index to spread the values and minimize clash
                # does not work across batch sizes but at least Flux will have something mostly deterministic
                seeds = [
                    torch.Generator().manual_seed(int(b.view(b.numel())[b.numel() // 2].item() * 1e4) * (step + 1))
                    for b in sample
                ]

            noise = torch.stack(
                [
                    torch.randn(
                        sample.shape[1:],  # exclude batch
                        generator=g,  # should we even use the generators?
                        dtype=torch.float32,  # lock to cpu f32 for consistent seeds
                        device="cpu",
                    ).to(device=sample.device, dtype=self.compute_scale)
                    for g in seeds
                ]
            )
        else:
            noise = None

        if return_dict:
            raise ValueError
        else:
            sampled = self.sampler.sample(
                sample=sample.to(dtype=self.compute_scale),
                output=model_output.to(dtype=self.compute_scale),
                sigma_schedule=schedule[:, 1],
                step=step,
                noise=noise,
                previous=self._previous,
                subnormal=self.schedule.subnormal,
            )
            self._previous.append(sampled)
            return (
                sampled.final.to(device=model_output.device, dtype=model_output.dtype),
                sampled.prediction.to(device=model_output.device, dtype=model_output.dtype),
            )
