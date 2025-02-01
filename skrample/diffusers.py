import collections
import math

import numpy as np
import torch
from torch import Tensor


class SkrampleScheduler:
    _timesteps: Tensor = torch.empty([0], dtype=torch.float64)
    _sigmas: Tensor = torch.empty([0], dtype=torch.float64)
    _device: torch.device = torch.device("cpu")
    flow: bool  # Hack for before we split

    def __init__(self, flow: bool = False):
        self.flow = flow
        self.set_timesteps(1000)

    @property
    def timesteps(self) -> Tensor:
        return self._timesteps.clone().to(self._device)

    @property
    def sigmas(self) -> Tensor:
        return self._sigmas.clone().to(self._device)

    @property
    def order(self) -> int:
        return 1

    @property
    def config(self):
        # Flux
        fake_config = {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "num_train_timesteps": 1000,
            "shift": 3.0,
            "use_dynamic_shifting": True,
        }
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

        if device is not None:
            self._device = torch.device(device)

        # config
        num_train_timesteps = 1000
        beta_start = 0.00085
        beta_end = 0.012

        shift = 3.0  # Flux

        if self.flow:
            sigmas = torch.linspace(1, 0, num_inference_steps + 1, dtype=torch.float32)[:-1]

            if mu is not None:  # dynamic
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:  # non-dynamic
                sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

            new_timesteps = (sigmas * num_train_timesteps).numpy()
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        else:
            ratio = num_train_timesteps / num_inference_steps

            # # https://arxiv.org/abs/2305.08891 Table 2
            # # Linspace
            # new_timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.float64)[::-1].copy()
            # # Leading
            new_timesteps = (np.arange(0, num_inference_steps) * ratio).round()[::-1].copy()
            # # Trailing
            # new_timesteps = np.arange(num_train_timesteps, 0, -ratio).round().copy() - 1

            # Step offset for SD
            new_timesteps += 1

            # Default scaled sigma schedule
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
            alphas = 1 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            sigmas_np = np.interp(new_timesteps, np.arange(0, len(sigmas)), sigmas.numpy())
            sigmas = torch.cat([torch.from_numpy(sigmas_np), torch.zeros(1, device=sigmas.device)])

        self._sigmas = sigmas
        self._timesteps = torch.from_numpy(new_timesteps)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        step = self._timesteps.tolist().index(timesteps[0].item())
        sigma = self._sigmas[step]
        return original_samples + noise * sigma

    def scale_noise(self, sample: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        step = self._timesteps.tolist().index(timestep.item())
        sigma = self._sigmas[step]
        return sigma * noise + (1.0 - sigma) * sample

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
        step = self._timesteps.tolist().index(timestep if isinstance(timestep, (int, float)) else timestep.item())
        print(step, timestep)

        sigma = self._sigmas[step]
        sigma_next = self._sigmas[step + 1]

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

    def scale_model_input(self, sample: Tensor, timestep: float | Tensor) -> Tensor:
        step = self._timesteps.tolist().index(timestep if isinstance(timestep, (int | float)) else timestep.item())
        sigma = self._sigmas[step]
        return sample / ((sigma**2 + 1) ** 0.5)
