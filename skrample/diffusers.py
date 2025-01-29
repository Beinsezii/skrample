import torch
from torch import Tensor


class SkrampleScheduler:
    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
    ):
        pass

    @property
    def timesteps(self) -> Tensor:
        return torch.zeros([20], dtype=torch.float64)

    @property
    def order(self) -> int:
        return 1

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return original_samples + noise

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
        return torch.zeros_like(model_output), torch.ones_like(model_output)

    def scale_model_input(self, sample: Tensor, timestep: float | Tensor) -> Tensor:
        return sample
