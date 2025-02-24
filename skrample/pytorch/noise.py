from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch


@dataclass
class SkrampleTensorNoise(ABC):
    @abstractmethod
    def generate(self, step: int) -> torch.Tensor:
        pass


@dataclass
class TensorNoiseCommon(SkrampleTensorNoise):
    shape: tuple[int, ...]
    seed: torch.Generator
    dtype: torch.dtype
    device: torch.device

    @classmethod
    @abstractmethod
    def from_inputs(
        cls,
        sample: torch.Tensor,
        schedule: np.typing.NDArray[np.float64],
        seed: torch.Generator,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        raise NotImplementedError


@dataclass
class Random(TensorNoiseCommon):
    def generate(self, step: int) -> torch.Tensor:
        return torch.randn(
            self.shape,
            generator=self.seed,
            dtype=self.dtype,
            device=self.device,
        )

    @classmethod
    def from_inputs(
        cls,
        sample: torch.Tensor,
        schedule: np.typing.NDArray[np.float64],
        seed: torch.Generator,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        return cls(
            tuple(sample.shape),
            seed,
            dtype,
            device,
        )


@dataclass
class Brownian(TensorNoiseCommon):
    sigma_schedule: np.typing.NDArray[np.float64]

    def __post_init__(self):
        import torchsde

        self._tree = torchsde.BrownianInterval(
            size=self.shape,
            entropy=self.seed.initial_seed(),
            dtype=self.dtype,
            device=self.device,
        )

        self.sigma_schedule = self.sigma_schedule / self.sigma_schedule.max()

    def generate(self, step: int) -> torch.Tensor:
        schedule = self.sigma_schedule / self.sigma_schedule.max()
        sigma = schedule[step]
        sigma_next = 0 if step + 1 >= len(schedule) else schedule[step + 1]

        return self._tree(sigma_next, sigma) / abs(sigma_next - sigma) ** 0.5

    @classmethod
    def from_inputs(
        cls,
        sample: torch.Tensor,
        schedule: np.typing.NDArray[np.float64],
        seed: torch.Generator,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        return cls(
            shape=tuple(sample.shape),
            seed=seed,
            sigma_schedule=schedule[:, 1],
            dtype=dtype,
            device=device,
        )


@dataclass
class BatchTensorNoise(SkrampleTensorNoise):
    generators: list[SkrampleTensorNoise]

    def generate(
        self,
        step: int,
    ) -> torch.Tensor:
        return torch.stack([g.generate(step) for g in self.generators])

    @classmethod
    def from_batch_inputs(
        cls,
        subclass: type[TensorNoiseCommon],
        sample: torch.Tensor,
        schedule: np.typing.NDArray[np.float64],
        seeds: list[torch.Generator],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        return cls(
            [
                subclass.from_inputs(slice, schedule, seed, dtype, device)
                for slice, seed in zip(sample, seeds, strict=True)
            ]
        )
