import math
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
class Offset(Random):
    "Simple random offset along dimension[s]"

    dims: tuple[int, ...] = (0,)
    strength: float = 0.2  # low enough to not go boom ...usually
    static: bool = False

    def __post_init__(self):
        if self.static:
            self.static_offset: torch.Tensor | None = self.offset()
        else:
            self.static_offset = None

    def offset(self) -> torch.Tensor:
        shape = [d if n in self.dims else 1 for n, d in enumerate(self.shape)]
        return torch.randn(shape, generator=self.seed, dtype=self.dtype, device=self.device) * self.strength**2

    def generate(self, step: int) -> torch.Tensor:
        if self.static and self.static_offset is not None:
            offset = self.static_offset
        else:
            offset = self.offset()
        return super().generate(step) + offset


@dataclass
class Pyramid(Random):
    """Progressively scaling noise interpolated across dimension[s]
    https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2"""

    dims: tuple[int] | tuple[int, int] | tuple[int, int, int] = (-1, -2)
    strength: float = 0.3  # low by default so it doesnt grenade the average model
    static: bool = False

    def __post_init__(self):
        if self.static:
            self._static_pyramid = self.pyramid()
        else:
            self._static_pyramid = None

    def pyramid(self) -> torch.Tensor:
        "Just the added 'pyramid' component"
        dims = [len(self.shape) + d if d < 0 else d for d in self.dims]
        mask = [n in dims for n in range(len(self.shape))]

        target = tuple([s for m, s in zip(mask, self.shape) if m])
        mode = ["linear", "bilinear", "bicubic"][len(target) - 1]

        noise = torch.zeros(self.shape, dtype=self.dtype, device=self.device)

        running_shape = list(self.shape)

        for i in range(99):
            # Rather than always going 2x,
            r = torch.rand([1], dtype=self.dtype, device=self.device, generator=self.seed).item() * 2 + 2
            running_shape = [max(1, int(s / (r**i))) if m else s for m, s in zip(mask, running_shape)]

            # Reduced size noise
            variance = torch.randn(running_shape, dtype=self.dtype, device=self.device, generator=self.seed)

            # Permutation so resized dims are on end
            permutation = list(sorted(zip(mask, range(len(self.shape)), list(running_shape)), key=lambda t: t[0]))
            permuted_mask = list(map(lambda t: t[0], permutation))
            permuted_dims = list(map(lambda t: t[1], permutation))
            permuted_shape = list(map(lambda t: t[2], permutation))

            # Compact leading non-resized dims for iteration
            leading = permuted_mask.index(True)
            compact_permuation_shape = tuple([math.prod(permuted_shape[:leading])] + permuted_shape[leading:])

            # Perform the permutation and iteration, unsqueezeing because interpolate() expects B,C,H,W
            variance = variance.permute(permuted_dims).reshape(compact_permuation_shape)
            variance = torch.stack(
                [  # TODO: is there a less jank interpolate that doesnt require hellish logic?
                    torch.nn.functional.interpolate(v.unsqueeze(0).unsqueeze(0), target, mode=mode).squeeze().squeeze()
                    for v in variance
                ]
            )

            # Reverse the permutation
            unpermuted_dims = torch.tensor(permuted_dims, dtype=torch.int).argsort().tolist()
            variance = variance.reshape([compact_permuation_shape[0]] + list(target)).permute(unpermuted_dims)

            noise += variance.reshape(self.shape) * self.strength**i

            if any([s <= 1 for m, s in zip(mask, running_shape) if m]):
                break  # Lowest resolution is 1x1

        return noise

    def generate(self, step: int) -> torch.Tensor:
        if self.static and self._static_pyramid is not None:
            noise = super().generate(step) + self._static_pyramid
        else:
            noise = super().generate(step) + self.pyramid()
        return noise / noise.std()  # Scaled back to roughly unit variance


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
