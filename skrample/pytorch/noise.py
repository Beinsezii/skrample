import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray


def schedule_to_ramp(schedule: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([[0], np.flip(schedule[:, 1])])


@dataclass(frozen=True)
class TensorNoiseProps:
    pass


@dataclass
class SkrampleTensorNoise(ABC):
    @abstractmethod
    def generate(self) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class TensorNoiseCommon[T: TensorNoiseProps | None](SkrampleTensorNoise):
    shape: tuple[int, ...]
    seed: torch.Generator
    dtype: torch.dtype
    props: T

    def _randn(self, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        return torch.randn(
            shape if shape is not None else self.shape,
            generator=self.seed,
            dtype=self.dtype,
            device=self.seed.device,
        )

    @classmethod
    @abstractmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: T = None,
        dtype: torch.dtype = torch.float32,
        ramp: NDArray[np.float64] = np.linspace(0, 1, 2, dtype=np.float64),
    ) -> Self:
        raise NotImplementedError


@dataclass
class Random(TensorNoiseCommon[None]):
    @classmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: None = None,
        dtype: torch.dtype = torch.float32,
        ramp: NDArray[np.float64] = np.linspace(0, 1, 2, dtype=np.float64),
    ) -> Self:
        return cls(shape, seed, dtype, props)

    def generate(self) -> torch.Tensor:
        return self._randn()


@dataclass(frozen=True)
class OffsetProps(TensorNoiseProps):
    dims: tuple[int, ...] = (0,)
    strength: float = 0.2  # low enough to not go boom ...usually
    static: bool = False


@dataclass
class Offset(TensorNoiseCommon[OffsetProps]):
    "Simple random offset along dimension[s]"

    @classmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: OffsetProps = OffsetProps(),
        dtype: torch.dtype = torch.float32,
        ramp: NDArray[np.float64] = np.linspace(0, 1, 2, dtype=np.float64),
    ) -> Self:
        return cls(shape, seed, dtype, props)

    def __post_init__(self) -> None:
        if self.props.static:
            self.static_offset: torch.Tensor | None = self.offset()
        else:
            self.static_offset = None

    def offset(self) -> torch.Tensor:
        shape = tuple([d if n in self.props.dims else 1 for n, d in enumerate(self.shape)])
        return self._randn(shape) * self.props.strength**2

    def generate(self) -> torch.Tensor:
        if self.props.static and self.static_offset is not None:
            offset = self.static_offset
        else:
            offset = self.offset()
        return self._randn() + offset


@dataclass(frozen=True)
class PyramidProps(OffsetProps):
    dims: tuple[int] | tuple[int, int] | tuple[int, int, int] = (-1, -2)
    strength: float = 0.3  # low by default so it doesnt grenade the average model


@dataclass
class Pyramid(TensorNoiseCommon[PyramidProps]):
    """Progressively scaling noise interpolated across dimension[s]
    https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2"""

    def __post_init__(self) -> None:
        if self.props.static:
            self._static_pyramid = self.pyramid()
        else:
            self._static_pyramid = None

    @classmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: PyramidProps = PyramidProps(),
        dtype: torch.dtype = torch.float32,
        ramp: NDArray[np.float64] = np.linspace(0, 1, 2, dtype=np.float64),
    ) -> Self:
        return cls(shape, seed, dtype, props)

    def pyramid(self) -> torch.Tensor:
        "Just the added 'pyramid' component"
        dims = [len(self.shape) + d if d < 0 else d for d in self.props.dims]
        mask = [n in dims for n in range(len(self.shape))]

        target = tuple([s for m, s in zip(mask, self.shape) if m])
        mode = ["linear", "bilinear", "bicubic"][len(target) - 1]

        noise = torch.zeros(self.shape, dtype=self.dtype, device=self.seed.device)

        running_shape = list(self.shape)

        for i in range(99):
            # Rather than always going 2x,
            r = torch.rand([1], dtype=self.dtype, device=self.seed.device, generator=self.seed).item() * 2 + 2
            running_shape = [max(1, int(s / (r**i))) if m else s for m, s in zip(mask, running_shape)]

            # Reduced size noise
            variance = torch.randn(running_shape, dtype=self.dtype, device=self.seed.device, generator=self.seed)

            # Permutation so resized dims are on end
            permutation = sorted(zip(mask, range(len(self.shape)), list(running_shape)), key=lambda t: t[0])
            permuted_mask = [t[0] for t in permutation]
            permuted_dims = [t[1] for t in permutation]
            permuted_shape = [t[2] for t in permutation]

            # Compact leading non-resized dims for iteration
            leading = permuted_mask.index(True)
            compact_permuation_shape = tuple([math.prod(permuted_shape[:leading])] + permuted_shape[leading:])

            # Perform the permutation and iteration, unsqueezeing because interpolate() expects B,C,H,W
            variance = variance.permute(permuted_dims).reshape(compact_permuation_shape)
            variance = torch.stack(
                [  # TODO(beinsezii): is there a less jank interpolate that doesnt require hellish logic?
                    torch.nn.functional.interpolate(v.unsqueeze(0).unsqueeze(0), target, mode=mode).squeeze().squeeze()
                    for v in variance
                ]
            )

            # Reverse the permutation
            unpermuted_dims = torch.tensor(permuted_dims, dtype=torch.int).argsort().tolist()
            variance = variance.reshape([compact_permuation_shape[0], *target]).permute(unpermuted_dims)

            noise += variance.reshape(self.shape) * self.props.strength**i

            if any(s <= 1 for m, s in zip(mask, running_shape) if m):
                break  # Lowest resolution is 1x1

        return noise

    def generate(self) -> torch.Tensor:
        if self.props.static and self._static_pyramid is not None:
            noise = self._randn() + self._static_pyramid
        else:
            noise = self._randn() + self.pyramid()
        return noise / noise.std()  # Scaled back to roughly unit variance


@dataclass(frozen=True)
class BrownianProps(TensorNoiseProps):
    reverse: bool = False


@dataclass
class Brownian(TensorNoiseCommon[BrownianProps]):
    ramp: NDArray[np.float64]

    def __post_init__(self) -> None:
        import torchsde

        if len(self.ramp) < 2:
            err = "Brownian.ramp must have at least two positions"
            raise ValueError(err)

        self._tree = torchsde.BrownianInterval(
            size=self.shape,
            entropy=self.seed.initial_seed(),
            dtype=self.dtype,
            device=self.seed.device,
        )

        self._step: int = 0

        # Basic sanitization to normalize 0->1
        self.ramp -= self.ramp.min()
        self.ramp /= self.ramp.max()
        if self.ramp[0] > self.ramp[-1]:
            self.ramp = np.flip(self.ramp)

    def generate(self) -> torch.Tensor:
        if self._step + 1 >= len(self.ramp):
            raise StopIteration

        if self.props.reverse:
            # - 2 because you still get the next sequentiall
            step = len(self.ramp) - self._step - 2
        else:
            step = self._step

        sigma = self.ramp[step]
        sigma_next = self.ramp[step + 1]
        self._step += 1

        return self._tree(sigma, sigma_next) / abs(sigma_next - sigma) ** 0.5

    @classmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: BrownianProps = BrownianProps(),
        dtype: torch.dtype = torch.float32,
        ramp: NDArray[np.float64] = np.linspace(0, 1, 1000, dtype=np.float64),
    ) -> Self:
        return cls(shape=shape, seed=seed, ramp=ramp, dtype=dtype, props=props)


@dataclass
class BatchTensorNoise[T: TensorNoiseProps | None](SkrampleTensorNoise):
    generators: list[TensorNoiseCommon[T]]

    def generate(self) -> torch.Tensor:
        return torch.stack([g.generate() for g in self.generators])

    @classmethod
    def from_batch_inputs[U: TensorNoiseProps | None](  # pyright fails if you use the outer generic
        cls,
        subclass: type[TensorNoiseCommon[U]],
        unit_shape: tuple[int, ...],
        seeds: list[torch.Generator],
        props: U | None = None,
        dtype: torch.dtype = torch.float32,
        ramp: NDArray[np.float64] = np.linspace(0, 1, 2, dtype=np.float64),
    ) -> "BatchTensorNoise[U]":
        return cls(
            [
                subclass.from_inputs(unit_shape, seed, props, dtype, ramp)
                if props is not None
                else subclass.from_inputs(
                    unit_shape,
                    seed,
                    dtype=dtype,
                    ramp=ramp,
                )  # type: ignore  # Safe from ABC
                for seed in seeds
            ]
        )
