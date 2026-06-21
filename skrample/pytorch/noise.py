import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import torch

from skrample.common import Step, divf, rescale_positive


@dataclass(frozen=True)
class TensorNoiseProps:
    """Configurable properties for the noise generator.
    Re-use this data structure, not the generator itself."""


@dataclass
class SkrampleTensorNoise(ABC):
    @abstractmethod
    def generate(self, step: Step | None) -> torch.Tensor:
        """Next noise tensor in the sequence.
        May raise an exception if at the end of sequence.
        Should be assumed to be stateful, and not used for multiple jobs"""
        raise NotImplementedError


@dataclass
class TensorNoiseCommon[T: TensorNoiseProps | None](SkrampleTensorNoise):
    "Common properties and helpers for most base generators."

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
        props: T = None,  # ty: ignore # is ABC
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        """Create the noise agnostically from common inputs typically available during inference.
        It is strongly recommended to set `ramp` to the sigma/noise schedule if available."""
        raise NotImplementedError


@dataclass
class Random(TensorNoiseCommon[None]):
    """Pure random noise on a normal distribution.
    Sugar for torch.randn"""

    @classmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        return cls(shape, seed, dtype, props)

    def generate(self, step: Step | None) -> torch.Tensor:
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

    def generate(self, step: Step | None) -> torch.Tensor:
        if self.props.static and self.static_offset is not None:
            offset = self.static_offset
        else:
            offset = self.offset()
        return self._randn() + offset


@dataclass(frozen=True)
class PyramidProps(OffsetProps):
    dims: tuple[int] | tuple[int, int] | tuple[int, int, int] = (-1, -2)
    strength: float = 0.3  # low by default so it doesnt grenade the average model

    depth: int = 99
    "Maximum depth of pyramid steps, from the top"


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
    ) -> Self:
        return cls(shape, seed, dtype, props)

    def pyramid(self) -> torch.Tensor:
        "Just the added 'pyramid' component"
        dims = [len(self.shape) + d if d < 0 else d for d in self.props.dims]
        mask = [n in dims for n in range(len(self.shape))]

        target = tuple([s for m, s in zip(mask, self.shape) if m])
        mode = ["linear", "bilinear", "bicubic"][len(target) - 1]

        noise = torch.zeros(self.shape, dtype=self.dtype, device=self.seed.device)
        pyramid_steps: list[torch.Tensor] = []

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
            compact_permuation_shape = (math.prod(permuted_shape[:leading]), *permuted_shape[leading:])

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
            variance = variance.reshape(
                # If there's no leading dims, compact[0] is an extra `1` size dim from prod([]) so it must be excluded
                [compact_permuation_shape[0], *target] if leading > 0 else target
            ).permute(unpermuted_dims)

            pyramid_steps.append(variance.reshape(self.shape) * self.props.strength**i)

            if any(s <= 1 for m, s in zip(mask, running_shape) if m):
                break  # Lowest resolution is 1x1

        steps = len(pyramid_steps) - 1
        skip = min(steps, max(0, steps - self.props.depth))
        return noise + sum(pyramid_steps[skip:])

    def generate(self, step: Step | None) -> torch.Tensor:
        if self.props.static and self._static_pyramid is not None:
            noise = self._randn() + self._static_pyramid
        else:
            noise = self._randn() + self.pyramid()
        return noise / noise.std()  # Scaled back to roughly unit variance


@dataclass(frozen=True)
class BrownianProps(TensorNoiseProps):
    max_steps: int = 10_000
    """Target resolution of the brownian tree.
    DT sizes below 1/max_steps run the risk of failing to generate.
    Increasing this negatively impacts performance."""


@dataclass
class Brownian(TensorNoiseCommon[BrownianProps]):
    """Uses torchsde.BrownianInterval to generate noise deterministically over Step"""

    def __post_init__(self) -> None:
        import torchsde

        self._tree = torchsde.BrownianInterval(
            t0=0,
            t1=1,
            size=self.shape,
            entropy=self.seed.initial_seed(),
            dtype=self.dtype,
            device=self.seed.device,
            halfway_tree=True,
            tol=1 / (self.props.max_steps * 10),  # 1 order of magnitude more than min step size
            pool_size=2**6,  # tolerance is 99% of the perf hit at this size
            cache_size=round(math.log2(self.props.max_steps * 10) * 1.3),  # binary for halfway + 30%
        )

    def generate(self, step: Step | None) -> torch.Tensor:
        if not step:
            return self._randn()
        step = step.normal().clamp()
        return self._tree(*step) / math.sqrt(step.distance())  # pyright: ignore[reportOperatorIssue]

    @classmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: BrownianProps = BrownianProps(),
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        return cls(shape=shape, seed=seed, dtype=dtype, props=props)


@dataclass(frozen=True)
class ColoredProps(TensorNoiseProps):
    energy: float | None = None
    """Target standard deviation of the output tensor, effectively the scale of noise.
    When `None`, noise is normalized back to uncolored variance."""

    color_start: float = 1 / 4
    """Power-law exponent at the beginning of the schedule (`step` = None).
    Higher values produce redder (lower frequency) noise,
    lower values produce bluer (higher frequency) noise."""
    color_end: float = -2
    """Power-law exponent at the end of the schedule (`step.time_to` = 1).
    Higher values produce redder (lower frequency) noise,
    lower values produce bluer (higher frequency) noise."""
    color_curve: float = 2
    """Curvature of power-law exponent gradient, similar to FlowShift.
    Higher values bias `color_start`, lower values bias `color_end`."""


@dataclass
class Colored(TensorNoiseCommon[ColoredProps]):
    """Power-law colored noise generator with schedule-driven exponent interpolation.

    Generates noise whose power spectrum follows `f^{-exponent/2}` by shaping
    white noise in the Fourier domain.  The `exponent` is interpolated between
    `color_start` and `color_end` as a function of the diffusion step,
    so the color of the noise evolves over the generation timeline.
    """

    @staticmethod
    def _radial_freq_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Build a normalized radial-frequency tensor matching rfftn output shape.

        For `rfftn(x)` on a tensor of spatial shape `(D₁, …, Dₙ)`, the complex
        output has shape `(D₁, …, Dₙ//2+1)` when the last dim is even (full size
        if odd).  This function returns a radius grid of *exactly* that trailing-D
        shape so it broadcasts naturally over any leading batch/channel dims.

        Values are in `[0, 1]` with 0 = DC and 1 = the farthest Nyquist bin.

        Parameters
        ---
        shape : torch.Size
            The spatial shape of the tensor being transformed.
        device : torch.device
            Where to allocate frequency tensors.

        Returns
        ---
        torch.Tensor
            Normalized radial frequency grid.

        Notes
        ---
        Implementation by Qwen 3.6 27B
        """
        ndim = len(shape)

        # Build per-axis frequency coordinates in normalized form.
        # rFFT always keeps only the non-redundant half on the last axis (N//2+1 bins).
        freqs_per_axis: list[torch.Tensor] = []
        for i, dim in enumerate(shape):
            if i == ndim - 1:
                # Last axis: rFFT output has N//2 + 1 non-redundant frequency bins [0 .. N/2]
                n_bins = dim // 2 + 1
                idx = torch.arange(n_bins, device=device)
                freqs_per_axis.append(idx / dim)  # normalized [0, 0.5]
            else:
                # Other axes: full FFT - use abs(fftfreq) for radial distance symmetry
                freqs_per_axis.append(torch.fft.fftfreq(dim, d=1.0, device=device).abs())

        # meshgrid → stack → radial norm
        grid = torch.stack(torch.meshgrid(*freqs_per_axis, indexing="ij"), dim=-1)
        radius = grid.norm(p=2, dim=-1)

        # Normalize to [0, 1]
        r_max = radius.max()
        if r_max > 0:
            radius = radius / r_max

        return radius  # shape exactly matches trailing ndim of rfftn output

    @staticmethod
    def colorize_noise(white: torch.Tensor, exponent: float = 0.0, energy: float | None = None) -> torch.Tensor:
        """Colors the input white noise according to the Gaussian power-law spectrum `f^{-exponent}`.

        Takes an existing white-noise tensor and colors it in the Fourier
        domain so that its amplitude falls (or rises) with radial frequency.
        The result is normalized back to input deviation (or to `energy`, if provided).

        Single element dimensions are excluded from FFT.
        Batching is NOT accounted for. Batched tensors must be passed individually.

        Examples
        ---
        >>> import torch
        >>> white = torch.randn(64, 64)
        >>>
        >>> # Pink-ish noise - richer low-frequency structure
        >>> pink = Colored.colorize_noise(white, exponent=1.0)
        >>>
        >>> # Blue noise - high-frequency detail emphasized
        >>> blue = Colored.colorize_noise(white, exponent=-2.0)

        Notes
        ---
        Initial implementation by Qwen 3.6 27B
        """

        # Step 1: white noise
        wstd = white.std()

        # Fast path: t == 0 is plain white noise - no FFT overhead
        if exponent == 0.0:
            return white if energy is None or wstd < 1e-8 else white * (energy / wstd)

        w = white.squeeze()

        if w.dtype not in [torch.float32, torch.float64]:  # half/bfloat not fully supported
            w = w.to(torch.float32)

        # Step 2: forward FFT (real → complex)
        F = torch.fft.rfftn(w, norm="forward")

        # Step 3: normalized radial frequency grid
        freq_grid = Colored._radial_freq_grid(w.shape, w.device)

        # Step 4: power-law amplitude weights
        # PSD ∝ f^{-t}  ⇒  amplitude weight ∝ f^{-t/2}.
        #
        # The weight diverges at DC (f = 0).  We clip at half a frequency-bin
        # spacing in normalized coordinates, which is standard practice for
        # FFT-based colored-noise generation.  This gives the correct PSD slope
        # away from DC while keeping only one bin per radial direction clamped.
        N_eff = sum(w.shape) / len(w.shape) if w.shape else 1.0
        eps_clip = 0.5 / max(N_eff, 4.0)

        weights = torch.clamp(freq_grid, min=eps_clip) ** (-exponent / 2.0)

        # Step 5: multiply in Fourier domain
        F_colored = F * weights

        # Step 6: inverse FFT to spatial domain
        colored = torch.fft.irfftn(F_colored, s=w.shape, norm="forward")

        # Step 7: renormalize to input std (variance conservation)
        cstd = colored.std()
        if cstd > 1e-8:
            colored *= wstd / cstd if energy is None else energy / cstd

        return colored.view(white.shape).to(dtype=white.dtype)

    def generate(self, step: Step | None) -> torch.Tensor:
        noise = self._randn()

        if step is None:
            exponent = self.props.color_start  # t=0 equivalent
        elif self.props.color_curve == math.inf:
            exponent = self.props.color_end  # infinite curve makes a flat line
        else:
            step = step.normal().clamp()  # enforce 0..=1
            t = step.time_to  # t>0
            # Negative curve to match FlowShift since step is ascending more like alpha than sigma
            shift = rescale_positive(-self.props.color_curve)
            t = shift / (shift + (divf(1, t) - 1))
            exponent = (1 - t) * self.props.color_start + t * self.props.color_end

        # will short-circuit for exponent 0, but still has energy target
        noise = self.colorize_noise(noise, exponent=exponent, energy=self.props.energy)

        return noise

    @classmethod
    def from_inputs(
        cls,
        shape: tuple[int, ...],
        seed: torch.Generator,
        props: ColoredProps = ColoredProps(),
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        return cls(shape=shape, seed=seed, dtype=dtype, props=props)


@dataclass
class BatchTensorNoise[T: TensorNoiseProps | None](SkrampleTensorNoise):
    """Helper class for producing batches of noise while maintaining seeds across individual batch items.
    Manages N noise classes at once, returning the results in a stack."""

    generators: list[TensorNoiseCommon[T]]

    def generate(self, step: Step | None) -> torch.Tensor:
        return torch.stack([g.generate(step) for g in self.generators])

    @classmethod
    def from_batch_inputs[U: TensorNoiseProps | None](  # pyright fails if you use the outer generic
        cls,
        subclass: type[TensorNoiseCommon[U]],
        unit_shape: tuple[int, ...],
        seeds: list[torch.Generator],
        props: U | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "BatchTensorNoise[U]":
        """Batched equivalent of TensorNoiseCommon.from_inputs
        `unit_shape` is the shape per batch, which means the final result will be size [len(seeds), *unit_shape]"""
        return cls(  # ty: ignore  # Safe from ABC
            [
                subclass.from_inputs(unit_shape, seed, props, dtype)
                if props is not None
                else subclass.from_inputs(unit_shape, seed, dtype=dtype)  # pyright: ignore  # Safe from ABC
                for seed in seeds
            ]
        )
