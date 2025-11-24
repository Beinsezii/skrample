import abc
import dataclasses
import math
from collections.abc import Callable
from functools import wraps

from skrample.common import Sample, SigmaTransform


@dataclasses.dataclass(frozen=True)
class DiffusionModel(abc.ABC):
    """Common framework for diffusion model sampling."""

    @abc.abstractmethod
    def to_x[T: Sample](self, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "output -> X̂"

    @abc.abstractmethod
    def from_x[T: Sample](self, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "X̂ -> output"

    @abc.abstractmethod
    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        "σₜ, σₛ -> Γ"

    @abc.abstractmethod
    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        "σₜ, σₛ -> Δ"

    def forward[T: Sample](
        self, sample: T, output: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        "sample * Γ + output * Δ"
        gamma = self.gamma(sigma_from, sigma_to, sigma_transform)
        delta = self.delta(sigma_from, sigma_to, sigma_transform)
        return math.sumprod((sample, output), (gamma, delta))  # pyright: ignore [reportReturnType, reportArgumentType]

    def backward[T: Sample](
        self, sample: T, result: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        "(output - sample * Γ) / Δ"
        gamma = self.gamma(sigma_from, sigma_to, sigma_transform)
        delta = self.delta(sigma_from, sigma_to, sigma_transform)
        return (result - sample * gamma) / delta  # pyright: ignore [reportReturnType]


@dataclasses.dataclass(frozen=True)
class DataModel(DiffusionModel):
    """X-Prediction
    Predicts the clean image.
    Usually for single step models."""

    def to_x[T: Sample](self, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "output -> X̂"
        return output

    def from_x[T: Sample](self, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "X̂ -> output"
        return x

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        "σₜ, σₛ -> Γ"
        sigma_t, _alpha_t = sigma_transform(sigma_from)
        sigma_s, _alpha_s = sigma_transform(sigma_to)
        return sigma_s / sigma_t

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        "σₜ, σₛ -> Δ"
        sigma_t, alpha_t = sigma_transform(sigma_from)
        sigma_s, alpha_s = sigma_transform(sigma_to)
        return alpha_s - (alpha_t * sigma_s) / sigma_t


@dataclasses.dataclass(frozen=True)
class NoiseModel(DiffusionModel):
    """Ε-Prediction
    Predicts the added noise.
    If a model does not specify, this is usually what it needs."""  # noqa: RUF002

    def to_x[T: Sample](self, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return (sample - sigma_t * output) / alpha_t  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return (sample - alpha_t * x) / sigma_t  # pyright: ignore [reportReturnType]

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        _sigma_t, alpha_t = sigma_transform(sigma_from)
        _sigma_s, alpha_s = sigma_transform(sigma_to)
        return alpha_s / alpha_t

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma_from)
        sigma_s, alpha_s = sigma_transform(sigma_to)
        return sigma_s - (alpha_s * sigma_t) / alpha_t


@dataclasses.dataclass(frozen=True)
class FlowModel(DiffusionModel):
    """U-Prediction.
    Flow matching models use this, notably FLUX.1 and SD3"""

    def to_x[T: Sample](self, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return (sample - sigma_t * output) / (alpha_t + sigma_t)  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return (sample - (alpha_t + sigma_t) * x) / sigma_t  # pyright: ignore [reportReturnType]

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma_from)
        sigma_s, alpha_s = sigma_transform(sigma_to)
        return (sigma_s + alpha_s) / (sigma_t + alpha_t)

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma_from)
        sigma_s, alpha_s = sigma_transform(sigma_to)
        return (alpha_t * sigma_s - alpha_s * sigma_t) / (alpha_t + sigma_t)


@dataclasses.dataclass(frozen=True)
class VelocityModel(DiffusionModel):
    """V-Prediction.
    Rare, models will usually explicitly say they require velocity/vpred/zero terminal SNR"""

    def to_x[T: Sample](self, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return alpha_t * sample - sigma_t * output  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return (alpha_t * sample - x) / sigma_t  # pyright: ignore [reportReturnType]

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma_from)
        sigma_s, alpha_s = sigma_transform(sigma_to)
        return (sigma_s / sigma_t) * (1 - alpha_t * alpha_t) + alpha_s * alpha_t

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma_from)
        sigma_s, alpha_s = sigma_transform(sigma_to)
        return alpha_t * sigma_s - alpha_s * sigma_t


@dataclasses.dataclass(frozen=True)
class FakeModel(DiffusionModel):
    "Marker for transforms that are only used for alternative sampling of other models."


@dataclasses.dataclass(frozen=True)
class ScaleX(FakeModel):
    "X / Sample prediction with sampling bias"

    bias: float = 3
    """Bias for sample prediction.
    Higher values create a stronger image."""

    def x_scale(self, sigma_t: float, alpha_t: float) -> float:
        # Remap -∞ → 0 → ∞ » 0 → 1 → log(∞)
        if self.bias < 0:
            # -∞ → 0⁻ » 0⁺ → 1⁻
            factor = 1 / math.log(math.e - self.bias)
        else:
            # 0 → ∞ » 1 → log(∞)
            factor = math.log(math.e + self.bias)

        # Rescale sigma_t to average bias scale on VP and NV schedules
        sigma_mean = sigma_t / (sigma_t + alpha_t)
        return factor**sigma_mean

    def to_x[T: Sample](self, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return output * self.x_scale(sigma_t, alpha_t)  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return x / self.x_scale(sigma_t, alpha_t)  # pyright: ignore [reportReturnType]

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, _alpha_t = sigma_transform(sigma_from)
        sigma_s, _alpha_s = sigma_transform(sigma_to)
        return sigma_s / sigma_t

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma_from)
        sigma_s, alpha_s = sigma_transform(sigma_to)
        return (alpha_s - alpha_t * sigma_s / sigma_t) * self.x_scale(sigma_t, alpha_t)


@dataclasses.dataclass(frozen=True)
class ModelConvert:
    transform_from: DiffusionModel
    transform_to: DiffusionModel

    def output_to[T: Sample](self, sample: T, output_from: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        if self.transform_to is self.transform_from:
            return output_from
        else:
            return self.transform_to.from_x(
                sample,
                self.transform_from.to_x(sample, output_from, sigma, sigma_transform),
                sigma,
                sigma_transform,
            )

    def output_from[T: Sample](self, sample: T, output_to: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        if self.transform_from is self.transform_to:
            return output_to
        else:
            return self.transform_from.from_x(
                sample,
                self.transform_to.to_x(sample, output_to, sigma, sigma_transform),
                sigma,
                sigma_transform,
            )

    def wrap_model_call[T: Sample](
        self, model: Callable[[T, float, float], T], sigma_transform: SigmaTransform
    ) -> Callable[[T, float, float], T]:
        @wraps(model)
        def converted(sample: T, timestep: float, sigma: float) -> T:
            return self.output_to(sample, model(sample, timestep, sigma), sigma, sigma_transform)

        return converted
