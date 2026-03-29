import abc
import dataclasses
import math
from collections.abc import Callable
from functools import wraps

from skrample.common import DeltaPoint, Point, Sample


@dataclasses.dataclass(frozen=True)
class DiffusionModel(abc.ABC):
    """Common framework for diffusion model sampling."""

    @abc.abstractmethod
    def to_x[T: Sample](self, sample: T, output: T, point: Point) -> T:
        "output -> X̂"

    @abc.abstractmethod
    def from_x[T: Sample](self, sample: T, x: T, point: Point) -> T:
        "X̂ -> output"

    @abc.abstractmethod
    def gamma(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        "σₜ, σₛ, η -> Γ"

    @abc.abstractmethod
    def delta(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        "σₜ, σₛ, η -> Δ"

    def zeta_ts(self, delta: DeltaPoint, eta: float = 1.0, epsilon: float = 1e-8) -> float:
        """Co-authored by Gemini 3.1 Pro"""
        if abs(eta) < epsilon or abs(delta.point_to.sigma) < epsilon:  # both of these collapse output to zero
            return 0

        # Universal conditional variance mapping
        ratio = (delta.point_from.alpha * delta.point_to.sigma) / (delta.point_to.alpha * delta.point_from.sigma)
        variance = (delta.point_to.sigma**2) * (1.0 - ratio**2)
        return eta * math.sqrt(max(0.0, variance))

    def zeta(self, point_from: Point, point_to: Point, eta: float = 1.0) -> float:
        "σₜ, σₛ, η -> ζ"
        return self.zeta_ts(DeltaPoint(point_from, point_to), eta)

    def sigma_transform_eta(self, point_from: Point, point_to: Point, eta: float = 0) -> DeltaPoint:
        "Co-authored by Gemini 3.1 Pro"
        sa_t, sa_s = point_from, point_to

        if (zeta := self.zeta_ts(DeltaPoint(sa_t, sa_s), eta)) != 0:  # zeta_ts already checks <1e-8
            sa_s = Point(sa_s.timestep, math.sqrt(max(0.0, sa_s.sigma**2 - zeta**2)), sa_s.alpha)

        return DeltaPoint(sa_t, sa_s)

    def forward[T: Sample](
        self,
        sample: T,
        output: T,
        point_from: Point,
        point_to: Point,
        noise: T | None = None,
        eta: float = 0,
    ) -> T:
        "sample * Γ + output * Δ + noise * ζ"
        gamma = self.gamma(point_from, point_to, eta)
        delta = self.delta(point_from, point_to, eta)
        if noise is not None and (zeta := self.zeta(point_from, point_to, eta)) != 0:
            return math.sumprod((sample, output, noise), (gamma, delta, zeta))  # type: ignore # sumprod is always T
        else:
            return math.sumprod((sample, output), (gamma, delta))  # type: ignore # sumprod is always T

    def backward[T: Sample](
        self,
        sample: T,
        result: T,
        point_from: Point,
        point_to: Point,
        noise: T | None = None,
        eta: float = 0,
    ) -> T:
        "(result - sample * Γ - noise * ζ) / Δ"
        gamma = self.gamma(point_from, point_to, eta)
        delta = self.delta(point_from, point_to, eta)
        if noise is not None and (zeta := self.zeta(point_from, point_to, eta)) != 0:
            return (result - sample * gamma - noise * zeta) / delta  # pyright: ignore [reportReturnType]
        else:
            return (result - sample * gamma) / delta  # pyright: ignore [reportReturnType]


@dataclasses.dataclass(frozen=True)
class DataModel(DiffusionModel):
    """X-Prediction
    Predicts the clean image.
    Usually for single step models."""

    def to_x[T: Sample](self, sample: T, output: T, point: Point) -> T:
        "output -> X̂"
        return output

    def from_x[T: Sample](self, sample: T, x: T, point: Point) -> T:
        "X̂ -> output"
        return x

    def gamma(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return ts.point_to.sigma / ts.point_from.sigma

    def delta(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return ts.point_to.alpha - ts.point_from.alpha * ts.point_to.sigma / ts.point_from.sigma


@dataclasses.dataclass(frozen=True)
class NoiseModel(DiffusionModel):
    """Ε-Prediction
    Predicts the added noise.
    If a model does not specify, this is usually what it needs."""  # noqa: RUF002

    def to_x[T: Sample](self, sample: T, output: T, point: Point) -> T:
        _timestep, sigma_t, alpha_t = point
        return (sample - sigma_t * output) / alpha_t  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, point: Point) -> T:
        _timestep, sigma_t, alpha_t = point
        return (sample - alpha_t * x) / sigma_t  # pyright: ignore [reportReturnType]

    def gamma(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        return point_to.alpha / point_from.alpha

    def delta(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return ts.point_to.sigma - (ts.point_to.alpha * ts.point_from.sigma) / ts.point_from.alpha


@dataclasses.dataclass(frozen=True)
class FlowModel(DiffusionModel):
    """U-Prediction.
    Flow matching models use this, notably FLUX.1 and SD3"""

    def to_x[T: Sample](self, sample: T, output: T, point: Point) -> T:
        _timestep, sigma_t, alpha_t = point
        return (sample - sigma_t * output) / (alpha_t + sigma_t)  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, point: Point) -> T:
        _timestep, sigma_t, alpha_t = point
        return (sample - (alpha_t + sigma_t) * x) / sigma_t  # pyright: ignore [reportReturnType]

    def gamma(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return (ts.point_to.sigma + ts.point_to.alpha) / (ts.point_from.sigma + ts.point_from.alpha)

    def delta(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return (ts.point_from.alpha * ts.point_to.sigma - ts.point_to.alpha * ts.point_from.sigma) / (
            ts.point_from.alpha + ts.point_from.sigma
        )


@dataclasses.dataclass(frozen=True)
class VelocityModel(DiffusionModel):
    """V-Prediction.
    Rare, models will usually explicitly say they require velocity/vpred/zero terminal SNR"""

    def to_x[T: Sample](self, sample: T, output: T, point: Point) -> T:
        _timestep, sigma_t, alpha_t = point
        return alpha_t * sample - sigma_t * output  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, point: Point) -> T:
        _timestep, sigma_t, alpha_t = point
        return (alpha_t * sample - x) / sigma_t  # pyright: ignore [reportReturnType]

    def gamma(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return (ts.point_to.sigma / ts.point_from.sigma) * (
            1 - ts.point_from.alpha * ts.point_from.alpha
        ) + ts.point_to.alpha * ts.point_from.alpha

    def delta(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return ts.point_from.alpha * ts.point_to.sigma - ts.point_to.alpha * ts.point_from.sigma


@dataclasses.dataclass(frozen=True)
class FakeModel(DiffusionModel):
    "Marker for transforms that are only used for alternative sampling of other models."


@dataclasses.dataclass(frozen=True)
class ScaleX(FakeModel):
    "X / Sample prediction with sampling bias"

    bias: float = 3
    """Bias for sample prediction.
    Higher values create a stronger image."""

    def x_scale(self, point: Point) -> float:
        # > 0 increase data distance, < 0 increase noise distance
        # Negative power since sa always < 1
        # e^xt
        return math.exp(-math.log10(abs(self.bias) + 1) * (point.sigma if self.bias < 0 else point.alpha))

    def to_x[T: Sample](self, sample: T, output: T, point: Point) -> T:
        return output * self.x_scale(point)  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, point: Point) -> T:
        return x / self.x_scale(point)  # pyright: ignore [reportReturnType]

    def gamma(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return ts.point_to.sigma / ts.point_from.sigma

    def delta(self, point_from: Point, point_to: Point, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(point_from, point_to, eta)
        return (ts.point_to.alpha - ts.point_from.alpha * ts.point_to.sigma / ts.point_from.sigma) * self.x_scale(
            ts.point_from
        )


@dataclasses.dataclass(frozen=True)
class ModelConvert:
    transform_from: DiffusionModel
    transform_to: DiffusionModel

    def output_to[T: Sample](self, sample: T, output_from: T, point: Point) -> T:
        if self.transform_to is self.transform_from:
            return output_from
        else:
            return self.transform_to.from_x(sample, self.transform_from.to_x(sample, output_from, point), point)

    def output_from[T: Sample](self, sample: T, output_to: T, point: Point) -> T:
        if self.transform_from is self.transform_to:
            return output_to
        else:
            return self.transform_from.from_x(sample, self.transform_to.to_x(sample, output_to, point), point)

    def wrap_model_call[T: Sample](
        self, model: Callable[[T, float, float, float], T]
    ) -> Callable[[T, float, float, float], T]:
        @wraps(model)
        def converted(x: T, t: float, s: float, a: float) -> T:
            return self.output_to(x, model(x, t, s, a), Point(t, s, a))

        return converted
