import abc
import dataclasses
import math
from collections.abc import Callable
from functools import wraps

from skrample.common import Sample, SigmaSA, SigmaTransform, SigmaTS


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
    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        "σₜ, σₛ, η -> Γ"

    @abc.abstractmethod
    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        "σₜ, σₛ, η -> Δ"

    def zeta_ts(self, sigma_ts: SigmaTS, eta: float = 1.0, epsilon: float = 1e-8) -> float:
        """Co-authored by Gemini 3.1 Pro"""
        if abs(eta) < epsilon or abs(sigma_ts.s.sigma) < epsilon:  # both of these collapse output to zero
            return 0

        # Universal conditional variance mapping
        ratio = (sigma_ts.t.alpha * sigma_ts.s.sigma) / (sigma_ts.s.alpha * sigma_ts.t.sigma)
        variance = (sigma_ts.s.sigma**2) * (1.0 - ratio**2)
        return eta * math.sqrt(max(0.0, variance))

    def zeta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 1.0) -> float:
        "σₜ, σₛ, η -> ζ"
        return self.zeta_ts(SigmaTS(sigma_transform(sigma_from), sigma_transform(sigma_to)), eta)

    def sigma_transform_eta(
        self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0
    ) -> SigmaTS:
        "Co-authored by Gemini 3.1 Pro"
        sa_t, sa_s = sigma_transform(sigma_from), sigma_transform(sigma_to)

        if (zeta := self.zeta_ts(SigmaTS(sa_t, sa_s), eta)) != 0:  # zeta_ts already checks <1e-8
            sa_s = SigmaSA(math.sqrt(max(0.0, sa_s.sigma**2 - zeta**2)), sa_s.alpha)

        return SigmaTS(sa_t, sa_s)

    def forward[T: Sample](
        self,
        sample: T,
        output: T,
        sigma_from: float,
        sigma_to: float,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        eta: float = 0,
    ) -> T:
        "sample * Γ + output * Δ + noise * ζ"
        gamma = self.gamma(sigma_from, sigma_to, sigma_transform, eta)
        delta = self.delta(sigma_from, sigma_to, sigma_transform, eta)
        if noise is not None and (zeta := self.zeta(sigma_from, sigma_to, sigma_transform, eta)) != 0:
            return math.sumprod((sample, output, noise), (gamma, delta, zeta))  # type: ignore # sumprod is always T
        else:
            return math.sumprod((sample, output), (gamma, delta))  # type: ignore # sumprod is always T

    def backward[T: Sample](
        self,
        sample: T,
        result: T,
        sigma_from: float,
        sigma_to: float,
        sigma_transform: SigmaTransform,
        noise: T | None = None,
        eta: float = 0,
    ) -> T:
        "(result - sample * Γ - noise * ζ) / Δ"
        gamma = self.gamma(sigma_from, sigma_to, sigma_transform, eta)
        delta = self.delta(sigma_from, sigma_to, sigma_transform, eta)
        if noise is not None and (zeta := self.zeta(sigma_from, sigma_to, sigma_transform, eta)) != 0:
            return (result - sample * gamma - noise * zeta) / delta  # pyright: ignore [reportReturnType]
        else:
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

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return ts.s.sigma / ts.t.sigma

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return ts.s.alpha - ts.t.alpha * ts.s.sigma / ts.t.sigma


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

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        return sigma_transform(sigma_to).alpha / sigma_transform(sigma_from).alpha

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return ts.s.sigma - (ts.s.alpha * ts.t.sigma) / ts.t.alpha


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

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return (ts.s.sigma + ts.s.alpha) / (ts.t.sigma + ts.t.alpha)

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return (ts.t.alpha * ts.s.sigma - ts.s.alpha * ts.t.sigma) / (ts.t.alpha + ts.t.sigma)


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

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return (ts.s.sigma / ts.t.sigma) * (1 - ts.t.alpha * ts.t.alpha) + ts.s.alpha * ts.t.alpha

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return ts.t.alpha * ts.s.sigma - ts.s.alpha * ts.t.sigma


@dataclasses.dataclass(frozen=True)
class FakeModel(DiffusionModel):
    "Marker for transforms that are only used for alternative sampling of other models."


@dataclasses.dataclass(frozen=True)
class ScaleX(FakeModel):
    "X / Sample prediction with sampling bias"

    bias: float = 3
    """Bias for sample prediction.
    Higher values create a stronger image."""

    def x_scale(self, sigma_sa: SigmaSA) -> float:
        # > 0 increase data distance, < 0 increase noise distance
        # Negative power since sa always < 1
        # e^xt
        return math.exp(-math.log10(abs(self.bias) + 1) * (sigma_sa.sigma if self.bias < 0 else sigma_sa.alpha))

    def to_x[T: Sample](self, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return output * self.x_scale(sigma_transform(sigma))  # pyright: ignore [reportReturnType]

    def from_x[T: Sample](self, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return x / self.x_scale(sigma_transform(sigma))  # pyright: ignore [reportReturnType]

    def gamma(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return ts.s.sigma / ts.t.sigma

    def delta(self, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform, eta: float = 0) -> float:
        ts = self.sigma_transform_eta(sigma_from, sigma_to, sigma_transform, eta)
        return (ts.s.alpha - ts.t.alpha * ts.s.sigma / ts.t.sigma) * self.x_scale(ts.t)


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
