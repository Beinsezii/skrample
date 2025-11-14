import math

from skrample.common import Sample, SigmaTransform, divf, predict_epsilon, predict_flow, predict_velocity


class DiffusionModel:
    "Implements euler method forward and backward through novel method described in https://diffusionflow.github.io/"

    @classmethod
    def to_x[T: Sample](cls, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "output -> X̂"
        return output

    @classmethod
    def from_x[T: Sample](cls, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "X̂ -> output"
        return x

    @classmethod
    def to_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "zₜ -> z̃ₜ"
        return sample

    @classmethod
    def from_z[T: Sample](cls, z: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "z̃ₜ -> zₜ"
        return z

    @classmethod
    def to_eta(cls, sigma: float, sigma_transform: SigmaTransform) -> float:
        "σₜ -> ηₜ"
        return sigma

    @classmethod
    def to_h(cls, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform) -> float:
        "Shorthand for σₜ, σₛ -> ηₛ-ηₜ"
        return cls.to_eta(sigma_to, sigma_transform) - cls.to_eta(sigma_from, sigma_transform)

    @classmethod
    def forward[T: Sample](
        cls, sample: T, output: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        """Perform the Euler method.
        z̃ₛ = z̃ₜ + output · (ηₛ - ηₜ)
        Equation (5) @ https://diffusionflow.github.io/"""
        if math.isinf(h := cls.to_h(sigma_from, sigma_to, sigma_transform)):
            return cls.to_x(sample, output, sigma_from, sigma_transform)
        else:
            z_t = cls.to_z(sample, sigma_from, sigma_transform)
            z_s = z_t + output * h
            return cls.from_z(z_s, sigma_to, sigma_transform)  # pyright: ignore [reportReturnType]

    @classmethod
    def backward[T: Sample](
        cls, sample: T, result: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        """Undo the Euler method.
        output = (z̃ₛ - z̃ₜ) / (ηₛ - ηₜ)
        Equation (5) @ https://diffusionflow.github.io/"""
        if math.isinf(h := cls.to_h(sigma_from, sigma_to, sigma_transform)):
            return cls.from_x(sample, result, sigma_from, sigma_transform)
        else:
            z_t = cls.to_z(sample, sigma_from, sigma_transform)
            z_s = cls.to_z(result, sigma_to, sigma_transform)
            return (z_s - z_t) / h  # pyright: ignore [reportReturnType]


class EpsilonModel(DiffusionModel):
    "Typically used with the variance-preserving (VP) noise schedule"

    @classmethod
    def to_x[T: Sample](cls, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return predict_epsilon(sample, output, sigma, sigma_transform)

    @classmethod
    def from_x[T: Sample](cls, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        output = (sample - alpha_t * x) / sigma_t
        return output  # pyright: ignore [reportReturnType]

    @classmethod
    def to_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        _sigma_t, alpha_t = sigma_transform(sigma)
        z_t = sample / alpha_t
        return z_t  # pyright: ignore [reportReturnType]

    @classmethod
    def from_z[T: Sample](cls, z: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        _sigma_t, alpha_t = sigma_transform(sigma)
        z_t = z * alpha_t
        return z_t  # pyright: ignore [reportReturnType]

    @classmethod
    def to_eta(cls, sigma: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma)
        eta_t = divf(sigma_t, alpha_t)
        return eta_t


class FlowModel(DiffusionModel):
    "Typically used with the linear noise schedule"

    @classmethod
    def to_x[T: Sample](cls, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return predict_flow(sample, output, sigma, sigma_transform)

    @classmethod
    def from_x[T: Sample](cls, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        output = (sample - x) / sigma
        return output  # pyright: ignore [reportReturnType]

    @classmethod
    def to_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return sample / (alpha_t + sigma_t)  # pyright: ignore [reportReturnType]

    @classmethod
    def from_z[T: Sample](cls, z: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return z * (alpha_t + sigma_t)  # pyright: ignore [reportReturnType]

    @classmethod
    def to_eta(cls, sigma: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma)
        return sigma_t / (alpha_t + sigma_t)


class VelocityModel(EpsilonModel):
    """Typically used with the linear noise schedule.
    Simply converts output to Epsilon during forward/backward"""

    @classmethod
    def to_x[T: Sample](cls, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return predict_velocity(sample, output, sigma, sigma_transform)

    @classmethod
    def from_x[T: Sample](cls, sample: T, x: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        output = (alpha_t * sample - x) / sigma_t
        return output  # pyright: ignore [reportReturnType]

    @classmethod
    def forward[T: Sample](
        cls, sample: T, output: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        output = cls.to_x(sample, output, sigma_from, sigma_transform)
        output = super().from_x(sample, output, sigma_from, sigma_transform)
        return super().forward(sample, output, sigma_from, sigma_to, sigma_transform)

    @classmethod
    def backward[T: Sample](
        cls, sample: T, result: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        output: T = super().backward(sample, result, sigma_from, sigma_to, sigma_transform)
        output = super().to_x(sample, output, sigma_from, sigma_transform)
        output = cls.from_x(sample, output, sigma_from, sigma_transform)
        return output


class XModel(DiffusionModel):
    "Direct data prediction"

    @classmethod
    def to_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, _alpha_t = sigma_transform(sigma)
        z_t = sample / sigma_t
        return z_t  # pyright: ignore [reportReturnType]

    @classmethod
    def from_z[T: Sample](cls, z: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, _alpha_t = sigma_transform(sigma)
        z_t = z * sigma_t
        return z_t  # pyright: ignore [reportReturnType]

    @classmethod
    def to_eta(cls, sigma: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma)
        eta_t = divf(alpha_t, sigma_t)
        return eta_t


type ModelTransform = DiffusionModel | type[DiffusionModel]
