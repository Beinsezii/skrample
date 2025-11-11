from skrample.common import Sample, SigmaTransform, predict_epsilon, predict_flow


class DiffusionModel:
    "Implements euler method forward and backward through novel method described in https://diffusionflow.github.io/"

    @classmethod
    def predict[T: Sample](cls, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return sample

    @classmethod
    def to_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "zₜ -> z̃ₜ"
        return sample

    @classmethod
    def from_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        "z̃ₜ -> zₜ"
        return sample

    @classmethod
    def to_eta(cls, sigma: float, sigma_transform: SigmaTransform) -> float:
        "σₜ -> ηₜ"
        return sigma

    @classmethod
    def forward[T: Sample](
        cls, sample: T, output: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        """Perform the Euler method.
        z̃ₛ = z̃ₜ + output · (ηₛ - ηₜ)
        Equation (5) @ https://diffusionflow.github.io/"""
        z_t = cls.to_z(sample, sigma_from, sigma_transform)
        eta_t = cls.to_eta(sigma_from, sigma_transform)
        eta_s = cls.to_eta(sigma_to, sigma_transform)

        z_s = z_t + output * (eta_s - eta_t)
        return cls.from_z(z_s, sigma_to, sigma_transform)  # pyright: ignore [reportReturnType]

    @classmethod
    def backward[T: Sample](
        cls, sample: T, result: T, sigma_from: float, sigma_to: float, sigma_transform: SigmaTransform
    ) -> T:
        """Undo the Euler method.
        output = (z̃ₛ - z̃ₜ) / (ηₛ - ηₜ)
        Equation (5) @ https://diffusionflow.github.io/"""
        z_t = cls.to_z(sample, sigma_from, sigma_transform)
        eta_t = cls.to_eta(sigma_from, sigma_transform)
        eta_s = cls.to_eta(sigma_to, sigma_transform)

        z_s = cls.to_z(result, sigma_to, sigma_transform)
        return (z_s - z_t) / (eta_s - eta_t)  # pyright: ignore [reportReturnType]


class EpsilonModel(DiffusionModel):
    "Typically used with the variance-preserving (VP) noise schedule"

    @classmethod
    def predict[T: Sample](cls, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return predict_epsilon(sample, output, sigma, sigma_transform)

    @classmethod
    def to_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        _sigma_t, alpha_t = sigma_transform(sigma)
        z_t = sample / alpha_t
        return z_t  # pyright: ignore [reportReturnType]

    @classmethod
    def from_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        _sigma_t, alpha_t = sigma_transform(sigma)
        z_t = sample * alpha_t
        return z_t  # pyright: ignore [reportReturnType]

    @classmethod
    def to_eta(cls, sigma: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma)
        eta_t = sigma_t / alpha_t
        return eta_t


class FlowModel(DiffusionModel):
    "Typically used with the linear noise schedule"

    @classmethod
    def predict[T: Sample](cls, sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return predict_flow(sample, output, sigma, sigma_transform)

    @classmethod
    def to_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return sample / (alpha_t + sigma_t)  # pyright: ignore [reportReturnType]

    @classmethod
    def from_z[T: Sample](cls, sample: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        sigma_t, alpha_t = sigma_transform(sigma)
        return sample * (alpha_t + sigma_t)  # pyright: ignore [reportReturnType]

    @classmethod
    def to_eta(cls, sigma: float, sigma_transform: SigmaTransform) -> float:
        sigma_t, alpha_t = sigma_transform(sigma)
        return sigma_t / (alpha_t + sigma_t)


type ModelTransform = DiffusionModel | type[DiffusionModel]
