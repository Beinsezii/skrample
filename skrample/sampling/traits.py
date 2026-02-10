import abc
import dataclasses

from skrample import common

from . import models


@dataclasses.dataclass(frozen=True)
class SamplingCommon:
    def merge_noise[T: common.Sample](
        self,
        sample: T,
        noise: T,
        sigma: float,
        sigma_transform: common.SigmaTransform,
    ) -> T:
        """Merge noise into a sample at a given time.
        Some old samplers used to have different implmenetations,
        but now pretty for pretty much everything this is just points to `common.merge_noise`."""
        return common.merge_noise(sample, noise, sigma, sigma_transform)


@dataclasses.dataclass(frozen=True)
class HigherOrder(abc.ABC):
    order: int = 2
    """Order of the solver.
    Higher values use more model evaluations to calcualte the sample update step.
    Compute cost varies by implementation.
    The actual order used may be less for a given schedule and step.
    Order 1 is almost always equivalent to the Euler method."""

    @staticmethod
    def min_order() -> int:
        "Minimum order that this solver will attempt to use."
        return 1

    @staticmethod
    @abc.abstractmethod
    def max_order() -> int:
        "Minimum order that this solver will attempt to use."


@dataclasses.dataclass(frozen=True)
class DerivativeTransform:
    "Common trait for sapmlers that can perform computations in a different space than the original data."

    derivative_transform: models.DiffusionModel | None = models.DataModel()  # noqa: RUF009 # is immutable
    "Transform model output to this space when computing the result."
