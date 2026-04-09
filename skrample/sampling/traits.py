import abc
import dataclasses

from skrample import common

from . import models


@dataclasses.dataclass(frozen=True)
class SamplingCommon:
    def add_noise[T: common.Sample](self, sample: T, noise: T, point: common.Point) -> T:
        """Merge noise into a sample at a given time.
        Some old samplers used to have different implementations,
        but now pretty much everything is just an alias to Point."""
        return point.add_noise(sample, noise)

    def remove_noise[T: common.Sample](self, sample: T, noise: T, point: common.Point) -> T:
        """Merge noise into a sample at a given time.
        Some old samplers used to have different implementations,
        but now pretty much everything is just an alias to Point."""
        return point.remove_noise(sample, noise)


@dataclasses.dataclass(frozen=True)
class HigherOrder(abc.ABC):
    order: int = 2
    """Order of the solver.
    Higher values use more model evaluations to calculate the sample update step.
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
class Stochastic:
    stochasticity: float = 0
    """0 for a fully deterministic ODE,
    1 for a fully stochastic SDE"""


@dataclasses.dataclass(frozen=True)
class DerivativeTransform:
    "Common trait for samplers that can perform computations in a different space than the original data."

    derivative_transform: models.DiffusionModel | None = models.DataModel()  # noqa: RUF009 # is immutable
    "Transform model output to this space when computing the result."


@dataclasses.dataclass(frozen=True)
class UnifiedModelling(DerivativeTransform, Stochastic, HigherOrder):
    "Joint class of the most common traits that take advantage of skrample's modelling system for a consistent MRO"
