import enum
import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import repeat
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

    type Sample = float | NDArray[np.floating] | Tensor
else:
    # Avoid pulling all of torch as the code doesn't explicitly depend on it.
    type Sample = float | NDArray[np.floating]


type FloatSchedule = Sequence[Point]
"Sequence of timestep, sigma"

type RNG[T: Sample] = Callable[[], T]
"Distribution should match model, typically normal"


class Point(NamedTuple):
    timestep: float
    sigma: float
    alpha: float


class DeltaPoint(NamedTuple):
    point_from: Point
    point_to: Point

    @property
    def sigmas(self) -> tuple[float, float]:
        return self.point_from.sigma, self.point_to.sigma

    @property
    def timesteps(self) -> tuple[float, float]:
        return self.point_from.timestep, self.point_to.timestep

    def dt(self) -> Point:
        return Point(
            timestep=self.point_to.timestep - self.point_from.timestep,
            sigma=self.point_to.sigma - self.point_from.sigma,
            alpha=self.point_to.alpha - self.point_from.alpha,
        )


class Step(NamedTuple):
    """Structured tuple representing two points in time, or one step of sampling.
    Internally represented as a normal range 0.0..=1.0 for direct usage in scheduler.points
    with many compatibility methods for adapting to/from integer ranges like `for n in range(steps)`"""

    time_from: float
    "Time at which this sample was generated."
    time_to: float
    "Time at which we are sampling to."

    @staticmethod
    def from_int(position: int, amount: int) -> "Step":
        "Convert integer steps into a time based representation."
        return Step(position / amount, (position + 1) / amount)

    def distance(self) -> float:
        "Distance of time_from -> time_to"
        return self.time_to - self.time_from

    def offset(self, steps: int | float) -> "Step":
        """Roll this step forward or backward by some amount of steps
        Does not perform bounds checking, add .clamp() if you need to ensure the position is never > amount"""
        offset = self.distance() * steps
        return Step(self.time_from + offset, self.time_to + offset)

    def clamp(self) -> "Step":
        """Ensures position is not less than zero or not more than amount-1
        This ensures distance() will always be > 0, ie position < amount."""
        return Step(clamp(self.time_from, high=1 - self.distance()), clamp(self.time_to, low=self.distance()))

    def position(self) -> float:
        """Compute the denormalized position/index of this step in the total.
        Roughly an inverse of `from_int`"""
        return self.time_from / self.distance()

    def amount(self) -> float:
        """Compute the denormalized total amount of steps.
        Roughly an inverse of `from_int`"""
        return 1 / self.distance()

    def normal(self) -> "Step":
        "Ensures the direction time_from -> time_to is positive, ie time is always moving forwards."
        return Step(min(self), max(self))


@enum.unique
class MergeStrategy(enum.StrEnum):  # str for easy UI options
    "Control how two lists should be merged"

    Ours = enum.auto()
    "Only our list"
    Theirs = enum.auto()
    "Only their list"
    After = enum.auto()
    "Their list after our list"
    Before = enum.auto()
    "Their before our list"
    UniqueAfter = enum.auto()
    "Their list after our list, excluding duplicates from theirs"
    UniqueBefore = enum.auto()
    "Their before our list, excluding duplicates from ours"

    def merge[T](self, ours: list[T], theirs: list[T], cmp: Callable[[T, T], bool] = lambda a, b: a == b) -> list[T]:
        match self:
            case MergeStrategy.Ours:
                return ours
            case MergeStrategy.Theirs:
                return theirs
            case MergeStrategy.After:
                return ours + theirs
            case MergeStrategy.Before:
                return theirs + ours
            case MergeStrategy.UniqueAfter:
                return ours + [i for i in theirs if not any(map(cmp, ours, repeat(i)))]
            case MergeStrategy.UniqueBefore:
                return theirs + [i for i in ours if not any(map(cmp, theirs, repeat(i)))]


def merge_noise[T: Sample](sample: T, noise: T, point: Point) -> T:
    _t, sigma_u, sigma_v = point
    return sample * sigma_v + noise * sigma_u  # pyright: ignore [reportReturnType] # float rhs is always T


def divf(lhs: float, rhs: float) -> float:
    "Float division with infinity"
    if rhs != 0:
        return lhs / rhs
    elif lhs == 0:
        raise ZeroDivisionError
    else:
        return math.copysign(math.inf, lhs)


def ln(x: float) -> float:
    "Natural logarithm with infinity"
    if x > 0:
        return math.log(x)
    elif x < 0:
        raise ValueError
    else:
        return -math.inf


def normalize[T: Sample](regular: T, start: float, end: float = 0) -> T:
    "Rescales an array to 1..0"
    return (regular - end) / (start - end)  # pyright: ignore [reportReturnType] # float rhs is always T


def regularize[T: Sample](normal: T, start: float, end: float = 0) -> T:
    "Rescales an array from 1..0 back up"
    return normal * (start - end) + end  # pyright: ignore [reportReturnType] # float rhs is always T


def rescale_positive(x: float) -> float:
    "-inf..inf -> 0..inf"
    return (abs(x) + 1) ** math.copysign(1, x)


def rescale_subnormal(x: float) -> float:
    "-inf..inf -> -1..1"
    return math.copysign(1 - (abs(x) + 1) ** -1, x)


def exp[T: Sample](x: T) -> T:
    return math.e**x  # pyright: ignore [reportReturnType] # float rhs is always T


def sigmoid[T: Sample](array: T) -> T:
    arrexp: T = exp(array)
    return arrexp / (1 + arrexp)  # pyright: ignore [reportReturnType] # float rhs is always T


def softmax[T: tuple[Sample, ...]](elems: T) -> T:
    sm = sum(map(exp, elems))  # ty: ignore # tuple is always __iter__
    return tuple(exp(e) / sm for e in elems)  # type: ignore # tuple always same len


def spowf[T: Sample](x: T, f: float) -> T:
    """Computes x^f in absolute then re-applies the sign to stabilize chaotic inputs.
    More computationally expensive than plain `math.pow`"""
    return abs(x) ** f * (-1 * (x < 0) | 1)  # pyright: ignore [reportReturnType] # float rhs is always T


def mean(x: Sample) -> float:
    "For an array this returns mean().item(). For a float this returns x"
    if isinstance(x, float | int):
        return x
    else:
        return x.mean().item()


def clamp(x: float, low: float = 0, high: float = 1) -> float:
    return max(low, min(high, x))


@lru_cache
def bashforth(order: int) -> tuple[float, ...]:  # tuple return so lru isnt mutable
    "Bashforth coefficients for a given order"
    return tuple(
        np.linalg.solve(
            [[(-j) ** k for j in range(order)] for k in range(order)],
            [1 / (k + 1) for k in range(order)],
        ).tolist()
    )
