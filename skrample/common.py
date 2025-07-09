import enum
import math
from collections.abc import Callable
from functools import lru_cache
from itertools import repeat
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch.types import Tensor

    SampleVar = TypeVar("SampleVar", float, NDArray[np.floating], Tensor)
    Sample = float | NDArray[np.floating] | Tensor
else:
    # Avoid pulling all of torch as the code doesn't explicitly depend on it.
    SampleVar = TypeVar("SampleVar", float, NDArray[np.floating])
    Sample = float | NDArray[np.floating]


SigmaTransform = Callable[[float], tuple[float, float]]
"Transforms a single noise sigma into a pair"

Predictor = Callable[[SampleVar, SampleVar, float, SigmaTransform], SampleVar]
"sample, output, sigma, sigma_transform"


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


def sigma_complement(sigma: float) -> tuple[float, float]:
    return sigma, 1 - sigma


def sigma_polar(sigma: float) -> tuple[float, float]:
    theta = math.atan(sigma)
    return math.sin(theta), math.cos(theta)


def predict_epsilon[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "If a model does not specify, this is usually what it needs."
    sigma_u, sigma_v = sigma_transform(sigma)
    return (sample - sigma_u * output) / sigma_v  # type: ignore


def predict_sample[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "No prediction. Only for single step afaik."
    return output


def predict_velocity[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "Rare, models will usually explicitly say they require velocity/vpred/zero terminal SNR"
    sigma_u, sigma_v = sigma_transform(sigma)
    return sigma_v * sample - sigma_u * output  # type: ignore


def predict_flow[T: Sample](sample: T, output: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    "Flow matching models use this, notably FLUX.1 and SD3"
    # TODO(beinsezii): this might need to be u * output. Don't trust diffusers
    # Our tests will fail if we do so, leaving here for now.
    return sample - sigma * output  # type: ignore


def safe_log(x: float) -> float:
    "Returns inf rather than throw an err"
    try:
        return math.log(x)
    except ValueError:
        return math.inf


def normalize(regular_array: NDArray[np.float64], start: float, end: float = 0) -> NDArray[np.float64]:
    "Rescales an array to 1..0"
    return np.divide(regular_array - end, start - end)


def regularize(normal_array: NDArray[np.float64], start: float, end: float = 0) -> NDArray[np.float64]:
    "Rescales an array from 1..0 back up"
    return normal_array * (start - end) + end


def exp[T: Sample](x: T) -> T:
    return math.e**x  # type: ignore


def sigmoid[T: Sample](array: T) -> T:
    return 1 / (1 + exp(array))  # type: ignore


def softmax[T: tuple[Sample, ...]](elems: T) -> T:
    sm = sum(map(exp, elems))
    return tuple(exp(e) / sm for e in elems)  # type: ignore


def spowf[T: Sample](x: T, f: float) -> T:
    """Computes x^f in absolute then re-applies the sign to stabilize chaotic inputs.
    More computationally expensive than plain `math.pow`"""
    return abs(x) ** f * (-1 * (x < 0) | 1)  # type: ignore


@lru_cache
def bashforth(order: int) -> tuple[float, ...]:  # tuple return so lru isnt mutable
    "Bashforth coefficients for a given order"
    return tuple(
        np.linalg.solve(
            [[(-j) ** k for j in range(order)] for k in range(order)],
            [1 / (k + 1) for k in range(order)],
        ).tolist()
    )
