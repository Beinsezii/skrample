import enum
import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import repeat
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch.types import Tensor

    type Sample = float | NDArray[np.floating] | Tensor
else:
    # Avoid pulling all of torch as the code doesn't explicitly depend on it.
    type Sample = float | NDArray[np.floating]


type SigmaTransform = Callable[[float], tuple[float, float]]
"Transforms a single noise sigma into a pair"

type Predictor[S: Sample] = Callable[[S, S, float, SigmaTransform], S]
"sample, output, sigma, sigma_transform"

type DictOrProxy[T, U] = MappingProxyType[T, U] | dict[T, U]  # Mapping does not implement __or__
"Simple union type for a possibly immutable dictionary"

type FloatSchedule = Sequence[tuple[float, float]]
"Sequence of timestep, sigma"

type RNG[T: Sample] = Callable[[], T]
"Distribution should match model, typically normal"


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


def get_sigma_uv(step: int, schedule: FloatSchedule, sigma_transform: SigmaTransform) -> tuple[float, float]:
    """Gets sigma u/v with bounds check.
    If step >= len(schedule), the sigma is assumed to be zero."""
    return sigma_transform(schedule[step][1] if step < len(schedule) else 0)


def scaled_delta(sigma: float, sigma_next: float, sigma_transform: SigmaTransform) -> tuple[float, float]:
    "Returns delta (h) and scale factor to perform the euler method."
    sigma_u, sigma_v = sigma_transform(sigma)
    sigma_u_next, sigma_v_next = sigma_transform(sigma_next)

    scale = sigma_u_next / sigma_u
    delta = sigma_v_next - sigma_v * scale  # aka `h` or `dt`
    return delta, scale


def euler[T: Sample](sample: T, prediction: T, sigma: float, sigma_next: float, sigma_transform: SigmaTransform) -> T:
    "Perform the euler method using scaled_delta"
    # Returns delta, scale so prediction is first
    return math.sumprod((prediction, sample), scaled_delta(sigma, sigma_next, sigma_transform))  # type: ignore


def scaled_delta_step(
    step: int, schedule: FloatSchedule, sigma_transform: SigmaTransform, step_size: int = 1
) -> tuple[float, float]:
    """Returns delta (h) and scale factor to perform the euler method.
    If step + step_size > len(schedule), assumes the next timestep and sigma are zero"""
    step_next = step + step_size
    return scaled_delta(schedule[step][1], schedule[step_next][1] if step_next < len(schedule) else 0, sigma_transform)


def euler_step[T: Sample](
    sample: T, prediction: T, step: int, schedule: FloatSchedule, sigma_transform: SigmaTransform, step_size: int = 1
) -> T:
    "Perform the euler method using scaled_delta_step"
    return math.sumprod((prediction, sample), scaled_delta_step(step, schedule, sigma_transform, step_size))  # type: ignore


def merge_noise[T: Sample](sample: T, noise: T, sigma: float, sigma_transform: SigmaTransform) -> T:
    sigma_u, sigma_v = sigma_transform(sigma)
    return sample * sigma_v + noise * sigma_u  # type: ignore


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


def normalize(regular_array: NDArray[np.float64], start: float, end: float = 0) -> NDArray[np.float64]:
    "Rescales an array to 1..0"
    return np.divide(regular_array - end, start - end)


def regularize(normal_array: NDArray[np.float64], start: float, end: float = 0) -> NDArray[np.float64]:
    "Rescales an array from 1..0 back up"
    return normal_array * (start - end) + end


def exp[T: Sample](x: T) -> T:
    return math.e**x  # type: ignore


def sigmoid[T: Sample](array: T) -> T:
    arrexp: T = exp(array)
    return arrexp / (1 + arrexp)  # type: ignore


def softmax[T: tuple[Sample, ...]](elems: T) -> T:
    sm = sum(map(exp, elems))
    return tuple(exp(e) / sm for e in elems)  # type: ignore


def spowf[T: Sample](x: T, f: float) -> T:
    """Computes x^f in absolute then re-applies the sign to stabilize chaotic inputs.
    More computationally expensive than plain `math.pow`"""
    return abs(x) ** f * (-1 * (x < 0) | 1)  # type: ignore


def mean(x: Sample) -> float:
    "For an array this returns mean().item(). For a float this returns x"
    if isinstance(x, float | int):
        return x
    else:
        return x.mean().item()


@lru_cache
def bashforth(order: int) -> tuple[float, ...]:  # tuple return so lru isnt mutable
    "Bashforth coefficients for a given order"
    return tuple(
        np.linalg.solve(
            [[(-j) ** k for j in range(order)] for k in range(order)],
            [1 / (k + 1) for k in range(order)],
        ).tolist()
    )
