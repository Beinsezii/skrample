import enum
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


@enum.unique
class MergeStrategy(enum.StrEnum):  # str for easy UI options
    Ours = enum.auto()
    Theirs = enum.auto()
    After = enum.auto()
    Before = enum.auto()
    UniqueAfter = enum.auto()
    UniqueBefore = enum.auto()

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
                return ours + [i for i in theirs if not any(map(cmp, ours, [i] * len(theirs)))]
            case MergeStrategy.UniqueBefore:
                return theirs + [i for i in ours if not any(map(cmp, theirs, [i] * len(ours)))]


def safe_log(x: float) -> float:
    try:
        return math.log(x)
    except ValueError:
        return math.inf


def normalize(regular_array: NDArray[np.float64], start: float = 1, end: float = 0) -> NDArray[np.float64]:
    return (regular_array - end) / (start - end)


def regularize(normal_array: NDArray[np.float64], start: float = 1, end: float = 0) -> NDArray[np.float64]:
    return normal_array * (start - end) + end


def sigmoid(array: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(array))
