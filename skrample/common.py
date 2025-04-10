import enum
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


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
                return ours + [i for i in theirs if not any(map(cmp, ours, [i] * len(theirs)))]
            case MergeStrategy.UniqueBefore:
                return theirs + [i for i in ours if not any(map(cmp, theirs, [i] * len(ours)))]


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


def sigmoid(array: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(array))
