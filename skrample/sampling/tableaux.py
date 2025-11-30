import abc
import dataclasses
import enum
import math
from typing import Protocol

type TabNode = tuple[float, tuple[float, ...]]
type TabWeight = tuple[float, ...]

type Tableau = tuple[
    tuple[TabNode, ...],
    TabWeight,
]
type ExtendedTableau = tuple[
    tuple[TabNode, ...],
    TabWeight,
    TabWeight,
]


def validate_tableau(tab: Tableau | ExtendedTableau, tolerance: float = 1e-15) -> None | IndexError | ValueError:
    for index, node in enumerate(tab[0]):
        if index != (node_len := len(node[1])):
            return IndexError(f"{index=}, {node_len=}, {node=}")
        if tolerance < (node_err := abs(node[0] - math.fsum(node[1]))):
            return ValueError(f"{tolerance=}, {node_err=}, {node=}")

    for weight in tab[1:]:
        if (node_count := len(tab[0])) != (weight_len := len(weight)):
            return IndexError(f"{node_count=}, {weight_len=}, {weight=}")
        if tolerance < (weight_err := abs(1 - math.fsum(weight))):
            return ValueError(f"{tolerance=}, {weight_err=}, {weight=}")


def rk2_tableau(c1: float) -> Tableau:
    "Create a generic 2nd order Tableau from a given coefficient."
    return (
        (
            (0.0, ()),
            (c1, (c1,)),
        ),
        (1 - 1 / (2 * c1), 1 / (2 * c1)),
    )


def rk3_tableau(c1: float, c2: float) -> Tableau:
    "Create a generic 3rd order Tableau from given coefficients."
    return (
        (
            (0.0, ()),
            (c1, (c1,)),
            (c2, (c2 / c1 * ((c2 - 3 * c1 * (1 - c1)) / (3 * c1 - 2)), -c2 / c1 * ((c2 - c1) / (3 * c1 - 2)))),
        ),
        (
            1 - (3 * c1 + 3 * c2 - 2) / (6 * c1 * c2),
            (3 * c2 - 2) / (6 * c1 * (c2 - c1)),
            (2 - 3 * c1) / (6 * c2 * (c2 - c1)),
        ),
    )


def rk4_tableau(c1: float, c2: float) -> Tableau:
    """Create a generic 4th order Tableau from 3 coefficients.
    1/2, 1/2 (Classic) is a special case and cannot be computed using this function.
    https://pages.hmc.edu/ruye/MachineLearning/lectures/ch5/node10.html"""

    ### Automatically transcribed from website using QwenVL 235B Thinking

    D = 6 * c1 * c2 - 4 * (c1 + c2) + 3

    # Compute b coefficients
    b2 = (2 * c2 - 1) / (12 * c1 * (c2 - c1) * (1 - c1))
    b3 = (2 * c1 - 1) / (12 * c2 * (c1 - c2) * (1 - c2))
    b4 = D / (12 * (1 - c1) * (1 - c2))
    b1 = 1 - b2 - b3 - b4

    # Compute a31 and a32
    a32 = c2 * (c1 - c2) / (2 * c1 * (2 * c1 - 1))
    a31 = c2 - a32

    # Compute a41, a42, a43
    num_a42 = (4 * c2**2 - 5 * c2 - c1 + 2) * (1 - c1)
    denom_a42 = 2 * c1 * (c1 - c2) * D
    a42 = num_a42 / denom_a42

    num_a43 = (2 * c1 - 1) * (1 - c1) * (1 - c2)
    denom_a43 = c2 * (c1 - c2) * D
    a43 = num_a43 / denom_a43

    a41 = 1 - a42 - a43

    stages = (
        (0.0, ()),
        (c1, (c1,)),  # a21 = c1
        (c2, (a31, a32)),
        (1.0, (a41, a42, a43)),
    )

    b_vector = (b1, b2, b3, b4)

    return (stages, b_vector)


class TableauProvider[T: Tableau | ExtendedTableau](Protocol):
    @abc.abstractmethod
    def tableau(self) -> T:
        raise NotImplementedError


RK1: Tableau = (
    ((0, ()),),
    (1,),
)
"Euler method"


@dataclasses.dataclass(frozen=True)
class CustomTableau[T: Tableau | ExtendedTableau](TableauProvider[T]):
    custom: T

    def tableau(self) -> T:
        return self.custom


@dataclasses.dataclass(frozen=True)
class RK2Custom(TableauProvider):
    c1: float = 1.0

    def tableau(self) -> Tableau:
        return rk2_tableau(self.c1)


@dataclasses.dataclass(frozen=True)
class RK3Custom(TableauProvider):
    c1: float = 1 / 2
    c2: float = 1.0

    def tableau(self) -> Tableau:
        return rk3_tableau(self.c1, self.c2)


@dataclasses.dataclass(frozen=True)
class RK4Custom(TableauProvider):
    c1: float = 1 / 3
    c2: float = 2 / 3

    def tableau(self) -> Tableau:
        return rk4_tableau(self.c1, self.c2)


@enum.unique
class RK2(enum.Enum):
    Heun = rk2_tableau(1)
    Mid = rk2_tableau(1 / 2)
    Ralston = rk2_tableau(2 / 3)

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK3(enum.Enum):
    Kutta = rk3_tableau(1 / 2, 1)
    Heun = rk3_tableau(1 / 3, 2 / 3)
    Ralston = rk3_tableau(1 / 2, 3 / 4)
    Wray = rk3_tableau(8 / 15, 2 / 3)
    SSPRK3 = rk3_tableau(1, 1 / 2)

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK4(enum.Enum):
    Classic = (
        (
            (0, ()),
            (1 / 2, (1 / 2,)),
            (1 / 2, (0, 1 / 2)),
            (1, (0, 0, 1)),
        ),
        (1 / 6, 1 / 3, 1 / 3, 1 / 6),
    )
    Eighth = rk4_tableau(1 / 3, 2 / 3)
    Ralston = rk4_tableau(2 / 5, (14 - 3 * math.sqrt(5)) / 16)

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK5(enum.Enum):
    Nystrom = (
        (
            (0, ()),
            (1 / 3, (1 / 3,)),
            (2 / 5, (4 / 25, 6 / 25)),
            (1, (1 / 4, -3, 15 / 4)),
            (2 / 3, (2 / 27, 10 / 9, -50 / 81, 8 / 81)),
            (4 / 5, (2 / 25, 12 / 25, 2 / 15, 8 / 75, 0)),
        ),
        (23 / 192, 0, 125 / 192, 0, -27 / 64, 125 / 192),
    )

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RKE2(enum.Enum):
    Heun = (
        (
            (0, ()),
            (1, (1,)),
        ),
        (1 / 2, 1 / 2),
        (1, 0),
    )
    # Fehlberg = enum.auto()

    def tableau(self) -> ExtendedTableau:
        return self.value


@enum.unique
class RKE5(enum.Enum):
    Fehlberg = (
        (
            (0, ()),
            (1 / 4, (1 / 4,)),
            (3 / 8, (3 / 32, 9 / 32)),
            (12 / 13, (1932 / 2197, -7200 / 2197, 7296 / 2197)),
            (1, (439 / 216, -8, 3680 / 513, -845 / 4104)),
            (1 / 2, (-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40)),
        ),
        (16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55),
        (25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0),
    )
    # CashKarp = enum.auto()
    # DormandPrince = enum.auto()

    def tableau(self) -> ExtendedTableau:
        return self.value
