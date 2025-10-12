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


def rk2_tableau(alpha: float) -> Tableau:
    "Create a generic 2nd order Tableau from a given alpha value."
    alpha_w = 1 / (2 * alpha)
    return (
        (
            (0.0, ()),
            (alpha, (alpha,)),
        ),
        (1 - alpha_w, alpha_w),
    )


def rk3_tableau(alpha: float, beta: float) -> Tableau:
    "Create a generic 3rd order Tableau from a given alpha and beta values."

    return (
        (
            (0.0, ()),
            (alpha, (alpha,)),
            (
                beta,
                (
                    beta / alpha * ((beta - 3 * alpha * (1 - alpha)) / (3 * alpha - 2)),
                    -beta / alpha * ((beta - alpha) / (3 * alpha - 2)),
                ),
            ),
        ),
        (
            1 - (3 * alpha + 3 * beta - 2) / (6 * alpha * beta),
            (3 * beta - 2) / (6 * alpha * (beta - alpha)),
            (2 - 3 * alpha) / (6 * beta * (beta - alpha)),
        ),
    )


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
    alpha: float = 1.0

    def tableau(self) -> Tableau:
        return rk2_tableau(self.alpha)


@dataclasses.dataclass(frozen=True)
class RK3Custom(TableauProvider):
    alpha: float = 1 / 2
    beta: float = 1.0

    def tableau(self) -> Tableau:
        return rk3_tableau(self.alpha, self.beta)


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
    Eighth = (
        (
            (0, ()),
            (1 / 3, (1 / 3,)),
            (2 / 3, (-1 / 3, 1)),
            (1, (1, -1, 1)),
        ),
        (1 / 8, 3 / 8, 3 / 8, 1 / 8),
    )
    Ralston = (
        (
            (0, ()),
            (2 / 5, (2 / 5,)),
            ((14 - 3 * math.sqrt(5)) / 16, ((-2889 + 1428 * math.sqrt(5)) / 1024, (3785 - 1620 * math.sqrt(5)) / 1024)),
            (
                1,
                (
                    (-3365 + 2094 * math.sqrt(5)) / 6040,
                    (-975 - 3046 * math.sqrt(5)) / 2552,
                    (467040 + 203968 * math.sqrt(5)) / 240845,
                ),
            ),
        ),
        (
            (263 + 24 * math.sqrt(5)) / 1812,
            (125 - 1000 * math.sqrt(5)) / 3828,
            (3426304 + 1661952 * math.sqrt(5)) / 5924787,
            (30 - 4 * math.sqrt(5)) / 123,
        ),
    )

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
