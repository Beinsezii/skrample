import abc
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


class TableauProvider(Protocol):
    @abc.abstractmethod
    def tableau(self) -> Tableau:
        raise NotImplementedError


class ExtendedTableauProvider(Protocol):
    @abc.abstractmethod
    def tableau(self) -> ExtendedTableau:
        raise NotImplementedError


RK1: Tableau = (
    ((0, ()),),
    (1,),
)
"Euler method"


@enum.unique
class RK2(enum.StrEnum):
    Heun = enum.auto()
    Mid = enum.auto()
    Ralston = enum.auto()

    def tableau(self) -> Tableau:
        match self:
            case self.Heun:
                return (
                    (
                        (0, ()),
                        (1, (1,)),
                    ),
                    (1 / 2, 1 / 2),
                )
            case self.Mid:
                return (
                    (
                        (0, ()),
                        (1 / 2, (1 / 2,)),
                    ),
                    (0, 1),
                )
            case self.Ralston:
                return (
                    (
                        (0, ()),
                        (2 / 3, (2 / 3,)),
                    ),
                    (1 / 4, 3 / 4),
                )


@enum.unique
class RK3(enum.StrEnum):
    Kutta = enum.auto()
    Heun = enum.auto()
    Ralston = enum.auto()
    Wray = enum.auto()
    SSPRK3 = enum.auto()

    def tableau(self) -> Tableau:
        match self:
            case self.Kutta:
                return (
                    (
                        (0, ()),
                        (1 / 2, (1 / 2,)),
                        (1, (-1, 2)),
                    ),
                    (1 / 6, 2 / 3, 1 / 6),
                )
            case self.Heun:
                return (
                    (
                        (0, ()),
                        (1 / 3, (1 / 3,)),
                        (2 / 3, (0, 2 / 3)),
                    ),
                    (1 / 4, 0, 3 / 4),
                )
            case self.Ralston:
                return (
                    (
                        (0, ()),
                        (1 / 2, (1 / 2,)),
                        (3 / 4, (0, 3 / 4)),
                    ),
                    (2 / 9, 1 / 3, 4 / 9),
                )
            case self.Wray:
                return (
                    (
                        (0, ()),
                        (8 / 15, (8 / 15,)),
                        (2 / 3, (1 / 4, 5 / 12)),
                    ),
                    (1 / 4, 0, 3 / 4),
                )
            case self.SSPRK3:
                return (
                    (
                        (0, ()),
                        (1, (1,)),
                        (1 / 2, (1 / 4, 1 / 4)),
                    ),
                    (1 / 6, 1 / 6, 2 / 3),
                )


@enum.unique
class RK4(enum.StrEnum):
    Classic = enum.auto()
    Eighth = enum.auto()
    Ralston = enum.auto()

    def tableau(self) -> Tableau:
        match self:
            case self.Classic:
                return (
                    (
                        (0, ()),
                        (1 / 2, (1 / 2,)),
                        (1 / 2, (0, 1 / 2)),
                        (1, (0, 0, 1)),
                    ),
                    (1 / 6, 1 / 3, 1 / 3, 1 / 6),
                )
            case self.Eighth:
                return (
                    (
                        (0, ()),
                        (1 / 3, (1 / 3,)),
                        (2 / 3, (-1 / 3, 1)),
                        (1, (1, -1, 1)),
                    ),
                    (1 / 8, 3 / 8, 3 / 8, 1 / 8),
                )
            case self.Ralston:
                sq5: float = math.sqrt(5)
                return (
                    (
                        (0, ()),
                        (2 / 5, (2 / 5,)),
                        (
                            (14 - 3 * sq5) / 16,
                            (
                                (-2889 + 1428 * sq5) / 1024,
                                (3785 - 1620 * sq5) / 1024,
                            ),
                        ),
                        (
                            1,
                            (
                                (-3365 + 2094 * sq5) / 6040,
                                (-975 - 3046 * sq5) / 2552,
                                (467040 + 203968 * sq5) / 240845,
                            ),
                        ),
                    ),
                    (
                        (263 + 24 * sq5) / 1812,
                        (125 - 1000 * sq5) / 3828,
                        (3426304 + 1661952 * sq5) / 5924787,
                        (30 - 4 * sq5) / 123,
                    ),
                )


@enum.unique
class RK5(enum.StrEnum):
    Nystrom = enum.auto()

    def tableau(self) -> Tableau:
        match self:
            case self.Nystrom:
                return (
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


@enum.unique
class RKE2(enum.StrEnum):
    Heun = enum.auto()
    # Fehlberg = enum.auto()

    def tableau(self) -> ExtendedTableau:
        match self:
            case self.Heun:
                return (
                    (
                        (0, ()),
                        (1, (1,)),
                    ),
                    (1 / 2, 1 / 2),
                    (1, 0),
                )


@enum.unique
class RKE5(enum.StrEnum):
    Fehlberg = enum.auto()
    # CashKarp = enum.auto()
    # DormandPrince = enum.auto()

    def tableau(self) -> ExtendedTableau:
        match self:
            case self.Fehlberg:
                return (
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
