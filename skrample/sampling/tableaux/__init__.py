import abc
import dataclasses
import enum
import math
from typing import Protocol

from . import stepanov_10_15

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

V5 = math.sqrt(5)


def validate_tableau(tab: Tableau | ExtendedTableau, tolerance: float = 1e-12) -> None | IndexError | ValueError:
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
    "2nd order, 2 calls"

    Heun = rk2_tableau(1)
    Mid = rk2_tableau(1 / 2)
    Ralston = rk2_tableau(2 / 3)

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK3(enum.Enum):
    "3rd order, 3 calls"

    Kutta = rk3_tableau(1 / 2, 1)
    Heun = rk3_tableau(1 / 3, 2 / 3)
    Ralston = rk3_tableau(1 / 2, 3 / 4)
    Wray = rk3_tableau(8 / 15, 2 / 3)
    SSPRK3 = rk3_tableau(1, 1 / 2)

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK4(enum.Enum):
    "4th order, 4 calls"

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
    Ralston = rk4_tableau(2 / 5, (14 - 3 * V5) / 16)

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RKZ(enum.Enum):
    """Tableaux provided by this method do not have clean generic forms, and require more calls than their order.
    Since these are rare, they are all categorized into one enum"""

    Nystrom5 = (
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

    Butcher6 = (
        (
            (0, ()),
            ((5 - V5) / 10, ((5 - V5) / 10,)),
            ((5 + V5) / 10, (-V5 / 10, (5 + 2 * V5) / 10)),
            ((5 - V5) / 10, ((-15 + 7 * V5) / 20, (-1 + V5) / 4, (15 - 7 * V5) / 10)),
            ((5 + V5) / 10, ((5 - V5) / 60, 0, 1 / 6, (15 + 7 * V5) / 60)),
            ((5 - V5) / 10, ((5 + V5) / 60, 0, (9 - 5 * V5) / 12, 1 / 6, (-5 + 3 * V5) / 10)),
            (1.0, (1 / 6, 0, (-55 + 25 * V5) / 12, (-25 - 7 * V5) / 12, 5 - 2 * V5, (5 + V5) / 2)),
        ),
        (1 / 12, 0, 0, 0, 5 / 12, 5 / 12, 1 / 12),
    )
    """On Runge-Kutta processes of high order, J. C. Butcher
    https://www.cambridge.org/core/services/aop-cambridge-core/content/view/40DFE501CAB781C9AAE1439B6B8F481A/S1446788700023387a.pdf/on-runge-kutta-processes-of-high-order.pdf
    Figure [15]"""

    Stepanov10 = stepanov_10_15.TABLEAU
    """On Runge-Kutta methods of order 10, Misha Stepanov
    https://arxiv.org/pdf/2504.17329"""

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
    Fehlberg = (
        (
            (0, ()),
            (1 / 2, (1 / 2,)),
            (1, (1 / 256, 255 / 256)),
        ),
        (1 / 512, 255 / 256, 1 / 512),
        (1 / 256, 255 / 256, 0),
    )

    def tableau(self) -> ExtendedTableau:
        return self.value


@enum.unique
class RKE3(enum.Enum):
    BogackiShampine = (
        (
            (0, ()),
            (1 / 2, (1 / 2,)),
            (3 / 4, (0, 3 / 4)),
            (1, (2 / 9, 1 / 3, 4 / 9)),
        ),
        (2 / 9, 1 / 3, 4 / 9, 0),
        (7 / 24, 1 / 4, 1 / 3, 1 / 8),
    )

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
    CashKarp = (
        (
            (0, ()),
            (1 / 5, (1 / 5,)),
            (3 / 10, (3 / 40, 9 / 40)),
            (3 / 5, (3 / 10, -9 / 10, 6 / 5)),
            (1, (-11 / 54, 5 / 2, -70 / 27, 35 / 27)),
            (7 / 8, (1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096)),
        ),
        (37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771),
        (2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4),
    )
    DormandPrince = (
        (
            (0, ()),
            (1 / 5, (1 / 5,)),
            (3 / 10, (3 / 40, 9 / 40)),
            (4 / 5, (44 / 45, -56 / 15, 32 / 9)),
            (8 / 9, (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729)),
            (1, (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656)),
            (1, (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)),
        ),
        (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0),
        (5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40),
    )

    def tableau(self) -> ExtendedTableau:
        return self.value


@enum.unique
class Shanks1965(enum.Enum):
    """Higher order approximations of runge-kutta type, E. B. Shanks
    https://ntrs.nasa.gov/citations/19650022581
    Note RK5_5, RK6_6, RK7_7 and RK8_10 are only approximations of their respective orders.
    """

    RK4_4 = rk4_tableau(1 / 100, 3 / 5)
    RK5_5 = (
        (
            (0, ()),
            (1 / 9000, (1 / 9000,)),
            (3 / 10, tuple(x / 10 for x in (-4047, 4050))),
            (3 / 4, tuple(x / 8 for x in (20241, -20250, 15))),
            (1, tuple(x / 81 for x in (-931041, 931500, -490, 112))),
        ),
        tuple(x / 1134 for x in (105, 0, 500, 448, 81)),
    )
    "Not a true 5th order"
    RK6_6 = (
        (
            (0, ()),
            (1 / 300, (1 / 300,)),
            (1 / 5, tuple(x / 5 for x in (-29, 30))),
            (3 / 5, tuple(x / 5 for x in (323, -330, 10))),
            (14 / 15, tuple(x / 810 for x in (-510104, 521640, -12705, 1925))),
            (1, tuple(x / 77 for x in (-417923, 427350, -10605, 1309, -54))),
        ),
        tuple(x / 3696 for x in (198, 0, 1225, 1540, 810, -77)),
    )
    "Not a true 6th order"
    RK7_7 = (
        (
            (0, ()),
            (1 / 192, (1 / 192,)),
            (1 / 6, tuple(x / 6 for x in (-15, 16))),
            (1 / 2, tuple(x / 186 for x in (4867, -5072, 298))),
            (1, tuple(x / 31 for x in (-19995, 20896, -1025, 155))),
            (5 / 6, tuple(x / 5022 for x in (-469805, 490960, -22736, 5580, 186))),
            (1, tuple(x / 2604 for x in (914314, -955136, 47983, -6510, -558, 2511))),
        ),
        tuple(x / 300 for x in (14, 0, 81, 110, 0, 81, 14)),
    )
    "Not a true 7th order"
    RK7_9 = (
        (
            (0, ()),
            (2 / 9, (2 / 9,)),
            (1 / 3, tuple(x / 12 for x in (1, 3))),
            (1 / 2, tuple(x / 8 for x in (1, 0, 3))),
            (1 / 6, tuple(x / 216 for x in (23, 0, 21, -8))),
            (8 / 9, tuple(x / 729 for x in (-4136, 0, -13584, 5264, 13104))),
            (1 / 9, tuple(x / 151632 for x in (105131, 0, 302016, -107744, -284256, 1701))),
            (5 / 6, tuple(x / 1375920 for x in (-775229, 0, -2770950, 1735136, 2547216, 81891, 328536))),
            (1, tuple(x / 251888 for x in (23569, 0, -122304, -20384, 695520, -99873, -466560, 241920))),
        ),
        tuple(x / 2140320 for x in (110201, 0, 0, 767936, 635040, -59049, -59049, 635040, 110201)),
    )
    RK8_10 = (
        (
            (0, ()),
            (4 / 27, (4 / 27,)),
            (2 / 9, tuple(x / 18 for x in (1, 3))),
            (1 / 3, tuple(x / 12 for x in (1, 0, 3))),
            (1 / 2, tuple(x / 8 for x in (1, 0, 0, 3))),
            (2 / 3, tuple(x / 54 for x in (13, 0, -27, 42, 8))),
            (1 / 6, tuple(x / 4320 for x in (389, 0, -54, 966, -824, 243))),
            (1, tuple(x / 20 for x in (-231, 0, 81, -1164, 656, -122, 800))),
            (5 / 6, tuple(x / 288 for x in (-127, 0, 18, -678, 456, -9, 576, 4))),
            (1, tuple(x / 820 for x in (1481, 0, -81, 7104, -3376, 72, -5040, -60, 720))),
        ),
        tuple(x / 840 for x in (41, 0, 0, 27, 272, 27, 216, 0, 216, 41)),
    )
    "Not a true 8th order"
    RK8_12 = (
        (
            (0, ()),
            (1 / 9, (1 / 9,)),
            (1 / 6, tuple(x / 24 for x in (1, 3))),
            (1 / 4, tuple(x / 16 for x in (1, 0, 3))),
            (1 / 10, tuple(x / 500 for x in (29, 0, 33, -12))),
            (1 / 6, tuple(x / 972 for x in (33, 0, 0, 4, 125))),
            (1 / 2, tuple(x / 36 for x in (-21, 0, 0, 76, 125, -162))),
            (2 / 3, tuple(x / 243 for x in (-30, 0, 0, -32, 125, 0, 99))),
            (1 / 3, tuple(x / 324 for x in (1175, 0, 0, -3456, -6250, 8424, 242, -27))),
            (5 / 6, tuple(x / 324 for x in (293, 0, 0, -852, -1375, 1836, -118, 162, 324))),
            (5 / 6, tuple(x / 1620 for x in (1303, 0, 0, -4260, -6875, 9990, 1030, 0, 0, 162))),
            (1, tuple(x / 4428 for x in (-8595, 0, 0, 30720, 48750, -66096, 378, -729, -1944, -1296, 3240))),
        ),
        tuple(x / 840 for x in (41, 0, 0, 0, 0, 216, 272, 27, 27, 36, 180, 41)),
    )

    def tableau(self) -> Tableau:
        return self.value
