import abc
import dataclasses
import enum
import math
from typing import Protocol

from . import feagin_10_17, feagin_12_25, feagin_14_35, harrier_10_17, ono_10_17, stepanov_10_15, zhang_10_16
from .common import ButcherCoeffs, EmbeddedTableau, Stage, Tableau, TableauType, pretty_tableau

V2 = math.sqrt(2)
V5 = math.sqrt(5)
V21 = math.sqrt(21)


def rk2_tableau(c1: float) -> Tableau:
    "Create a generic 2nd order Tableau from a given coefficient."
    return Tableau(
        (
            Stage(0.0, ()),
            Stage(c1, (c1,)),
        ),
        (1 - 1 / (2 * c1), 1 / (2 * c1)),
    )


def rk3_tableau(c1: float, c2: float) -> Tableau:
    "Create a generic 3rd order Tableau from given coefficients."
    return Tableau(
        (
            Stage(0.0, ()),
            Stage(c1, (c1,)),
            Stage(c2, (c2 / c1 * ((c2 - 3 * c1 * (1 - c1)) / (3 * c1 - 2)), -c2 / c1 * ((c2 - c1) / (3 * c1 - 2)))),
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

    return Tableau(
        (
            Stage(0.0, ()),
            Stage(c1, (c1,)),  # a21 = c1
            Stage(c2, (a31, a32)),
            Stage(1.0, (a41, a42, a43)),
        ),
        (b1, b2, b3, b4),
    )


def ees25_tableau(x: float) -> Tableau:
    """Create a 2nd order 3-stage EES Tableau.
    Explicit and Effectively Symmetric Runge-Kutta Methods (2025)
    https://arxiv.org/abs/2507.21006"""
    return Tableau(
        (
            Stage(0.0, ()),
            Stage((1 + 2 * x) / (4 * (1 - x)), ((1 + 2 * x) / (4 * (1 - x)),)),
            Stage(3 / (4 * (1 - x)), ((4 * x - 1) ** 2 / (4 * (x - 1) * (1 - 4 * x**2)), (1 - x) / (1 - 4 * x**2))),
        ),
        (x, 1 / 2, 1 / 2 - x),
    )


def ees27_tableau(x: float) -> Tableau:
    """Create a 2nd order 4-stage EES Tableau.
    Explicit and Effectively Symmetric Runge-Kutta Methods (2025)
    https://arxiv.org/abs/2507.21006"""
    V2 = math.sqrt(2)
    A = (2 * x + V2) / ((2 * x - 1) * (-2 * x - V2 + 1))
    B = 1 / ((2 * x - 1) * (1 - V2 - 2 * x) * (2 - V2 - 2 * x))

    a2 = ((-2 + V2 * (1 - 2 * x)) / (4 * (x - 1)),)
    a3 = ((((2 * x + V2 - 2) * (4 * x + V2 - 2)) / (4 * V2 * (x - 1))) * A, (0.5 * (-1 + V2)) * A)
    a4 = (
        ((2 * x - V2) * (-40 * x**4 + (80 - 40 * V2) * x**3 - (88 - 60 * V2) * x**2 + (48 - 34 * V2) * x + 7 * V2 - 10))
        / (4 * (x - 1) * (2 * x**2 - 1))
        * B,
        # WARN: This is the algo in the paper, but their tableau on (8.6)
        # shows the A42 factor as exactly double what the algorithm would suggest.
        # I'm not sure which one is actually correct.
        (2 - V2) * x * (x - 1) * (4 * x + V2 - 2) * B,  # INFO: to match the (8.6) tableau
        # 1 / 2 * (2 - V2) * x * (x - 1) * (4 * x + V2 - 2) * B,  # INFO: to match the algorithm
        ((2 - V2) * (2 * x - V2) * (2 + V2 - 2 * x) * (x - 1) * (2 * x - 1))
        / (4 * (2 * x**2 - 1) * (2 * x**2 - 4 * x + 1)),
    )
    return Tableau(
        (
            Stage(0.0, ()),
            Stage(math.fsum(a2), a2),
            Stage(math.fsum(a3), a3),
            Stage(math.fsum(a4), a4),
        ),
        (x, 1 / 2 * (2 - V2) - (1 - V2) * x, (1 - V2) * (x - 1), 1 / 2 * (2 - V2) - x),
    )


class TableauProvider[T: TableauType](Protocol):
    @abc.abstractmethod
    def tableau(self) -> T:
        raise NotImplementedError

    def pretty(self) -> str:
        return pretty_tableau(self.tableau())


@dataclasses.dataclass(frozen=True)
class CustomTableau[T: TableauType](TableauProvider[T]):
    custom: T

    def tableau(self) -> T:
        return self.custom


@dataclasses.dataclass(frozen=True)
class RK2Custom(TableauProvider[Tableau]):
    c1: float = 1.0

    def tableau(self) -> Tableau:
        return rk2_tableau(self.c1)


@dataclasses.dataclass(frozen=True)
class RK3Custom(TableauProvider[Tableau]):
    c1: float = 1 / 2
    c2: float = 1.0

    def tableau(self) -> Tableau:
        return rk3_tableau(self.c1, self.c2)


@dataclasses.dataclass(frozen=True)
class RK4Custom(TableauProvider[Tableau]):
    c1: float = 1 / 3
    c2: float = 2 / 3

    def tableau(self) -> Tableau:
        return rk4_tableau(self.c1, self.c2)


@enum.unique
class RK1(enum.Enum):
    Euler = Tableau(
        (Stage(0, ()),),
        (1,),
    )

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK2(enum.Enum):
    Mid = rk2_tableau(1 / 2)
    Ralston = rk2_tableau(2 / 3)
    Golden = rk2_tableau((1 + V5) / 4)
    "B row is (1-1/φ, 1/φ)"

    EES5_SYM = ees25_tableau(1 / 4)
    """Explicit and Effectively Symmetric Runge-Kutta Methods (2025)
    https://arxiv.org/abs/2507.21006
    EES(2, 5; 1/4), Figure (8.3)"""
    EES5_MIN = ees25_tableau(1 / 10)
    """Explicit and Effectively Symmetric Runge-Kutta Methods (2025)
    https://arxiv.org/abs/2507.21006
    EES(2, 5; 1/10), Figure (8.4)"""

    EES7_SYM = ees27_tableau(1 / 4 * (2 - V2))
    """Explicit and Effectively Symmetric Runge-Kutta Methods (2025)
    https://arxiv.org/abs/2507.21006
    EES(2, 7; 1/4(2-√2)), Figure (8.5)"""
    EES7_MIN = ees27_tableau(1 / 14 * (5 - 3 * V2))
    """Explicit and Effectively Symmetric Runge-Kutta Methods (2025)
    https://arxiv.org/abs/2507.21006
    EES(2, 7; 1/14(5 - 3√2)), Figure (8.6)"""

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK3(enum.Enum):
    Kutta = rk3_tableau(1 / 2, 1)
    Heun = rk3_tableau(1 / 3, 2 / 3)
    Ralston = rk3_tableau(1 / 2, 3 / 4)
    """Runge-Kutta Methods With Minimum Error Bounds, Anthony Ralston (1962)
    https://www.ams.org/journals/mcom/1962-16-080/S0025-5718-1962-0150954-0/S0025-5718-1962-0150954-0.pdf"""
    Wray = rk3_tableau(8 / 15, 2 / 3)

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RK4(enum.Enum):
    Kutta = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 2, (1 / 2,)),
            Stage(1 / 2, (0, 1 / 2)),
            Stage(1, (0, 0, 1)),
        ),
        (1 / 6, 1 / 3, 1 / 3, 1 / 6),
    )
    Eighth = rk4_tableau(1 / 3, 2 / 3)
    Ralston = rk4_tableau(2 / 5, (14 - 3 * V5) / 16)
    """Runge-Kutta Methods With Minimum Error Bounds, Anthony Ralston (1962)
    https://www.ams.org/journals/mcom/1962-16-080/S0025-5718-1962-0150954-0/S0025-5718-1962-0150954-0.pdf"""

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RKZ(enum.Enum):
    """Tableaux provided by this method do not have clean generic forms, and require more calls than their order.
    Since these are rare, they are all categorized into one enum"""

    Nystrom5 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 3, (1 / 3,)),
            Stage(2 / 5, (4 / 25, 6 / 25)),
            Stage(1, (1 / 4, -3, 15 / 4)),
            Stage(2 / 3, (2 / 27, 10 / 9, -50 / 81, 8 / 81)),
            Stage(4 / 5, (2 / 25, 12 / 25, 2 / 15, 8 / 75, 0)),
        ),
        (23 / 192, 0, 125 / 192, 0, -27 / 64, 125 / 192),
    )

    Butcher6 = Tableau(
        (
            Stage(0, ()),
            Stage((5 - V5) / 10, ((5 - V5) / 10,)),
            Stage((5 + V5) / 10, (-V5 / 10, (5 + 2 * V5) / 10)),
            Stage((5 - V5) / 10, ((-15 + 7 * V5) / 20, (-1 + V5) / 4, (15 - 7 * V5) / 10)),
            Stage((5 + V5) / 10, ((5 - V5) / 60, 0, 1 / 6, (15 + 7 * V5) / 60)),
            Stage((5 - V5) / 10, ((5 + V5) / 60, 0, (9 - 5 * V5) / 12, 1 / 6, (-5 + 3 * V5) / 10)),
            Stage(1.0, (1 / 6, 0, (-55 + 25 * V5) / 12, (-25 - 7 * V5) / 12, 5 - 2 * V5, (5 + V5) / 2)),
        ),
        (1 / 12, 0, 0, 0, 5 / 12, 5 / 12, 1 / 12),
    )
    """On Runge-Kutta processes of high order, J. C. Butcher
    https://www.cambridge.org/core/services/aop-cambridge-core/content/view/40DFE501CAB781C9AAE1439B6B8F481A/S1446788700023387a.pdf/on-runge-kutta-processes-of-high-order.pdf
    Figure [15]"""

    CV8 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 2, (1 / 2,)),
            Stage(1 / 2, (1 / 4, 1 / 4)),
            Stage(1 / 2 + 1 / 14 * V21, (1 / 7, -1 / 14 - 3 / 98 * V21, 3 / 7 + 5 / 49 * V21)),
            Stage(1 / 2 + 1 / 14 * V21, (11 / 84 + 1 / 84 * V21, 0, 2 / 7 + 4 / 63 * V21, 1 / 12 - 1 / 252 * V21)),
            Stage(
                1 / 2,
                (5 / 48 + 1 / 48 * V21, 0, 1 / 4 + 1 / 36 * V21, -77 / 120 + 7 / 180 * V21, 63 / 80 - 7 / 80 * V21),
            ),
            Stage(
                1 / 2 - 1 / 14 * V21,
                (
                    5 / 21 - 1 / 42 * V21,
                    0,
                    -48 / 35 + 92 / 315 * V21,
                    211 / 30 - 29 / 18 * V21,
                    -36 / 5 + 23 / 14 * V21,
                    9 / 5 - 13 / 35 * V21,
                ),
            ),
            Stage(1 / 2 - 1 / 14 * V21, (1 / 14, 0, 0, 0, 1 / 9 - 1 / 42 * V21, 13 / 63 - 1 / 21 * V21, 1 / 9)),
            Stage(
                1 / 2,
                (
                    1 / 32,
                    0,
                    0,
                    0,
                    91 / 576 - 7 / 192 * V21,
                    11 / 72,
                    -385 / 1152 - 25 / 384 * V21,
                    63 / 128 + 13 / 128 * V21,
                ),
            ),
            Stage(
                1 / 2 + 1 / 14 * V21,
                (
                    1 / 14,
                    0,
                    0,
                    0,
                    1 / 9,
                    -733 / 2205 - 1 / 15 * V21,
                    515 / 504 + 37 / 168 * V21,
                    -51 / 56 - 11 / 56 * V21,
                    132 / 245 + 4 / 35 * V21,
                ),
            ),
            Stage(
                1,
                (
                    0,
                    0,
                    0,
                    0,
                    -7 / 3 + 7 / 18 * V21,
                    -2 / 5 + 28 / 45 * V21,
                    -91 / 24 - 53 / 72 * V21,
                    301 / 72 + 53 / 72 * V21,
                    28 / 45 - 28 / 45 * V21,
                    49 / 18 - 7 / 18 * V21,
                ),
            ),
        ),
        (1 / 20, 0, 0, 0, 0, 0, 0, 49 / 180, 16 / 45, 49 / 180, 1 / 20),
    )
    "Some Explicit Runge-Kutta Methods of High Order, G. J. Cooper & J. H. Verner (1972)"

    Stepanov10 = stepanov_10_15.TABLEAU
    """On Runge-Kutta methods of order 10, Misha Stepanov (2025)
    https://arxiv.org/pdf/2504.17329"""

    Ono10 = ono_10_17.TABLEAU
    """Hiroshi Ono's 17 stage order 10 Runge-Kutta scheme (2003)
    http://www.peterstone.name/Maplepgs/Maple/nmthds/RKcoeff/Runge_Kutta_schemes/RK10/RKcoeff10f_1.pdf"""

    Harrier10 = harrier_10_17.TABLEAU

    Zhang10 = zhang_10_16.TABLEAU
    """Discovering New Runge-Kutta Methods Using Unstructured Numerical Search, David Zhang (2019)
    https://arxiv.org/pdf/1911.00318"""

    Feagin10 = feagin_10_17.TABLEAU

    Feagin12 = feagin_12_25.TABLEAU
    """An Explicit Runge-Kutta Method Of Order Twelve, Terry Feagin (2007)"""

    Feagin14 = feagin_14_35.TABLEAU

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class RKE2(enum.Enum):
    Heun = EmbeddedTableau(
        (
            Stage(0, ()),
            Stage(1, (1,)),
        ),
        (1 / 2, 1 / 2),
        (1, 0),
    )
    Fehlberg = EmbeddedTableau(
        (
            Stage(0, ()),
            Stage(1 / 2, (1 / 2,)),
            Stage(1, (1 / 256, 255 / 256)),
        ),
        (1 / 512, 255 / 256, 1 / 512),
        (1 / 256, 255 / 256, 0),
    )

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> EmbeddedTableau:
        return self.value


@enum.unique
class RKE3(enum.Enum):
    BogackiShampine = EmbeddedTableau(
        (
            Stage(0, ()),
            Stage(1 / 2, (1 / 2,)),
            Stage(3 / 4, (0, 3 / 4)),
            Stage(1, (2 / 9, 1 / 3, 4 / 9)),
        ),
        (2 / 9, 1 / 3, 4 / 9, 0),
        (7 / 24, 1 / 4, 1 / 3, 1 / 8),
    )
    SSPRK3_4 = EmbeddedTableau(
        (
            Stage(0, ()),
            Stage(1 / 2, (1 / 2,)),
            Stage(1, (1 / 2, 1 / 2)),
            Stage(1 / 2, (1 / 6, 1 / 6, 1 / 6)),
        ),
        (1 / 6, 1 / 6, 1 / 6, 1 / 2),
        (1 / 4, 1 / 4, 1 / 4, 1 / 4),
    )
    "https://arxiv.org/pdf/2104.06836"

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> EmbeddedTableau:
        return self.value


@enum.unique
class RKE5(enum.Enum):
    Fehlberg = EmbeddedTableau(
        (
            Stage(0, ()),
            Stage(1 / 4, (1 / 4,)),
            Stage(3 / 8, (3 / 32, 9 / 32)),
            Stage(12 / 13, (1932 / 2197, -7200 / 2197, 7296 / 2197)),
            Stage(1, (439 / 216, -8, 3680 / 513, -845 / 4104)),
            Stage(1 / 2, (-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40)),
        ),
        (16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55),
        (25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0),
    )
    CashKarp = EmbeddedTableau(
        (
            Stage(0, ()),
            Stage(1 / 5, (1 / 5,)),
            Stage(3 / 10, (3 / 40, 9 / 40)),
            Stage(3 / 5, (3 / 10, -9 / 10, 6 / 5)),
            Stage(1, (-11 / 54, 5 / 2, -70 / 27, 35 / 27)),
            Stage(7 / 8, (1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096)),
        ),
        (37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771),
        (2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4),
    )
    DormandPrince = EmbeddedTableau(
        (
            Stage(0, ()),
            Stage(1 / 5, (1 / 5,)),
            Stage(3 / 10, (3 / 40, 9 / 40)),
            Stage(4 / 5, (44 / 45, -56 / 15, 32 / 9)),
            Stage(8 / 9, (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729)),
            Stage(1, (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656)),
            Stage(1, (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)),
        ),
        (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0),
        (5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40),
    )

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> EmbeddedTableau:
        return self.value


@enum.unique
class SSP(enum.Enum):
    """Global Optimization Of Explicit Strong-Stability-Preserving Runge-Kutta Methods, Steven J. Ruuth (2006)
    https://www.ams.org/journals/mcom/2006-75-253/S0025-5718-05-01772-2/S0025-5718-05-01772-2.pdf"""

    RK3_3 = rk3_tableau(1, 1 / 2)

    RK3_5 = ButcherCoeffs.from_shu_osher(
        [
            [1],
            [0, 1],
            [0.355909775063327, 0, 0.644090224936674],
            [0.367933791638137, 0, 0, 0.632066208361863],
            [0, 0, 0.237593836598569, 0, 0.762406163401431],
        ],
        [
            [0.377268915331368],
            [0, 0.377268915331368],
            [0, 0, 0.242995220537396],
            [0, 0, 0, 0.238458932846290],
            [0, 0, 0, 0, 0.287632146308408],
        ],
    ).compose()
    RK3_6 = ButcherCoeffs.from_shu_osher(
        [
            [1],
            [0, 1],
            [0, 0, 1],
            [0.476769811285196, 0.098511733286064, 0, 0.424718455428740],
            [0, 0, 0, 0, 1],
            [0, 0, 0.155221702560091, 0, 0, 0.844778297439909],
        ],
        [
            [0.284220721334261],
            [0, 0.284220721334261],
            [0, 0, 0.284220721334261],
            [0, 0, 0, 0.120713785765930],
            [0, 0, 0, 0, 0.284220721334261],
            [0, 0, 0, 0, 0, 0.240103497065900],
        ],
    ).compose()
    RK3_7 = ButcherCoeffs.from_shu_osher(
        [
            [1],
            [0, 1],
            [0, 0, 1],
            [0.184962588071072, 0, 0, 0.815037411928928],
            [0.180718656570380, 0.314831034403793, 0, 0, 0.504450309025826],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0.120199000000000, 0, 0, 0.879801000000000],
        ],
        [
            [0.233213863663009],
            [0, 0.233213863663009],
            [0, 0, 0.233213863663009],
            [0, 0, 0, 0.190078023865845],
            [0, 0, 0, 0, 0.117644805593912],
            [0, 0, 0, 0, 0, 0.233213863663009],
            [0, 0, 0, 0, 0, 0, 0.205181790464579],
        ],
    ).compose()
    RK3_8 = ButcherCoeffs.from_shu_osher(
        [
            [1],
            [0, 1],
            [0, 0, 1],
            [0, 0, 0, 1],
            [0.421366967085359, 0.005949401107575, 0, 0, 0.572683631807067],
            [0, 0.004254010666365, 0, 0, 0, 0.995745989333635],
            [0, 0, 0.104380143093325, 0.243265240906726, 0, 0, 0.652354615999950],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        [
            [0.195804015330143],
            [0, 0.195804015330143],
            [0, 0, 0.195804015330143],
            [0, 0, 0, 0.195804015330143],
            [0, 0, 0, 0, 0.112133754621673],
            [0, 0, 0, 0, 0, 0.194971062960412],
            [0, 0, 0, 0, 0, 0, 0.127733653231944],
            [0, 0, 0, 0, 0, 0, 0, 0.195804015330143],
        ],
    ).compose()

    RK4_5 = ButcherCoeffs.from_shu_osher(
        [
            [1],
            [0.444370493651235, 0.555629506348765],
            [0.620101851488403, 0, 0.379898148511597],
            [0.178079954393132, 0, 0, 0.821920045606868],
            [0, 0, 0.517231671970585, 0.096059710526147, 0.386708617503269],
        ],
        [
            [0.391752226571890],
            [0, 0.368410593050371],
            [0, 0, 0.251891774271694],
            [0, 0, 0, 0.544974750228521],
            [0, 0, 0, 0.063692468666290, 0.226007483236906],
        ],
    ).compose()

    RK5_10 = ButcherCoeffs.from_shu_osher(
        [
            [1],
            [0.258168167463650, 0.741831832536350],
            [0, 0.037493531856076, 0.962506468143924],
            [0.595955269449077, 0, 0.404044730550923, 0],
            [0.331848124368345, 0, 0, 0.008466192609453, 0.659685683022202],
            [0.086976414344414, 0, 0, 0, 0, 0.913023585655586],
            [0.075863700003186, 0, 0.267513039663395, 0, 0, 0, 0.656623260333419],
            [0.005212058095597, 0, 0, 0.407430107306541, 0, 0, 0, 0.587357834597862],
            [0.122832051947995, 0, 0, 0, 0, 0, 0, 0, 0.877167948052005],
            [
                0.075346276482673,
                0.000425904246091,
                0,
                0,
                0,
                0.064038648145995,
                0.354077936287492,
                0,
                0,
                0.506111234837749,
            ],
        ],
        [
            [0.173586107937995],
            [0, 0.218485490268790],
            [0, 0.011042654588541, 0.283478934653295],
            [0, 0, 0.118999896166647, 0],
            [0.025030881091201, 0, 0, -0.002493476502164, 0.194291675763785],
            [0, 0, 0, 0, 0, 0.268905157462563],
            [0, 0, 0.066115378914543, 0, 0, 0, 0.193389726166555],
            [0, 0, 0, -0.119996962708895, 0, 0, 0, 0.172989562899406],
            [0.000000000000035, 0, 0, 0, 0, 0, 0, 0, 0.258344898092277],
            [0.016982542367506, 0, 0, 0, 0, 0.018860764424857, 0.098896719553054, 0, 0, 0.149060685217562],
        ],
    ).compose()

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class WSO(enum.Enum):
    """Methods with a higher weak stage order (WSO).
    Methods are annotated STAGES_ORDER_WSO, so RK_4_3_2 has 4 stages, 3rd order, and 2nd weak order.
    Explicit Runge-Kutta Methods That Alleviate Order Reduction, Biswas et al. (2023)
    https://arxiv.org/abs/2310.02817"""

    RK_3_2_2 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 2, (1 / 2,)),
            Stage(1, (1, 0)),
        ),
        (-1 / 2, 2, -1 / 2),
    )
    RK_4_3_2 = Tableau(
        (
            Stage(0, ()),
            Stage(3 / 10, (3 / 10,)),
            Stage(2 / 3, (2 / 3, 0)),
            Stage(3 / 4, (-21 / 320, 45 / 44, -729 / 3520)),
        ),
        (7 / 108, 500 / 891, -27 / 44, 80 / 81),
    )
    RK_5_3_3 = Tableau(
        (
            Stage(0, ()),
            Stage(3 / 11, (3 / 11,)),
            Stage(15 / 19, (285645 / 493487, 103950 / 493487)),
            Stage(5 / 6, (3075805 / 5314896, 1353275 / 5314896, 0)),
            Stage(1, (196687 / 177710, -129383023 / 426077496, 48013 / 42120, -2268 / 2405)),
        ),
        (5626 / 4725, -25289 / 13608, 569297 / 340200, 324 / 175, -13 / 7),
    )
    RK_6_4_3 = Tableau(
        (
            Stage(0, ()),
            Stage(1, (1,)),
            Stage(1 / 7, (461 / 3920, 99 / 3920)),
            Stage(8 / 11, (314 / 605, 126 / 605, 0)),
            Stage(5 / 9, (13193 / 197316, 39332 / 443961, 86632 / 190269, -294151 / 5327532)),
            Stage(
                4 / 5,
                (884721 / 773750, 52291 / 696375, -155381744 / 135793125, -53297233 / 355151250, 74881422 / 85499375),
            ),
        ),
        (113 / 2880, 7 / 1296, 91238 / 363285, -1478741 / 1321920, 147987 / 194480, 77375 / 72864),
    )
    RK_7_4_4 = Tableau(
        (
            Stage(0, ()),
            Stage(13 / 15, (13 / 15,)),
            Stage(
                193 / 360,
                (
                    354503406167294455217584527356969321310499849 / 679624939387359702842360408541392160411699600,
                    29553225679453489752042741666497760730650643 / 2038874818162079108527081225624176481235098800,
                ),
            ),
            Stage(719 / 720, (599677 / 612720, 1 / 185, 1 / 69)),
            Stage(
                11 / 80,
                (
                    11942118300581357822967470312387413892866711 / 90616658584981293712314721138852288054893280,
                    79816622789357424004900970571545142906303 / 18123331716996258742462944227770457610978656,
                    10939005 / 8358742409,
                    0,
                ),
            ),
            Stage(
                1 / 36,
                (
                    -2057331211140587771882165942948945576060485224020471
                    / 5094460906663329618583273674295283629198217174096496,
                    37580055896186727391837634951840677945750522481251
                    / 448734898514386546714588872865387677183262652640624,
                    -235459427251516205060 / 1472801902839731775141,
                    -787608360 / 15627214069,
                    24 / 43,
                ),
            ),
            Stage(
                193 / 240,
                (
                    793706393429237444430333112845341360638504851726921024780703
                    / 806700576848993242482064062984309812448909584075544854292960,
                    -33849235109708152171969081938954415033838967121633968102863
                    / 23685509164823635789628823956361427999363493832960729746080,
                    1821188984566562706805723220601 / 956185881514873346828934914081,
                    615685898929080 / 887641386333269,
                    -88 / 41,
                    63 / 79,
                ),
            ),
        ),
        (
            -27983058641859756462867613 / 8486495976646364788361250,
            266859550993073190375211 / 43133823812456533406250,
            -3642903731392259905073408 / 613543193666469780107625,
            -59466320887669359732170224 / 16752980798131655841946875,
            22530099787083474288594398 / 3662271198716324657203125,
            13086932957294488 / 71277904341826875,
            12256178974 / 9710853075,
        ),
    )
    RK_8_5_4 = Tableau(
        (
            Stage(0, ()),
            Stage(2 / 31, (2 / 31,)),
            Stage(8 / 39, (8 / 39, 0)),
            Stage(15 / 38, (15 / 38, 0, 0)),
            Stage(23 / 38, (23 / 38, 0, 0, 0)),
            Stage(
                31 / 39,
                (
                    -281846119171 / 64200240000,
                    289705767137 / 45358567000,
                    -779567154093 / 524247088000,
                    199824989 / 614863125,
                    -1 / 25,
                ),
            ),
            Stage(
                29 / 31,
                (
                    -5647052528401825871 / 514607937760800000,
                    80442150849469599005477 / 4661884215626994720000,
                    -271390788610093 / 44561002480000,
                    16919854802127127 / 33068912912100000,
                    918241790299 / 2569461804000,
                    -1 / 8,
                ),
            ),
            Stage(
                1,
                (
                    -69373518431251442108053395141546348749 / 4382652560085449761027489727918400000,
                    28436161533578442493717377903973791583 / 1122666693846436666675352841982200000,
                    -5846309065854115413909270194602947869 / 606644216141135157002900448063680000,
                    6129203519106929754603252009272053 / 11862175903109203056563899370081250,
                    242980026698914693640761833099573847 / 314274501092549835332737438438856250,
                    -38588365882306831 / 818781973666952750,
                    -508578133539464 / 4816364550982075,
                ),
            ),
        ),
        (
            -13932812614910970806212030308137 / 1494246680966212236480728656800,
            442315248050515865700725458450027 / 23731641831739396945145366137800,
            -21619621692735791984774655801338457 / 1572963107476970769686133552792800,
            4931046639398139760440943293895907 / 887688100270302681290608525794300,
            -808732636620048337464280245511529 / 1567883987541272156723519232078580,
            52162695 / 22722574,
            -42525800 / 8688043,
            190120171223750 / 63572266692433,
        ),
    )
    RK_9_5_5 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 19, (1 / 19,)),
            Stage(1 / 6, (1 / 6, 0)),
            Stage(5 / 16, (5 / 16, 0, 0)),
            Stage(1 / 2, (1 / 2, 0, 0, 0)),
            Stage(11 / 16, (11 / 16, 0, 0, 0, 0)),
            Stage(
                5 / 6,
                (
                    11448031 / 2850816,
                    -67411795275 / 16590798848,
                    51073011 / 43237376,
                    -23353 / 64148,
                    583825 / 8077312,
                    -1 / 116,
                ),
            ),
            Stage(
                16 / 17,
                (
                    30521441823091 / 1986340257792,
                    -745932230071621375 / 35792226257928192,
                    42324456085 / 5966757888,
                    775674925 / 6453417096,
                    -38065236125 / 28020473856,
                    18388001255 / 24775053336,
                    -25 / 138,
                ),
            ),
            Stage(
                1,
                (
                    544015925591990906117739018863 / 21097279127167116142731264000,
                    -51819957177912933732533469147783191 / 1292529408768612025127952939417600,
                    15141148893501140337719772533 / 769541606770966638202880000,
                    -22062343808701233885761491 / 5740046662014404900523000,
                    -180818957612953115541011736739 / 146721986657116762265358336000,
                    18393837528018836258241002593 / 22366927394951953576613895000,
                    -14372715851 / 701966192290,
                    -3316780581 / 34682124125,
                ),
            ),
        ),
        (
            201919428075343316424206867 / 7205146638186855485778750,
            -979811820279525173317561445351 / 23232888464237446713644747250,
            -659616477161155066954978 / 262813990730721440278125,
            10343523856053877739219144704 / 232857239079584284108576875,
            -2224588357354685208355760476 / 50108519801935858605643125,
            704220346724742597999572733952 / 31288349276326419946994221875,
            -13778944 / 1751475,
            92889088 / 11941875,
            -714103988224 / 149255126145,
        ),
    )

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


@enum.unique
class Shanks1965(enum.Enum):
    """Higher order approximations of runge-kutta type, E. B. Shanks
    https://ntrs.nasa.gov/citations/19650022581
    Note RK5_5, RK6_6, RK7_7 and RK8_10 are only approximations of their respective orders.
    """

    RK4_4 = rk4_tableau(1 / 100, 3 / 5)
    RK5_5 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 9000, (1 / 9000,)),
            Stage(3 / 10, tuple(x / 10 for x in (-4047, 4050))),
            Stage(3 / 4, tuple(x / 8 for x in (20241, -20250, 15))),
            Stage(1, tuple(x / 81 for x in (-931041, 931500, -490, 112))),
        ),
        tuple(x / 1134 for x in (105, 0, 500, 448, 81)),
    )
    "Not a true 5th order"
    RK6_6 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 300, (1 / 300,)),
            Stage(1 / 5, tuple(x / 5 for x in (-29, 30))),
            Stage(3 / 5, tuple(x / 5 for x in (323, -330, 10))),
            Stage(14 / 15, tuple(x / 810 for x in (-510104, 521640, -12705, 1925))),
            Stage(1, tuple(x / 77 for x in (-417923, 427350, -10605, 1309, -54))),
        ),
        tuple(x / 3696 for x in (198, 0, 1225, 1540, 810, -77)),
    )
    "Not a true 6th order"
    RK7_7 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 192, (1 / 192,)),
            Stage(1 / 6, tuple(x / 6 for x in (-15, 16))),
            Stage(1 / 2, tuple(x / 186 for x in (4867, -5072, 298))),
            Stage(1, tuple(x / 31 for x in (-19995, 20896, -1025, 155))),
            Stage(5 / 6, tuple(x / 5022 for x in (-469805, 490960, -22736, 5580, 186))),
            Stage(1, tuple(x / 2604 for x in (914314, -955136, 47983, -6510, -558, 2511))),
        ),
        tuple(x / 300 for x in (14, 0, 81, 110, 0, 81, 14)),
    )
    "Not a true 7th order"
    RK7_9 = Tableau(
        (
            Stage(0, ()),
            Stage(2 / 9, (2 / 9,)),
            Stage(1 / 3, tuple(x / 12 for x in (1, 3))),
            Stage(1 / 2, tuple(x / 8 for x in (1, 0, 3))),
            Stage(1 / 6, tuple(x / 216 for x in (23, 0, 21, -8))),
            Stage(8 / 9, tuple(x / 729 for x in (-4136, 0, -13584, 5264, 13104))),
            Stage(1 / 9, tuple(x / 151632 for x in (105131, 0, 302016, -107744, -284256, 1701))),
            Stage(5 / 6, tuple(x / 1375920 for x in (-775229, 0, -2770950, 1735136, 2547216, 81891, 328536))),
            Stage(1, tuple(x / 251888 for x in (23569, 0, -122304, -20384, 695520, -99873, -466560, 241920))),
        ),
        tuple(x / 2140320 for x in (110201, 0, 0, 767936, 635040, -59049, -59049, 635040, 110201)),
    )
    RK8_10 = Tableau(
        (
            Stage(0, ()),
            Stage(4 / 27, (4 / 27,)),
            Stage(2 / 9, tuple(x / 18 for x in (1, 3))),
            Stage(1 / 3, tuple(x / 12 for x in (1, 0, 3))),
            Stage(1 / 2, tuple(x / 8 for x in (1, 0, 0, 3))),
            Stage(2 / 3, tuple(x / 54 for x in (13, 0, -27, 42, 8))),
            Stage(1 / 6, tuple(x / 4320 for x in (389, 0, -54, 966, -824, 243))),
            Stage(1, tuple(x / 20 for x in (-231, 0, 81, -1164, 656, -122, 800))),
            Stage(5 / 6, tuple(x / 288 for x in (-127, 0, 18, -678, 456, -9, 576, 4))),
            Stage(1, tuple(x / 820 for x in (1481, 0, -81, 7104, -3376, 72, -5040, -60, 720))),
        ),
        tuple(x / 840 for x in (41, 0, 0, 27, 272, 27, 216, 0, 216, 41)),
    )
    "Not a true 8th order"
    RK8_12 = Tableau(
        (
            Stage(0, ()),
            Stage(1 / 9, (1 / 9,)),
            Stage(1 / 6, tuple(x / 24 for x in (1, 3))),
            Stage(1 / 4, tuple(x / 16 for x in (1, 0, 3))),
            Stage(1 / 10, tuple(x / 500 for x in (29, 0, 33, -12))),
            Stage(1 / 6, tuple(x / 972 for x in (33, 0, 0, 4, 125))),
            Stage(1 / 2, tuple(x / 36 for x in (-21, 0, 0, 76, 125, -162))),
            Stage(2 / 3, tuple(x / 243 for x in (-30, 0, 0, -32, 125, 0, 99))),
            Stage(1 / 3, tuple(x / 324 for x in (1175, 0, 0, -3456, -6250, 8424, 242, -27))),
            Stage(5 / 6, tuple(x / 324 for x in (293, 0, 0, -852, -1375, 1836, -118, 162, 324))),
            Stage(5 / 6, tuple(x / 1620 for x in (1303, 0, 0, -4260, -6875, 9990, 1030, 0, 0, 162))),
            Stage(1, tuple(x / 4428 for x in (-8595, 0, 0, 30720, 48750, -66096, 378, -729, -1944, -1296, 3240))),
        ),
        tuple(x / 840 for x in (41, 0, 0, 0, 0, 216, 272, 27, 27, 36, 180, 41)),
    )

    def pretty(self) -> str:
        return pretty_tableau(self.value, str(self))

    def tableau(self) -> Tableau:
        return self.value


del V2, V5, V21
