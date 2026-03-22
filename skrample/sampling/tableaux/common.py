import dataclasses
import math
from collections.abc import MutableSequence, Sequence
from typing import NamedTuple, Self


class Stage(NamedTuple):
    c: float
    a: tuple[float, ...]  # has to be hashable


class Tableau(NamedTuple):
    stages: tuple[Stage, ...]
    weights: tuple[float, ...]


class EmbeddedTableau(NamedTuple):
    stages: tuple[Stage, ...]
    weights: tuple[float, ...]
    error_weights: tuple[float, ...]

    def unembed(self) -> Tableau:
        return Tableau(self.stages, self.weights)


type TableauType = Tableau | EmbeddedTableau


@dataclasses.dataclass(frozen=True)
class ButcherCoeffs:
    one_index: bool
    c: MutableSequence[float]
    a: Sequence[MutableSequence[float]]
    b: MutableSequence[float]

    @classmethod
    def empty(cls, stages: int, fill: float = -math.inf, one_index: bool = False) -> Self:
        c = [fill] * (stages + one_index)
        a = [[fill] * n for n in range(0, (stages + one_index))]
        b = [fill] * (stages + one_index)
        # implicit coeffs
        c[one_index] = 0
        return cls(one_index, c=c, a=a, b=b)

    def compute_c(self) -> None:
        self.c[:] = [math.fsum(s) for s in self.a]

    def compose(self) -> Tableau:
        return Tableau(
            tuple(
                Stage(cx, tuple(ax[self.one_index :]))
                for cx, ax in zip(self.c[self.one_index :], self.a[self.one_index :], strict=True)
            ),
            tuple(self.b[self.one_index :]),
        )

    @classmethod
    def decompose(cls, tableau: Tableau) -> Self:
        return cls(
            False,
            c=[s.c for s in tableau.stages],
            a=[list(s.a) for s in tableau.stages],
            b=list(tableau.weights),
        )

    @classmethod
    def deserialize(cls, coeffs: list[float], stages: int, compute_c: bool = False, b_last: bool = True) -> Self:
        i: int = 0

        t = cls.empty(stages)

        assert len(coeffs) == len(t.c) * (not compute_c) + len(t.b) + sum(len(aa) for aa in t.a)

        if not compute_c:
            for n in range(len(t.c)):
                t.c[n] = coeffs[i]
                i += 1

        if not b_last:
            for n in range(len(t.b)):
                t.b[n] = coeffs[i]
                i += 1

        for x in range(1, len(t.a)):
            for y in range(len(t.a[x])):
                t.a[x][y] = coeffs[i]
                i += 1

        if compute_c:
            t.compute_c()

        if b_last:
            for n in range(len(t.b)):
                t.b[n] = coeffs[i]
                i += 1

        return t

    def serialize(self) -> Sequence[float]:
        return [*self.c, *(x for a in self.a for x in a), *self.b]

    @classmethod
    def from_shu_osher(cls, alphas: Sequence[Sequence[float]], betas: Sequence[Sequence[float]]) -> Self:
        stages = len(alphas)
        t = cls.empty(stages)

        # Compute internal stages of the Butcher matrix 'a'
        # i represents the Butcher stage index (1 to stages-1)
        for i in range(1, stages):
            # j represents the Butcher column index (0 to i-1)
            for j in range(i):
                # The summation term accounts for the recursive dependency on previous stages
                t.a[i][j] = math.fsum((betas[i - 1][j], *(alphas[i - 1][k] * t.a[k][j] for k in range(j + 1, i))))

        # Compute Butcher weights 'b' using the final Shu-Osher row
        for j in range(stages):
            # Final update is treated as the result of the final coefficient row
            t.b[j] = math.fsum(
                (betas[stages - 1][j], *(alphas[stages - 1][k] * t.a[k][j] for k in range(j + 1, stages)))
            )

        t.compute_c()

        return t


def pretty_tableau(tableau: TableauType, label: str | None = None) -> str:

    def pretnum(x: float) -> str:
        return f"{'+' if x >= 0 else '-'}{float(round(abs(x), 4)): <6}"

    stages: list[str] = [f"{pretnum(c)} | {' '.join(pretnum(x) for x in a)}" for c, a in tableau[0]]

    weights: list[str] = ["        | " + " ".join(pretnum(x) for x in w) for w in tableau[1:]]

    width = max(len(x) for x in (*weights, *stages))

    lines: list[str] = [label.rjust((width + len(label)) // 2)] if label is not None else []

    lines.extend((*stages, "-" * width, *weights))

    return "\n".join(lines)


def validate_tableau(tab: TableauType, tolerance: float = 1e-12) -> None | IndexError | ValueError:
    for index, stage in enumerate(tab.stages):
        if index != (stage_len := len(stage.a)):
            return IndexError(f"{index=}, {stage_len=}, {stage=}")
        if tolerance < (stage_err := abs(stage.c - math.fsum(stage[1]))):
            return ValueError(f"{tolerance=}, {stage_err=}, {stage=}")

    for weight in tab[1:]:
        if (stage_count := len(tab.stages)) != (weight_len := len(weight)):
            return IndexError(f"{stage_count=}, {weight_len=}, {weight=}")
        if tolerance < (weight_err := abs(1 - math.fsum(weight))):
            return ValueError(f"{tolerance=}, {weight_err=}, {weight=}")
