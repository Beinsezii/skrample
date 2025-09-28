import dataclasses
import enum
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from skrample import common, scheduling


@dataclasses.dataclass(frozen=True)
class FunctionalSampler(ABC):
    type SampleCallback[T: common.Sample] = Callable[[T], Any]
    "Return is ignored"
    type SampleableModel[T: common.Sample] = Callable[[T, float, float], T]
    "sample, timestep, sigma"
    type RNG[T: common.Sample] = Callable[[], T]
    "Distribution should match model, typically normal"

    schedule: scheduling.SkrampleSchedule

    @abstractmethod
    def sample_model[T: common.Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        steps: int,
        include: slice = slice(None),
        rng: RNG[T] | None = None,
        callback: SampleCallback | None = None,
    ) -> T: ...


@dataclasses.dataclass(frozen=True)
class FunctionalHigher(FunctionalSampler):
    order: int = 1

    @staticmethod
    def min_order() -> int:
        return 1

    @staticmethod
    @abstractmethod
    def max_order() -> int: ...

    def adjust_steps(self, steps: int) -> int:
        "Adjust the steps to approximate an equal amount of model calls"
        return round(steps / self.order)


@dataclasses.dataclass(frozen=True)
class FunctionalSinglestep(FunctionalSampler):
    @abstractmethod
    def step[T: common.Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        step: int,
        schedule: list[tuple[float, float]],
        rng: FunctionalSampler.RNG[T] | None = None,
    ) -> T: ...

    def sample_model[T: common.Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        steps: int,
        include: slice = slice(None),
        rng: FunctionalSampler.RNG[T] | None = None,
        callback: FunctionalSampler.SampleCallback | None = None,
    ) -> T:
        schedule: list[tuple[float, float]] = self.schedule.schedule(steps).tolist()

        for n in list(range(steps))[include]:
            sample = self.step(sample, model, n, schedule, rng)

            if callback:
                callback(sample)

        return sample


@dataclasses.dataclass(frozen=True)
class RKUltra(FunctionalHigher, FunctionalSinglestep):
    "Implements almost every single method from https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods"  # noqa: RUF002

    type Tableau = tuple[
        tuple[
            tuple[float, tuple[float, ...]],
            ...,
        ],
        tuple[float, ...],
    ]

    @enum.unique
    class RK2(enum.StrEnum):
        Heun = enum.auto()
        Mid = enum.auto()
        Ralston = enum.auto()

        def tableau(self) -> "RKUltra.Tableau":
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

        def tableau(self) -> "RKUltra.Tableau":
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

        def tableau(self) -> "RKUltra.Tableau":
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

        def tableau(self) -> "RKUltra.Tableau":
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

    order: int = 2
    rk2: RK2 = RK2.Ralston
    rk3: RK3 = RK3.Ralston
    rk4: RK4 = RK4.Ralston
    rk5: RK5 = RK5.Nystrom

    @staticmethod
    def max_order() -> int:
        return 5

    def adjust_steps(self, steps: int) -> int:
        return math.ceil(steps / self.order)  # since we skip a call on final step

    @staticmethod
    def fractional_step(
        schedule: list[tuple[float, float]],
        current: int,
        idx: tuple[float, ...],
    ) -> tuple[tuple[float, float], ...]:
        schedule_np = np.array([*schedule, (0, 0)])
        idx_np = np.array(idx) / len(schedule) + current / len(schedule)
        scale = np.linspace(0, 1, len(schedule_np))

        # TODO (beinszeii): better 2d interpolate
        result = tuple(
            zip(
                (np.interp(idx_np, scale, schedule_np[:, 0])).tolist(),
                (np.interp(idx_np, scale, schedule_np[:, 1])).tolist(),
            )
        )
        return result

    def step[T: common.Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        step: int,
        schedule: list[tuple[float, float]],
        rng: FunctionalSampler.RNG[T] | None = None,
    ) -> T:
        if self.order >= 5:
            tableau = self.rk5.tableau()
        elif self.order >= 4:
            tableau = self.rk4.tableau()
        elif self.order >= 3:
            tableau = self.rk3.tableau()
        elif self.order >= 2:
            tableau = self.rk2.tableau()
        else:  # Euler / RK1
            tableau = (
                ((0, ()),),
                (1,),
            )

        k_terms: list[T] = []
        fractions = self.fractional_step(schedule, step, tuple(f[0] for f in tableau[0]))
        for frac_sc, icoeffs in zip(fractions, (t[1] for t in tableau[0]), strict=True):
            if icoeffs:
                combined: T = common.euler(
                    sample,
                    math.sumprod(k_terms, icoeffs) / math.fsum(icoeffs),  # type: ignore
                    schedule[step][1],
                    frac_sc[1],
                    self.schedule.sigma_transform,
                )
            else:
                combined = sample

            # Do not call model on timestep = 0 or sigma = 0
            k_terms.append(model(combined, *frac_sc) if not any(abs(v) < 1e-8 for v in frac_sc) else combined)

        return common.euler(
            sample,
            math.sumprod(k_terms, tableau[1]),  # type: ignore
            schedule[step][1],
            schedule[step + 1][1] if step + 1 < len(schedule) else 0,
            self.schedule.sigma_transform,
        )
