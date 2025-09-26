import dataclasses
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
class RungeKutta(FunctionalHigher, FunctionalSinglestep):
    type Stage = tuple[float, tuple[float, ...]]
    type Final = tuple[float, ...]
    type Tableau = tuple[tuple[Stage, ...], Final]

    order: int = 2

    @staticmethod
    def max_order() -> int:
        return 4

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
        tableau: RungeKutta.Tableau
        effective_order = self.order if step + 1 < len(schedule) else 1
        if effective_order >= 3:
            if effective_order >= 4:  # RK4
                tableau = (
                    (
                        (0, ()),
                        (1 / 2, (1 / 2,)),
                        (1 / 2, (0, 1 / 2)),
                        (1, (0, 0, 1)),
                    ),
                    (1 / 6, 2 / 6, 2 / 6, 1 / 6),
                )
            else:  # RK3
                tableau = (
                    (
                        (0, ()),
                        (1 / 2, (1 / 2,)),
                        (1, (-1, 2)),
                    ),
                    (1 / 6, 4 / 6, 1 / 6),
                )
        elif effective_order >= 2:  # Heun / RK2
            tableau = (
                (
                    (0, ()),
                    (1, (1,)),
                ),
                (1 / 2, 1 / 2),
            )
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
            k_terms.append(model(combined, *frac_sc))

        return common.euler(
            sample,
            math.sumprod(k_terms, tableau[1]),  # type: ignore
            schedule[step][1],
            schedule[step + 1][1] if step + 1 < len(schedule) else 0,
            self.schedule.sigma_transform,
        )
