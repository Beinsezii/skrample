import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

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
    def step_increment(self) -> int:
        return 1

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
        schedule: list[tuple[float, float]] = self.schedule.schedule(steps * self.step_increment()).tolist()

        for n in list(range(steps))[include]:
            sample = self.step(sample, model, n * self.step_increment(), schedule, rng)

            if callback:
                callback(sample)

        return sample


@dataclasses.dataclass(frozen=True)
class RungeKutta(FunctionalHigher, FunctionalSinglestep):
    type Stage = tuple[int, tuple[float, ...]]
    type Final = tuple[float, ...]
    type Tableau = tuple[tuple[Stage, ...], Final]

    order: int = 2

    @staticmethod
    def max_order() -> int:
        return 4

    def adjust_steps(self, steps: int) -> int:
        return math.ceil(steps / self.order)  # since we skip a call on final step

    def step_increment(self) -> int:
        return 2 if self.order > 2 else 1

    def step[T: common.Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        step: int,
        schedule: list[tuple[float, float]],
        rng: FunctionalSampler.RNG[T] | None = None,
    ) -> T:
        step_next = step + self.step_increment()

        tableau: RungeKutta.Tableau
        effective_order = self.order if step_next < len(schedule) else 1
        if effective_order >= 3:
            assert (step + step_next) % 2 == 0
            step_mid = (step + step_next) // 2
            if effective_order >= 4:  # RK4
                tableau = (
                    (
                        (step, ()),
                        (step_mid, (1,)),
                        (step_mid, (0, 1)),
                        (step_next, (0, 0, 1)),
                    ),
                    (1 / 6, 2 / 6, 2 / 6, 1 / 6),
                )
            else:  # RK3
                tableau = (
                    (
                        (step, ()),
                        (step_mid, (1,)),
                        (step_next, (-1, 2)),
                    ),
                    (1 / 6, 4 / 6, 1 / 6),
                )
        elif effective_order >= 2:  # Heun / RK2
            tableau = (
                (
                    (step, ()),
                    (step_next, (1,)),
                ),
                (1 / 2, 1 / 2),
            )
        else:  # Euler / RK1
            tableau = (
                ((step, ()),),
                (1,),
            )

        k_terms: list[T] = []
        for istep, icoeffs in tableau[0]:
            if icoeffs:
                combined: T = common.euler(
                    sample,
                    math.sumprod(k_terms, icoeffs),  # type: ignore
                    schedule[step][1],
                    schedule[istep][1] if istep < len(schedule) else 0,
                    self.schedule.sigma_transform,
                )
            else:
                combined = sample
            k_terms.append(model(combined, *schedule[istep]))

        return common.euler(
            sample,
            math.sumprod(k_terms, tableau[1]),  # type: ignore
            schedule[step][1],
            schedule[step_next][1] if step_next < len(schedule) else 0,
            self.schedule.sigma_transform,
        )
