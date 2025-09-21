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
class FunctionalSinglestep(FunctionalSampler):
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

        for n in list(range(len(schedule)))[include]:
            sample = self.step(sample, model, n, schedule, rng)

            if callback:
                callback(sample)

        return sample


@dataclasses.dataclass(frozen=True)
class Heun(FunctionalSinglestep):
    order: int = 2

    @staticmethod
    def max_order() -> int:
        return 2

    def adjust_steps(self, steps: int) -> int:
        return math.ceil(steps / self.order)  # since we skip a call on final step

    def step[T: common.Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        step: int,
        schedule: list[tuple[float, float]],
        rng: FunctionalSampler.RNG[T] | None = None,
    ) -> T:
        prediction = model(sample, *schedule[step])
        sample_next = common.euler(
            sample,
            prediction,
            schedule[step][1],
            schedule[step + 1][1] if step + 1 < len(schedule) else 0,
            self.schedule.sigma_transform,
        )

        if step + 1 < len(schedule) and self.order > 1:
            prediction_next = model(sample_next, *schedule[step + 1])
            return common.euler(
                sample,
                (prediction + prediction_next) / 2,  # type: ignore
                schedule[step][1],
                schedule[step + 1][1],
                self.schedule.sigma_transform,
            )
        else:
            return sample_next
