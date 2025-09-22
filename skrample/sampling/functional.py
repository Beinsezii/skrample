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

        def euler_kt2(k: T, t2: int) -> T:
            return common.euler(sample, k, schedule[step][1], schedule[t2][1], self.schedule.sigma_transform)

        K1: T = model(sample, *schedule[step])

        if self.order > 2 and step_next < len(schedule):
            assert (step + step_next) % 2 == 0
            step_mid = (step + step_next) // 2

            S1: T = euler_kt2(K1, step_mid)

            K2: T = model(S1, *schedule[step_mid])

            if self.order > 3:
                S2: T = euler_kt2(K2, step_mid)

                K3: T = model(S2, *schedule[step_mid])
                S3: T = euler_kt2(K3, step_next)

                K4: T = model(S3, *schedule[step_next])
                return euler_kt2(
                    (K1 + 2 * K2 + 2 * K3 + K4) / 6,  # type: ignore
                    step_next,
                )
            else:
                S2: T = euler_kt2(
                    -K1 + 2 * K2,  # type: ignore
                    step_next,
                )

                K3: T = model(S2, *schedule[step_next])
                return euler_kt2(
                    (K1 + 4 * K2 + K3) / 6,  # type: ignore
                    step_next,
                )
        else:
            S1: T = common.euler(
                sample,
                K1,
                schedule[step][1],
                schedule[step_next][1] if step_next < len(schedule) else 0,
                self.schedule.sigma_transform,
            )

            if step_next < len(schedule) and self.order > 1:
                K2: T = model(S1, *schedule[step_next])
                return euler_kt2(
                    (K1 + K2) / 2,  # type: ignore
                    step_next,
                )
            else:
                return S1
