import dataclasses
import enum
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from skrample import common, scheduling
from skrample.common import Sample, SigmaTransform


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


@dataclasses.dataclass(frozen=True)
class FunctionalSampler(ABC):
    type SampleCallback[T: Sample] = Callable[[T, int, float, float], Any]
    "Return is ignored"
    type SampleableModel[T: Sample] = Callable[[T, float, float], T]
    "sample, timestep, sigma"
    type RNG[T: Sample] = Callable[[], T]
    "Distribution should match model, typically normal"

    schedule: scheduling.SkrampleSchedule

    def merge_noise[T: Sample](self, sample: T, noise: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return common.merge_noise(sample, noise, sigma, sigma_transform)

    @abstractmethod
    def sample_model[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        steps: int,
        include: slice = slice(None),
        rng: RNG[T] | None = None,
        callback: SampleCallback | None = None,
    ) -> T: ...

    """Runs the noisy sample through the model for a given range `include` of total steps.
    Calls callback every step with sampled value."""

    def generate_model[T: Sample](
        self,
        model: SampleableModel[T],
        rng: RNG[T],
        steps: int,
        include: slice = slice(None),
        initial: T | None = None,
        callback: SampleCallback | None = None,
    ) -> T:
        """Equivalent to `sample_model` except the noise is handled automatically
        rather than being pre-added to the initial value"""

        if initial is None and include.start is None:  # Short circuit for common case
            sample: T = rng()
        else:
            sigmas = scheduling.schedule_lru(self.schedule, steps)[:, 1]
            sample: T = self.merge_noise(
                0 if initial is None else initial,  # type: ignore
                rng(),
                sigmas[include.start or 0].item(),
                self.schedule.sigma_transform,
            ) / self.merge_noise(0.0, 1.0, sigmas[0].item(), self.schedule.sigma_transform)
            # Rescale sample by initial sigma. Mostly just to handle quirks with Scaled

        return self.sample_model(sample, model, steps, include, rng, callback)


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
    def step[T: Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        step: int,
        schedule: list[tuple[float, float]],
        rng: FunctionalSampler.RNG[T] | None = None,
    ) -> T: ...

    def sample_model[T: Sample](
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
                callback(sample, n, *schedule[n] if n < len(schedule) else (0, 0))

        return sample


@dataclasses.dataclass(frozen=True)
class FunctionalAdaptive(FunctionalSampler):
    type Evaluator[T: Sample] = Callable[[T, T], float]

    @staticmethod
    def mse[T: Sample](a: T, b: T) -> float:
        error: T = abs(a - b) ** 2  # type: ignore
        if isinstance(error, float | int):
            return error
        else:
            return error.mean().item()

    evaluator: Evaluator = mse
    threshold: float = 1e-2


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
    "2nd order methods"
    rk3: RK3 = RK3.Ralston
    "3rd order methods"
    rk4: RK4 = RK4.Ralston
    "4th order methods"
    rk5: RK5 = RK5.Nystrom
    "5th order methods"

    custom_tableau: Tableau | None = None
    "If set, will use this Butcher tableau instead of picking method based on `RKUltra.order`"

    @staticmethod
    def max_order() -> int:
        return 5

    def tableau(self, order: int | None = None) -> Tableau:
        if self.custom_tableau is not None:
            return self.custom_tableau
        elif order is None:
            order = self.order

        if order >= 5:
            return self.rk5.tableau()
        elif order >= 4:
            return self.rk4.tableau()
        elif order >= 3:
            return self.rk3.tableau()
        elif order >= 2:
            return self.rk2.tableau()
        else:  # Euler / RK1
            return (
                ((0, ()),),
                (1,),
            )

    def adjust_steps(self, steps: int) -> int:
        stages = self.tableau()[0]
        calls = len(stages)

        # Add back the skipped calls on penultimate T
        adjusted = steps / calls + sum(abs(1 - f[0]) < 1e-8 for f in stages) / calls

        return max(round(adjusted), 1)

    def step[T: Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        step: int,
        schedule: list[tuple[float, float]],
        rng: FunctionalSampler.RNG[T] | None = None,
    ) -> T:
        stages, composite = self.tableau()
        k_terms: list[T] = []
        fractions = fractional_step(schedule, step, tuple(f[0] for f in stages))

        for frac_sc, icoeffs in zip(fractions, (t[1] for t in stages), strict=True):
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
            math.sumprod(k_terms, composite),  # type: ignore
            schedule[step][1],
            schedule[step + 1][1] if step + 1 < len(schedule) else 0,
            self.schedule.sigma_transform,
        )


@dataclasses.dataclass(frozen=True)
class FastHeun(FunctionalAdaptive, FunctionalSinglestep, FunctionalHigher):
    order: int = 2

    threshold: float = 5e-2

    @staticmethod
    def min_order() -> int:
        return 2

    @staticmethod
    def max_order() -> int:
        return 2

    def adjust_steps(self, steps: int) -> int:
        return round(steps * 0.75 + 0.25)

    def step[T: Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        step: int,
        schedule: list[tuple[float, float]],
        rng: FunctionalSampler.RNG[T] | None = None,
    ) -> T:
        sigma = schedule[step][1]
        sigma_next = schedule[step + 1][1] if step + 1 < len(schedule) else 0

        sigma_u, sigma_v = self.schedule.sigma_transform(sigma)
        sigma_u_next, sigma_v_next = self.schedule.sigma_transform(sigma_next)

        scale = sigma_u_next / sigma_u
        dt = sigma_v_next - sigma_v * scale

        k1 = model(sample, *schedule[step])
        result: T = sample * scale + k1 * dt  # type: ignore

        # Multiplying by step size here is kind of an asspull, but so is this whole solver so...
        if (
            step + 1 < len(schedule)
            and self.evaluator(sample, result) / max(self.evaluator(0, result), 1e-16) > self.threshold * dt
        ):
            k2 = model(result, *schedule[step + 1])
            result: T = sample * scale + (k1 + k2) / 2 * dt  # type: ignore

        return result


@dataclasses.dataclass(frozen=True)
class RKMoire(FunctionalAdaptive, FunctionalHigher):
    type ExtendedTableau = tuple[
        tuple[
            tuple[float, tuple[float, ...]],
            ...,
        ],
        tuple[float, ...],
        tuple[float, ...],
    ]

    order: int = 2

    threshold: float = 1e-3

    initial: float = 1 / 50
    maximum: float = 1 / 4
    adaption: float = 0.3

    rescale_init: bool = True
    "Scale initial by a tableau's model evals."

    @enum.unique
    class RKE2(enum.StrEnum):
        Heun = enum.auto()
        # Fehlberg = enum.auto()

        def tableau(self) -> "RKMoire.ExtendedTableau":
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

        def tableau(self) -> "RKMoire.ExtendedTableau":
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

    custom_tableau: ExtendedTableau | None = None
    rke2: RKE2 = RKE2.Heun
    rke5: RKE5 = RKE5.Fehlberg

    @staticmethod
    def min_order() -> int:
        return 2

    @staticmethod
    def max_order() -> int:
        return 5

    def adjust_steps(self, steps: int) -> int:
        return steps

    def tableau(self, order: int | None = None) -> ExtendedTableau:
        if self.custom_tableau is not None:
            return self.custom_tableau
        elif order is None:
            order = self.order

        if order >= 5:
            return self.rke5.tableau()
        else:
            return self.rke2.tableau()

    def sample_model[T: Sample](
        self,
        sample: T,
        model: FunctionalSampler.SampleableModel[T],
        steps: int,
        include: slice = slice(None),
        rng: FunctionalSampler.RNG[T] | None = None,
        callback: FunctionalSampler.SampleCallback | None = None,
    ) -> T:
        stages, comp_high, comp_low = self.tableau()

        initial = self.initial
        if self.rescale_init:
            initial *= len(stages) / 2  # Heun is base so / 2

        step_size: int = max(round(steps * initial), 1)
        epsilon: float = 1e-16  # lgtm

        schedule: list[tuple[float, float]] = self.schedule.schedule(steps).tolist()

        indices: list[int] = list(range(steps))[include]
        step: int = indices[0]

        while step <= indices[-1]:
            step_next = min(step + step_size, indices[-1] + 1)

            k_terms: list[T] = []
            fractions = fractional_step(schedule, step, tuple(f[0] * step_size for f in stages))

            for frac_sc, icoeffs in zip(fractions, (t[1] for t in stages), strict=True):
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

            sample_high: T = common.euler(
                sample,
                math.sumprod(k_terms, comp_high),  # type: ignore
                schedule[step][1],
                schedule[step_next][1] if step_next < len(schedule) else 0,
                self.schedule.sigma_transform,
            )

            if step_next < len(schedule):
                sigma = schedule[step][1]
                sigma_next = schedule[step_next][1] if step_next < len(schedule) else 0
                sigma_next2 = schedule[step_next + step_size][1] if step_next + step_size < len(schedule) else 0

                sigma_u, sigma_v = self.schedule.sigma_transform(sigma)
                sigma_u_next, sigma_v_next = self.schedule.sigma_transform(sigma_next)
                sigma_u_next2, sigma_v_next2 = self.schedule.sigma_transform(sigma_next2)

                dt = sigma_v_next - sigma_v * (sigma_u_next / sigma_u)
                dt2 = sigma_v_next2 - sigma_v_next * (sigma_u_next2 / sigma_u_next)
                dt1x2 = dt2 / dt

                sample_low: T = common.euler(
                    sample,
                    math.sumprod(k_terms, comp_low),  # type: ignore
                    schedule[step][1],
                    schedule[step_next][1] if step_next < len(schedule) else 0,
                    self.schedule.sigma_transform,
                )

                # Normalize against pure error
                error = self.evaluator(sample_low, sample_high) / max(self.evaluator(0, sample_high), epsilon)
                # Offset adjustment by dt2 / dt to account for non-linearity
                # Basically if we want a 50% larger step but the next dt will already be 25% larger,
                # we should only set a 20% larger step ie 1.5 / 1.25
                # Really this could be iterated to contrast dt2/dt and thresh/error until they're 100% matched but eh
                adjustment: float = (self.threshold / max(error, epsilon)) ** self.adaption / dt1x2
                step_size = max(round(min(step_size * adjustment, steps * self.maximum)), 1)

            sample = sample_high

            if callback:
                callback(sample, step_next - 1, *schedule[step] if step < len(schedule) else (0, 0))

            step = step_next

        return sample
