import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from skrample import common, scheduling
from skrample.common import Sample, SigmaTransform

from . import tableaux


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

    order: int = 2

    providers: tuple[tableaux.TableauProvider | tableaux.ExtendedTableauProvider, ...] = (
        tableaux.RK2.Ralston,
        tableaux.RK3.Ralston,
        tableaux.RK4.Ralston,
        tableaux.RK5.Nystrom,
    )
    """Providers for a given order, starting from 2.
    Order 1 is always the Euler method."""

    custom_tableau: tableaux.Tableau | tableaux.ExtendedTableau | None = None
    "If set, will use this Butcher tableau instead of picking method based on `RKUltra.order`"

    @staticmethod
    def max_order() -> int:
        return 5

    def tableau(self, order: int | None = None) -> tableaux.Tableau:
        if self.custom_tableau is not None:
            return self.custom_tableau[:2]
        elif order is None:
            order = self.order

        if order >= 2 and (morder := len(self.providers)):
            return self.providers[min(order - 2, morder - 1)].tableau()[:2]
        else:  # Euler / RK1
            return tableaux.RK1

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
    order: int = 2

    providers: tuple[tableaux.ExtendedTableauProvider, ...] = (
        tableaux.RKE2.Heun,
        tableaux.RKE2.Heun,
        tableaux.RKE2.Heun,
        tableaux.RKE5.Fehlberg,
    )

    threshold: float = 1e-3

    initial: float = 1 / 50
    maximum: float = 1 / 4
    adaption: float = 0.3

    rescale_init: bool = True
    "Scale initial by a tableau's model evals."

    custom_tableau: tableaux.ExtendedTableau | None = None
    "If set, will use this Butcher tableau instead of picking method based on `RKUltra.order`"

    @staticmethod
    def min_order() -> int:
        return 2

    @staticmethod
    def max_order() -> int:
        return 5

    def adjust_steps(self, steps: int) -> int:
        return steps

    def tableau(self, order: int | None = None) -> tableaux.ExtendedTableau:
        if self.custom_tableau is not None:
            return self.custom_tableau
        elif order is None:
            order = self.order

        if order >= 2 and (morder := len(self.providers)):
            return self.providers[min(order - 2, morder - 1)].tableau()
        else:
            return tableaux.RKE2.Heun.tableau()

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
