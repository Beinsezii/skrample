import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import numpy as np

from skrample import common, scheduling
from skrample.common import RNG, DictOrProxy, FloatSchedule, Sample, SigmaTransform

from . import models, tableaux

type SampleCallback[T: Sample] = Callable[[T, int, float, float], Any]
"Return is ignored"
type SampleableModel[T: Sample] = Callable[[T, float, float], T]
"sample, timestep, sigma"


def fractional_step(
    schedule: FloatSchedule,
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


def step_tableau[T: Sample](
    tableau: tableaux.Tableau | tableaux.ExtendedTableau,
    sample: T,
    model: SampleableModel[T],
    model_transform: models.ModelTransform,
    step: int,
    schedule: FloatSchedule,
    sigma_transform: SigmaTransform,
    derivative_transform: models.ModelTransform | None = None,
    step_size: int = 1,
    epsilon: float = 1e-8,
) -> tuple[T, ...]:
    nodes, weights = tableau[0], tableau[1:]

    if len(nodes) > 1 and derivative_transform:
        model = models.ModelConvert(model_transform, derivative_transform).wrap_model_call(model, sigma_transform)
        model_transform = derivative_transform

    derivatives: list[T] = []
    S0 = schedule[step][1]
    S1 = schedule[step + step_size][1] if step + step_size < len(schedule) else 0

    fractions = fractional_step(schedule, step, tuple(f[0] * step_size for f in nodes))

    for frac_sc, icoeffs in zip(fractions, (t[1] for t in nodes), strict=True):
        sigma_i = frac_sc[1]
        if icoeffs:
            X: T = model_transform.forward(  # pyright: ignore [reportAssignmentType]
                sample,
                math.sumprod(derivatives, icoeffs) / math.fsum(icoeffs),  # pyright: ignore [reportArgumentType]
                S0,
                sigma_i,
                sigma_transform,
            )
        else:
            X = sample

        # Do not call model on timestep = 0 or sigma = 0
        if any(abs(v) < epsilon for v in frac_sc):
            derivatives.append(model_transform.backward(sample, X, S0, S1, sigma_transform))
        else:
            derivatives.append(model(X, *frac_sc))

    return tuple(  # pyright: ignore [reportReturnType]
        model_transform.forward(
            sample,
            math.sumprod(derivatives, w),  # pyright: ignore [reportArgumentType]
            S0,
            S1,
            sigma_transform,
        )
        for w in weights
    )


@dataclasses.dataclass(frozen=True)
class FunctionalSampler(ABC):
    schedule: scheduling.SkrampleSchedule

    def merge_noise[T: Sample](self, sample: T, noise: T, steps: int, start: int) -> T:
        sigmas = self.schedule.sigmas(steps)
        sigma = sigmas[start] if start < len(sigmas) else 0
        return common.merge_noise(sample, noise, sigma, self.schedule.sigma_transform)

    @abstractmethod
    def sample_model[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        model_transform: models.ModelTransform,
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
        model_transform: models.ModelTransform,
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
            sample: T = self.merge_noise(
                0 if initial is None else initial,  # type: ignore
                rng(),
                steps,
                include.start or 0,
            ) / self.merge_noise(0.0, 1.0, steps, 0)
            # Rescale sample by initial sigma. Mostly just to handle quirks with Scaled

        return self.sample_model(sample, model, model_transform, steps, include, rng, callback)


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
class FunctionalDerivative(FunctionalHigher):
    derivative_transform: models.ModelTransform | None = models.DiffusionModel()  # noqa: RUF009 # is immutable
    "Transform model to this space when computing higher order samples."


@dataclasses.dataclass(frozen=True)
class FunctionalSinglestep(FunctionalSampler):
    @abstractmethod
    def step[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        model_transform: models.ModelTransform,
        step: int,
        schedule: FloatSchedule,
        rng: RNG[T] | None = None,
    ) -> T: ...

    def sample_model[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        model_transform: models.ModelTransform,
        steps: int,
        include: slice = slice(None),
        rng: RNG[T] | None = None,
        callback: SampleCallback | None = None,
    ) -> T:
        schedule: FloatSchedule = self.schedule.schedule(steps)

        for n in list(range(steps))[include]:
            sample = self.step(sample, model, model_transform, n, schedule, rng)

            if callback:
                callback(sample, n, *schedule[n] if n < len(schedule) else (0, 0))

        return sample


@dataclasses.dataclass(frozen=True)
class FunctionalAdaptive(FunctionalSampler):
    type Evaluator[T: Sample] = Callable[[T, T], float]

    @staticmethod
    def mse[T: Sample](a: T, b: T) -> float:
        error: T = abs(a - b) ** 2  # type: ignore
        return common.mean(error)

    evaluator: Evaluator = mse
    "Function used to measure error of two samples"
    threshold: float = 1e-2
    "Target error threshold for a given evaluation"


@dataclasses.dataclass(frozen=True)
class RKUltra(FunctionalDerivative, FunctionalSinglestep):
    "Implements almost every single method from https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods"  # noqa: RUF002

    order: int = 2

    providers: DictOrProxy[int, tableaux.TableauProvider[tableaux.Tableau | tableaux.ExtendedTableau]] = (
        MappingProxyType(
            {
                2: tableaux.RK2.Heun,
                3: tableaux.RK3.Ralston,
                4: tableaux.RK4.Ralston,
                5: tableaux.RK5.Nystrom,
            }
        )
    )
    """Providers for a given order, starting from 2.
    Order 1 is always the Euler method."""

    @staticmethod
    def max_order() -> int:
        return 99

    def tableau(self, order: int | None = None) -> tableaux.Tableau:
        if order is None:
            order = self.order

        if order >= 2 and (morder := max(o for o in self.providers.keys() if o <= order)):
            return self.providers[morder].tableau()[:2]
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
        model: SampleableModel[T],
        model_transform: models.ModelTransform,
        step: int,
        schedule: FloatSchedule,
        rng: RNG[T] | None = None,
    ) -> T:
        return step_tableau(
            self.tableau(),
            sample,
            model,
            model_transform,
            step,
            schedule,
            self.schedule.sigma_transform,
            derivative_transform=self.derivative_transform,
        )[0]


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
        model: SampleableModel[T],
        model_transform: models.ModelTransform,
        step: int,
        schedule: FloatSchedule,
        rng: RNG[T] | None = None,
    ) -> T:
        k1 = model(sample, *schedule[step])
        sigma_from = schedule[step][1]
        sigma_to = schedule[step + 1][1] if step + 1 < len(schedule) else 0
        result: T = model_transform.forward(sample, k1, sigma_from, sigma_to, self.schedule.sigma_transform)

        # Multiplying by step size here is kind of an asspull, but so is this whole solver so...
        if step + 1 < len(schedule) and self.evaluator(sample, result) / max(
            self.evaluator(0, result), 1e-16
        ) > self.threshold * abs(sigma_from - sigma_to):
            k2: T = (k1 + model(result, *schedule[step + 1])) / 2  # pyright: ignore [reportAssignmentType]
            result: T = model_transform.forward(sample, k2, sigma_from, sigma_to, self.schedule.sigma_transform)

        return result


@dataclasses.dataclass(frozen=True)
class RKMoire(FunctionalAdaptive, FunctionalDerivative):
    order: int = 2

    providers: DictOrProxy[int, tableaux.TableauProvider[tableaux.ExtendedTableau]] = MappingProxyType(
        {
            2: tableaux.RKE2.Heun,
            5: tableaux.RKE5.Fehlberg,
        }
    )
    """Providers for a given order, starting from 2.
    Falls back to RKE2.Heun"""

    threshold: float = 1e-4

    initial: float = 1 / 50
    "Percent of schedule to take as an initial step."
    maximum: float = 1 / 4
    "Percent of schedule to take as a maximum step."
    adaption: float = 0.3
    "How fast to adjust step size in relation to error"
    discard: float = float("inf")
    "If the final adjustment down is more than this, the entire previous step is discarded."

    rescale_init: bool = True
    "Scale initial by a tableau's model evals."
    rescale_max: bool = False
    "Scale maximum by a tableau's model evals."

    @staticmethod
    def min_order() -> int:
        return 2

    @staticmethod
    def max_order() -> int:
        return 99

    def adjust_steps(self, steps: int) -> int:
        return steps

    def tableau(self, order: int | None = None) -> tableaux.ExtendedTableau:
        if order is None:
            order = self.order

        if order >= 2 and (morder := max(o for o in self.providers.keys() if o <= order)):
            return self.providers[morder].tableau()
        else:
            return tableaux.RKE2.Heun.tableau()

    def sample_model[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        model_transform: models.ModelTransform,
        steps: int,
        include: slice = slice(None),
        rng: RNG[T] | None = None,
        callback: SampleCallback | None = None,
    ) -> T:
        tab = self.tableau()

        initial = self.initial
        maximum = self.maximum
        if self.rescale_init:
            initial *= len(tab[0]) / 2  # Heun is base so / 2
        if self.rescale_max:
            maximum *= len(tab[0]) / 2  # Heun is base so / 2

        step_size: int = max(round(steps * initial), 1)
        epsilon: float = 1e-16  # lgtm

        schedule: FloatSchedule = self.schedule.schedule(steps)

        indices: list[int] = list(range(steps))[include]
        step: int = indices[0]

        while step <= indices[-1]:
            step_next = min(step + step_size, indices[-1] + 1)

            if step_next < len(schedule):
                sample_high, sample_low = step_tableau(
                    tab,
                    sample,
                    model,
                    model_transform,
                    step,
                    schedule,
                    self.schedule.sigma_transform,
                    self.derivative_transform,
                    step_size,
                )

                sigma0 = schedule[step][1]
                sigma1 = schedule[step_next][1]
                sigma2 = schedule[step_next + step_size][1] if step_next + step_size < len(schedule) else 0
                # Offset adjustment by dt2 / dt to account for non-linearity
                # Basically if we want a 50% larger step but the next dt will already be 25% larger,
                # we should only set a 20% larger step ie 1.5 / 1.25
                slope = abs(sigma0 - sigma1) / abs(sigma1 - sigma2)

                # Normalize against pure error
                error = self.evaluator(sample_low, sample_high) / max(self.evaluator(0, sample_high), epsilon)
                adjustment: float = (self.threshold / max(error, epsilon)) ** self.adaption / slope
                step_size = max(round(min(step_size * adjustment, steps * maximum)), 1)

                # Only discard if it will actually decrease step size
                if step_next - step > step_size and 1 / max(adjustment, epsilon) > self.discard:
                    continue

            else:  # Save the extra euler call since the 2nd weight isn't used
                sample_high = step_tableau(
                    tab[:2],
                    sample,
                    model,
                    model_transform,
                    step,
                    schedule,
                    self.schedule.sigma_transform,
                    self.derivative_transform,
                    step_size,
                )[0]

            sample = sample_high

            if callback:
                callback(sample, step_next - 1, *schedule[step] if step < len(schedule) else (0, 0))

            step = step_next

        return sample
