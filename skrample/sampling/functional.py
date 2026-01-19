import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from skrample import common, scheduling
from skrample.common import RNG, DictOrProxy, FloatSchedule, Sample, SigmaTransform, Step

from . import models, tableaux

type SampleCallback[T: Sample] = Callable[[T, int, float, float], Any]
"Return is ignored"
type SampleableModel[T: Sample] = Callable[[T, float, float], T]
"sample, timestep, sigma"


def step_tableau[T: Sample](
    tableau: tableaux.Tableau | tableaux.ExtendedTableau,
    sample: T,
    model: SampleableModel[T],
    model_transform: models.DiffusionModel,
    schedule: scheduling.SkrampleSchedule,
    step: Step,
    derivative_transform: models.DiffusionModel | None = None,
    epsilon: float = 1e-8,
) -> tuple[T, ...]:
    nodes, weights = tableau[0], tableau[1:]

    if derivative_transform:
        model = models.ModelConvert(
            model_transform,
            derivative_transform,
        ).wrap_model_call(model, schedule.sigma_transform)
        model_transform = derivative_transform

    derivatives: list[T] = []
    S0, S1 = schedule.ipoints(step)[:, 1].tolist()
    fractions: FloatSchedule = schedule.ipoints([step[0] + f[0] * (step[1] - step[0]) for f in nodes]).tolist()  # type: ignore

    for frac_sc, icoeffs in zip(fractions, (t[1] for t in nodes), strict=True):
        sigma_i = frac_sc[1]
        if icoeffs:
            X: T = model_transform.forward(  # pyright: ignore [reportAssignmentType]
                sample,
                math.sumprod(derivatives, icoeffs) / math.fsum(icoeffs),  # pyright: ignore [reportArgumentType]
                S0,
                sigma_i,
                schedule.sigma_transform,
            )
        else:
            X = sample

        # Do not call model on timestep = 0 or sigma = 0
        if any(abs(v) < epsilon for v in frac_sc):
            derivatives.append(model_transform.backward(sample, X, S0, S1, schedule.sigma_transform))
        else:
            derivatives.append(model(X, *frac_sc))

    return tuple(  # pyright: ignore [reportReturnType]
        model_transform.forward(
            sample,
            math.sumprod(derivatives, w),  # pyright: ignore [reportArgumentType]
            S0,
            S1,
            schedule.sigma_transform,
        )
        for w in weights
    )


@dataclasses.dataclass(frozen=True)
class FunctionalSampler(ABC):
    def merge_noise[T: Sample](self, sample: T, noise: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return common.merge_noise(sample, noise, sigma, sigma_transform)

    @abstractmethod
    def sample_model[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
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
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
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
                schedule.ipoint((include.start or 0) / steps)[1],
                schedule.sigma_transform,
            ) / self.merge_noise(0.0, 1.0, schedule.ipoint(0)[1], schedule.sigma_transform)
            # Rescale sample by initial sigma. Mostly just to handle quirks with Scaled

        return self.sample_model(sample, model, model_transform, schedule, steps, include, rng, callback)


@dataclasses.dataclass(frozen=True)
class FunctionalHigher(FunctionalSampler):
    order: int = 2

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
    derivative_transform: models.DiffusionModel | None = models.DataModel()  # noqa: RUF009 # is immutable
    "Transform model to this space when computing higher order samples."


@dataclasses.dataclass(frozen=True)
class FunctionalSinglestep(FunctionalSampler):
    @abstractmethod
    def step[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
        step: Step,
        rng: RNG[T] | None = None,
    ) -> T: ...

    def sample_model[T: Sample](
        self,
        sample: T,
        model: SampleableModel[T],
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
        steps: int,
        include: slice = slice(None),
        rng: RNG[T] | None = None,
        callback: SampleCallback | None = None,
    ) -> T:
        for n in list(range(steps))[include]:
            sample = self.step(sample, model, model_transform, schedule, Step.from_int(n, steps), rng)

            if callback:
                callback(sample, n, *schedule.ipoint(n / steps))

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

    providers: DictOrProxy[int, tableaux.TableauProvider[tableaux.Tableau | tableaux.ExtendedTableau]] = (
        MappingProxyType(
            {
                2: tableaux.RK2.Heun,
                3: tableaux.RK3.Ralston,
                4: tableaux.RK4.Ralston,
                5: tableaux.RKE5.CashKarp,
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
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
        step: Step,
        rng: RNG[T] | None = None,
    ) -> T:
        return step_tableau(
            self.tableau(),
            sample,
            model,
            model_transform,
            schedule,
            step,
            derivative_transform=self.derivative_transform,
        )[0]


@dataclasses.dataclass(frozen=True)
class FastHeun(FunctionalAdaptive, FunctionalSinglestep, FunctionalHigher):
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
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
        step: Step,
        rng: RNG[T] | None = None,
    ) -> T:
        [time_t, sig_t], [time_s, sig_s] = schedule.ipoints(step).tolist()

        k1 = model(sample, time_t, sig_t)
        result: T = model_transform.forward(sample, k1, sig_t, sig_s, schedule.sigma_transform)

        # Multiplying by step size here is kind of an asspull, but so is this whole solver so...
        if time_s > 0 and self.evaluator(sample, result) / max(self.evaluator(0, result), 1e-16) > self.threshold * abs(
            sig_t - sig_s
        ):
            k2: T = (k1 + model(result, time_s, sig_s)) / 2  # pyright: ignore [reportAssignmentType]
            result: T = model_transform.forward(sample, k2, sig_t, sig_s, schedule.sigma_transform)

        return result


@dataclasses.dataclass(frozen=True)
class RKMoire(FunctionalAdaptive, FunctionalDerivative):
    providers: DictOrProxy[int, tableaux.TableauProvider[tableaux.ExtendedTableau]] = MappingProxyType(
        {
            2: tableaux.RKE2.Heun,
            3: tableaux.RKE3.BogackiShampine,
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
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
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

        indices: list[int] = list(range(steps))[include]
        step: int = indices[0]

        while step <= indices[-1]:
            step_next = min(step + step_size, indices[-1] + 1)

            if step_next < steps:
                sample_high, sample_low = step_tableau(
                    tab,
                    sample,
                    model,
                    model_transform,
                    schedule,
                    Step(step / steps, step_next / steps),
                    self.derivative_transform,
                )

                [sigma0, sigma1, sigma2] = schedule.ipoints(
                    [step / steps, step_next / steps, (step_next + step_size) / steps]
                )[:, 1].tolist()

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
                    schedule,
                    Step(step / steps, 1),
                    self.derivative_transform,
                )[0]

            sample = sample_high

            if callback:
                callback(sample, step_next - 1, *schedule.ipoint(step / steps))

            step = step_next

        return sample
