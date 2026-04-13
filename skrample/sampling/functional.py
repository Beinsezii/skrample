import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from skrample import common, scheduling
from skrample.common import RNG, DeltaPoint, Sample, Step

from . import models, tableaux, traits

type SampleCallback[T: Sample] = Callable[[T, int, DeltaPoint], Any]
"Return is ignored"
type SampleableModel[T: Sample] = Callable[[T, float, float, float], T]
"sample, timestep, sigma, alpha"

DEFAULT_PROVIDERS: Mapping[int, tableaux.TableauProvider[tableaux.TableauType]] = {
    1: tableaux.RK1.Euler,
    2: tableaux.RK2.Mid,
    3: tableaux.RK2.EES5_MIN,
    4: tableaux.RK2.EES7_MIN,
    5: tableaux.SSP.RK4_5,
    6: tableaux.RKE5.CashKarp,
    7: tableaux.RKZ.Butcher6,
    8: tableaux.SSP.RK3_8,
    10: tableaux.SSP.RK5_10,
    11: tableaux.RKZ.CV8,
    15: tableaux.RKZ.Stepanov10,
}
"""Default RK tableau providers.
Optimized for latent diffusion models.
The indexes are based on number of stages, NOT mathematical order."""
STABLE_PROVIDERS: Mapping[int, tableaux.TableauProvider[tableaux.TableauType]] = {
    2: tableaux.RKE2.Heun,
    3: tableaux.SSP.RK3_3,
    4: tableaux.RKE3.SSPRK3_4,
    5: tableaux.SSP.RK3_5,
    6: tableaux.SSP.RK3_6,
    7: tableaux.SSP.RK3_7,
}
"""SSP RK providers.
Prioritizes stability.
The indexes are based on number of stages, NOT mathematical order."""

DEFAULT_EMBEDDED_PROVIDERS: Mapping[int, tableaux.TableauProvider[tableaux.EmbeddedTableau]] = {
    2: tableaux.RKE2.Heun,
    4: tableaux.RKE3.BogackiShampine,
    6: tableaux.RKE5.Fehlberg,
}
"""Default RK embedded tableau providers.
The indexes are based on number of stages, NOT mathematical order."""


def step_tableau[T: Sample](
    tableau: tableaux.Tableau | tableaux.EmbeddedTableau,
    sample: T,
    model: SampleableModel[T],
    model_transform: models.DiffusionModel,
    schedule: scheduling.SkrampleSchedule,
    step: Step,
    derivative_transform: models.DiffusionModel | None = None,
    noise: T | None = None,
    stochasticity: float = 0,
    epsilon: float = 1e-8,
) -> tuple[T, ...]:
    nodes, weights = tableau[0], tableau[1:]

    if derivative_transform:
        model = models.ModelConvert(
            model_transform,
            derivative_transform,
        ).wrap_model_call(model)
        model_transform = derivative_transform

    derivatives: list[T] = []
    S0, S1, *fractions = schedule.ipoints([*step, *(step[0] + f[0] * (step[1] - step[0]) for f in nodes)])
    delta = common.DeltaPoint(S0, S1)

    for frac_sc, icoeffs in zip(fractions, (t[1] for t in nodes), strict=True):
        if icoeffs:
            X: T = model_transform.forward(  # type: ignore # sumprod is T
                sample,
                math.sumprod(derivatives, icoeffs) / math.fsum(icoeffs),  # type: ignore # sumprod is T
                common.DeltaPoint(delta.point_from, frac_sc),
            )
        else:
            X = sample

        # Do not call model on timestep = 0 or sigma = 0
        if abs(frac_sc.timestep) < epsilon or abs(frac_sc.sigma) < epsilon:
            derivatives.append(model_transform.backward(sample, X, delta))
        else:
            derivatives.append(model(X, *frac_sc))

    return tuple(  # type: ignore # sumprod is T
        model_transform.forward(
            sample,
            math.sumprod(derivatives, w),  # type: ignore # sumprod is T
            delta,
            noise,
            stochasticity,
        )
        for w in weights
    )


@dataclasses.dataclass(frozen=True)
class FunctionalSampler(ABC, traits.SamplingCommon):
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
    ) -> T:
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
            sample: T = rng(None)
        else:
            sample: T = self.add_noise(  # type: ignore # 0 should be valid here because it's added to RNG so it becomes T
                0 if initial is None else initial,
                rng(None),
                schedule.ipoint((include.start or 0) / steps),
            ) / self.add_noise(0.0, 1.0, schedule.point_1)
            # Rescale sample by initial sigma. Mostly just to handle quirks with Scaled

        return self.sample_model(sample, model, model_transform, schedule, steps, include, rng, callback)


@dataclasses.dataclass(frozen=True)
class FunctionalHigher(traits.HigherOrder, FunctionalSampler):
    def adjust_steps(self, steps: int) -> int:
        "Adjust the steps to approximate an equal amount of model calls"
        return round(steps / self.order)


@dataclasses.dataclass(frozen=True)
class FunctionalUnified(traits.UnifiedModelling, FunctionalHigher): ...


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
            step = Step.from_int(n, steps)
            sample = self.step(sample, model, model_transform, schedule, step, rng)

            if callback:
                callback(sample, n, schedule.istep(step))

        return sample


@dataclasses.dataclass(frozen=True)
class FunctionalAdaptive(FunctionalSampler):
    type Evaluator[T: Sample] = Callable[[T, T], float]

    @staticmethod
    def mse[T: Sample](a: T, b: T) -> float:
        error: T = abs(a - b) ** 2  # pyright: ignore [reportAssignmentType] # float rhs is always T
        return common.mean(error)

    evaluator: Evaluator = mse
    "Function used to measure error of two samples"
    threshold: float = 1e-2
    "Target error threshold for a given evaluation"


@dataclasses.dataclass(frozen=True)
class RKUltra(FunctionalUnified, FunctionalSinglestep):
    "Implements almost every single method from https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods"  # noqa: RUF002

    providers: Mapping[int, tableaux.TableauProvider[tableaux.Tableau | tableaux.EmbeddedTableau]] = MappingProxyType(
        DEFAULT_PROVIDERS
    )
    """Tableau providers for a given order.
    Note the mapping can be arbitrary"""

    @staticmethod
    def max_order() -> int:
        return 99

    def tableau(self, order: int | None = None) -> tableaux.Tableau:
        if order is None:
            order = self.order

        if order >= min(self.providers.keys()) and (morder := max(o for o in self.providers.keys() if o <= order)):
            return tableaux.Tableau(self.providers[morder].tableau().stages, self.providers[morder].tableau().weights)
        else:
            return tableaux.RK1.Euler.value

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
            self.derivative_transform,
            rng(step) if rng else None,
            self.stochasticity,
        )[0]


@dataclasses.dataclass(frozen=True)
class DynasauRK(FunctionalUnified, FunctionalSinglestep):
    """Dynamic RK solver that generates tableaux on-the-fly.
    Initializes with a high stability method and moves towards
    higher convergence methods through exponential decay.
    This means that the overall stability properties change with step count,
    with higher steps using lower mean stability.
    The exact formulation is e^-st * e^-ST where
    s == per_step_decay    t == current NFEs
    S == total_step_decay  T == total NFEs"""

    per_step_decay: float = math.log(0.5) / -2  # 50% every 2 nfes
    """Exponential decay factor over each successive step.
    Negative values become exponential growth."""
    total_step_decay: float = math.log(0.5) / -20  # 50% every nfes
    """Exponential decay factor for total step count
    Negative values become exponential growth."""
    invert: bool = False
    "Invert the final gradient"

    @staticmethod
    def min_order() -> int:
        return 2

    @staticmethod
    def max_order() -> int:
        return 4

    def adjust_steps(self, steps: int) -> int:
        return max(round(steps / self.order), 1)

    def gradient(self, step: Step, stages: int) -> float:
        "Gradient from most stable (1.0) to most convergent (0.0) methods"
        step = step.normal().clamp()

        # e^-stn * e^-STn == e^(-st - ST)n
        gradient = math.exp((-self.total_step_decay * step.amount() - self.per_step_decay * step.position()) * stages)

        return abs(self.invert - min(max(gradient, 0), 1))

    def tableau(self, step: Step) -> tableaux.Tableau:
        "It's assumed that step sizes are uniform, ie in a for loop."
        if self.order >= 4:
            high: float = 1 / 4 * (2 - math.sqrt(2))  # SYM
            low: float = 1 / 14 * (5 - 3 * math.sqrt(2))  # MIN
            tf = tableaux.providers.ees27_tableau
        elif self.order >= 3:
            high: float = 0.25  # SYM
            low: float = 0.1  # MIN
            tf = tableaux.providers.ees25_tableau
        else:
            high: float = 1  # Heun
            low: float = 0.5  # Mid
            tf = tableaux.providers.rk2_tableau

        gradient = self.gradient(step, len(tf((high + low) / 2).stages))

        return tf(gradient * high + (1 - gradient) * low)

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
            self.tableau(step),
            sample,
            model,
            model_transform,
            schedule,
            step,
            self.derivative_transform,
            rng(step) if rng else None,
            self.stochasticity,
        )[0]


@dataclasses.dataclass(frozen=True)
class RKMoire(traits.DerivativeTransform, FunctionalAdaptive, FunctionalHigher):
    providers: Mapping[int, tableaux.TableauProvider[tableaux.EmbeddedTableau]] = MappingProxyType(
        DEFAULT_EMBEDDED_PROVIDERS
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

    def tableau(self, order: int | None = None) -> tableaux.EmbeddedTableau:
        if order is None:
            order = self.order

        if order >= min(self.providers.keys()) and (morder := max(o for o in self.providers.keys() if o <= order)):
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

                [sigma0, sigma1, sigma2] = schedule.ipoints_np(
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
                    tab.unembed(),
                    sample,
                    model,
                    model_transform,
                    schedule,
                    Step(step / steps, 1),
                    self.derivative_transform,
                )[0]

            sample = sample_high

            if callback:
                callback(sample, step_next - 1, schedule.istep(Step.from_int(step, steps)))

            step = step_next

        return sample
