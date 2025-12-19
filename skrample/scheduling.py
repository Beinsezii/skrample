import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from skrample.common import FloatSchedule, SigmaTransform, normalize, regularize, sigma_complement, sigma_polar, sigmoid

type NPSchedule = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]
"[sequence..., timestep:sigma]"

type NPSequence = np.ndarray[tuple[int], np.dtype[np.float64]]


@functools.lru_cache
def np_schedule_lru(schedule: "SkrampleSchedule", steps: int) -> NPSchedule:
    """Globally cached function for SkrampleSchedule.schedule(steps).
    Prefer moving SkrampleScheudle.schedule() outside of any loops if possible."""
    return schedule.schedule_np(steps)


@functools.lru_cache
def schedule_lru(schedule: "SkrampleSchedule", steps: int) -> FloatSchedule:
    """Globally cached function for SkrampleSchedule.schedule(steps).
    Prefer moving SkrampleScheudle.schedule() outside of any loops if possible."""
    return tuple(map(tuple, np_schedule_lru(schedule, steps).tolist()))  # pyright: ignore [reportReturnType] # Size indeterminate???


@dataclass(frozen=True)
class SkrampleSchedule(ABC):
    "Abstract class defining the bare minimum for a noise schedule"

    @property
    @abstractmethod
    def sigma_transform(self) -> SigmaTransform:
        "SigmaTransform required for a given noise schedule"

    @abstractmethod
    def _points(self, t: NPSequence) -> NPSchedule:
        """Core implementation of the continuously variable schedule.
        0.0 is more noise, 1.0 is no noise."""

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        """Sample the schedule along T points in time.
        0.0 is more noise, 1.0 is no noise."""
        return self._points(np.asarray(t, dtype=np.float64).clip(0, 1))

    def ipoints(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        """Inverse of `points`, or `inference` points.
        0.0 is more noise, 1.0 is no noise."""
        return self._points(1 - np.asarray(t, dtype=np.float64).clip(0, 1))

    def point(self, t: float) -> tuple[float, float]:
        """Sample the schedule at T point in time.
        0.0 is more noise, 1.0 is no noise."""
        return tuple(self._points(np.expand_dims(np.float64(t).clip(0, 1), 0))[0].tolist())

    def ipoint(self, t: float) -> tuple[float, float]:
        """Inverse of `point`, or `inference` points.
        0.0 is more noise, 1.0 is no noise."""
        return tuple(self._points(np.expand_dims(1 - np.float64(t).clip(0, 1), 0))[0].tolist())

    def schedule_np(self, steps: int) -> NPSchedule:
        """Return the full noise schedule, timesteps stacked on top of sigmas.
        Excludes the trailing zero"""
        return self._points(np.linspace(1, 0, steps, endpoint=False))

    def timesteps_np(self, steps: int) -> NPSequence:
        "Just the timesteps component as a 1-d array"
        return self.schedule_np(steps)[:, 0]

    def sigmas_np(self, steps: int) -> NPSequence:
        "Just the sigmas component as a 1-d array"
        return self.schedule_np(steps)[:, 1]

    def schedule(self, steps: int) -> FloatSchedule:
        """Return the full noise schedule, [(timestep, sigma), ...)
        Excludes the trailing zero"""
        return tuple(map(tuple, self.schedule_np(steps).tolist()))  # pyright: ignore [reportReturnType] # Size indeterminate???

    def timesteps(self, steps: int) -> Sequence[float]:
        "Just the timesteps component"
        return self.timesteps_np(steps).tolist()

    def sigmas(self, steps: int) -> Sequence[float]:
        "Just the sigmas component"
        return self.sigmas_np(steps).tolist()


@dataclass(frozen=True)
class ScheduleCommon(SkrampleSchedule):
    "Common attributes for base schedules"

    base_timesteps: int = 1000
    "Original timesteps the model was trained on"

    @functools.cached_property
    def all_points(self) -> NPSchedule:
        "Returns all points over `base_timesteps` with LRU cache"
        return self.points(np.linspace(0, 1, self.base_timesteps))

    @abstractmethod
    def _sigmas_to_points(self, sigmas: NPSequence) -> NPSchedule:
        pass

    def sigmas_to_points(self, sigmas: Sequence[float] | NPSequence) -> NPSchedule:
        return self._sigmas_to_points(np.asarray(sigmas, dtype=np.float64))


@dataclass(frozen=True)
class FixedSchedule(SkrampleSchedule):
    fixed_schedule: FloatSchedule | NPSchedule
    transform: SigmaTransform

    def _points(self, t: NPSequence) -> NPSchedule:
        return np.quantile(np.concatenate([np.asarray(self.fixed_schedule, dtype=np.float64), [[0, 0]]]), t, axis=0)

    @property
    def sigma_transform(self) -> SigmaTransform:
        return self.transform


@dataclass(frozen=True)
class Scaled(ScheduleCommon):
    "Standard noise schedule for Stable Diffusion and derivatives"

    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_scale: float = 2

    @property
    def sigma_transform(self) -> SigmaTransform:
        return sigma_polar

    def continuous_alphas_cumprod(self, t: NPSequence) -> NPSequence:
        # Continuous version of
        # betas = (
        #     np.linspace(
        #         self.beta_start ** (1 / self.beta_scale),
        #         self.beta_end ** (1 / self.beta_scale),
        #         self.base_timesteps,
        #         dtype=np.float64,
        #     )
        #     ** self.beta_scale
        # )
        # alphas_cumprod = np.cumprod(1 - betas, dtype=np.float64)
        #
        # Auto-generated by Qwen 235B VL

        k = self.beta_scale
        T = self.base_timesteps
        root_start = self.beta_start ** (1 / k)
        root_end = self.beta_end ** (1 / k)
        slope = root_end - root_start

        if abs(slope) < 1e-8:
            # Handle constant beta case
            beta_val = root_start**k
            integral_beta = beta_val * t
            integral_beta2 = (beta_val**2) * t
        else:
            # Integral of β(u) = (root_start + slope * u)^k from 0 to t
            integral_beta = ((root_start + slope * t) ** (k + 1) - root_start ** (k + 1)) / (slope * (k + 1))

            # Integral of β(u)^2 = (root_start + slope * u)^(2k) from 0 to t
            integral_beta2 = ((root_start + slope * t) ** (2 * k + 1) - root_start ** (2 * k + 1)) / (
                slope * (2 * k + 1)
            )

        # Combine first and second-order terms
        exponent = T * (integral_beta + integral_beta2 / 2)
        return np.exp(-exponent)

    def _points(self, t: NPSequence) -> NPSchedule:
        alphas_cumprod = self.continuous_alphas_cumprod(t)
        sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        return np.stack([t * self.base_timesteps, sigmas], 1)

    def _sigmas_to_points(self, sigmas: NPSequence) -> NPSchedule:
        # TODO (beinsezii): continuous version instead of LRU + interp?
        timesteps = np.interp(sigmas, self.all_points[:, 1], self.all_points[:, 0])
        return np.stack([timesteps, sigmas], axis=1)


@dataclass(frozen=True)
class ZSNR(Scaled):
    "Zero Terminal SNR schedule from https://arxiv.org/abs/2305.08891"

    # Just some funny number I made up when working on the diffusers PR that worked well. F32 smallest subnormal
    epsilon: float = 2**-24
    "Amount to shift the zero value by to keep calculations finite."

    def continuous_alphas_cumprod(self, t: NPSequence) -> NPSequence:
        ### from https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        alphas_bar_sqrt = np.sqrt(super().continuous_alphas_cumprod(np.concatenate([[0], t, [1]])))

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].item()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].item()

        # Strip 0, T
        alphas_bar_sqrt = alphas_bar_sqrt[1:-1]

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_cumprod = alphas_bar_sqrt**2  # Revert sqrt

        alphas_cumprod = np.maximum(self.epsilon, alphas_cumprod)  # Epsilon to avoid inf
        return alphas_cumprod


@dataclass(frozen=True)
class Linear(ScheduleCommon):
    "Simple linear schedule, from sigma_start...0"

    sigma_start: float = 1
    "Maximum (first) sigma value"

    custom_transform: SigmaTransform | None = None
    """If set, will be used for `self.sigma_transform`
    Otherwise, uses `sigma_polar` for sigma_start > 1 and sigma_complement for <= 1"""

    @property
    def sigma_transform(self) -> SigmaTransform:
        if self.custom_transform is None:
            return sigma_complement if self.sigma_start <= 1 else sigma_polar
        else:
            return self.custom_transform

    def _points(self, t: NPSequence) -> NPSchedule:
        return np.stack([t * self.base_timesteps, t * self.sigma_start], axis=1)

    def _sigmas_to_points(self, sigmas: NPSequence) -> NPSchedule:
        return np.stack([sigmas * (self.base_timesteps / self.sigma_start), sigmas], axis=1)


@dataclass(frozen=True)
class SigmoidCDF(Linear):
    """Normal cumulative distribution run through sigmoid.
    Produces an S-curve similar to the Beta modifier.
    This is the continuous equivalent of `np.sort(np.randn([steps]))` used in some training schedules"""

    cdf_scale: float = 3
    "Multiply the inverse CDF output before the sigmoid function is applied"

    def _points(self, t: NPSequence) -> NPSchedule:
        from scipy.stats import norm

        # Always include 1.0 for post-sigmoid normalize
        t = np.concatenate([[1], t])
        # 1.0 is invalid
        probabilities = regularize(t, 1 - 1e-8, 0)
        sigmas = sigmoid(norm.ppf(probabilities, scale=self.cdf_scale))
        sigmas = normalize(sigmas, sigmas[0])
        sigmas = sigmas[1:]

        return np.stack([sigmas * self.base_timesteps, sigmas * self.sigma_start], axis=1)


@dataclass(frozen=True)
class ScheduleModifier(SkrampleSchedule):
    """Generic class for schedules that modify other schedules.
    Unless otherwise specified, uses base schedule properties"""

    base: "ScheduleCommon | ScheduleModifier"
    "Schedule that this one will modify"

    @property
    def base_timesteps(self) -> int:
        return self.base.base_timesteps

    @property
    def sigma_transform(self) -> SigmaTransform:
        return self.base.sigma_transform

    @property
    def all_split(self) -> tuple[list["ScheduleModifier"], ScheduleCommon]:
        """All SkrampleModifiers recursively, including self.
        Separated for type safety."""
        bases: list[ScheduleModifier] = [self]
        last = self.base
        while isinstance(last, ScheduleModifier):
            bases.append(last)
            last = last.base

        return (bases, last)

    @property
    def all(self) -> list["ScheduleCommon | ScheduleModifier"]:
        "All SkrampleModifiers recursively, including self"
        mods, base = self.all_split
        return [*mods, base]

    @property
    def lowest(self) -> ScheduleCommon:
        "The basemost schedule of all modifiers"
        return self.all_split[1]

    @staticmethod
    def stack(modifiers: list["ScheduleModifier"], base: ScheduleCommon) -> "ScheduleModifier | ScheduleCommon":
        """Re-stacks the given modifiers, setting each `base` to the next modifier in the list before the true base.
        Inverse of ScheduleModifier.all_split"""
        last = base
        for mod in reversed(modifiers):
            last = replace(mod, base=last)
        return last

    def _sigmas_to_points(self, sigmas: NPSequence) -> NPSchedule:
        return self.base._sigmas_to_points(sigmas)

    def sigmas_to_points(self, sigmas: Sequence[float] | NPSequence) -> NPSchedule:
        return self._sigmas_to_points(np.asarray(sigmas, dtype=np.float64))

    def find[T: "ScheduleModifier"](self, skrample_schedule: type[T], exact: bool = False) -> T | None:
        """Find the first schedule of type T recursively in the modifier tree.
        If `exact` is True, requires an exact type match instead of any subclass."""
        for schedule in self.all_split[0]:
            if type(schedule) is skrample_schedule or (not exact and isinstance(schedule, skrample_schedule)):
                return schedule

    def find_split[T: "ScheduleModifier"](
        self,
        skrample_schedule: type[T],
        exact: bool = False,
    ) -> tuple[list["ScheduleModifier"], T, list["ScheduleModifier"], ScheduleCommon] | None:
        """Split version of ScheduleModifier.find().
        Modifiers are separated into before, found, after"""

        mods, base = self.all_split
        found: T | None = None
        before = []
        after = []

        for schedule in mods:
            if type(schedule) is skrample_schedule or (not exact and isinstance(schedule, skrample_schedule)):
                found = schedule
            elif found is None:
                before.append(schedule)
            else:
                after.append(schedule)

        if found:
            return (before, found, after, base)


@dataclass(frozen=True)
class NoMod(ScheduleModifier):
    "Does nothing. For generic programming against ScheduleModifier"

    def _points(self, t: NPSequence) -> NPSchedule:
        return self.base._points(t)


@dataclass(frozen=True)
class FlowShift(ScheduleModifier):
    shift: float = 3.0
    """Amount to shift noise schedule by."""

    def _points(self, t: NPSequence) -> NPSchedule:
        mask = t > 0
        t[mask] = self.shift / (self.shift + (1 / t[mask] - 1))
        return self.base._points(t)


@dataclass(frozen=True)
class Karras(ScheduleModifier):
    "Similar to Exponential, intended for 1st generation Stable Diffusion models"

    rho: float = 7.0
    "Ramp power"

    steps: float = 20
    "Steps for computing the scale values."

    def _points(self, t: NPSequence) -> NPSchedule:
        sigma_min, sigma_max = self.base.points([1 / self.steps, 1.0])[:, 1].tolist()

        t = np.concatenate([[1, 0], t])

        sigmas = ((sigma_min ** (1.0 / self.rho)) * (1 - t) + (sigma_max ** (1.0 / self.rho)) * t) ** self.rho

        sigmas = normalize(sigmas[2:], sigmas[0], sigmas[1]) * sigma_max

        return self.base._sigmas_to_points(sigmas)


@dataclass(frozen=True)
class Exponential(ScheduleModifier):
    "Also known as 'polyexponential' when rho != 1"

    rho: float = 1.0
    "Ramp power"

    steps: float = 20
    "Steps for computing the scale values."

    def _points(self, t: NPSequence) -> NPSchedule:
        sigma_min, sigma_max = self.base.points([1 / self.steps, 1.0])[:, 1].tolist()

        t = np.concatenate([[1, 0], t]) ** self.rho

        sigmas = np.exp(np.log(sigma_min) * (1 - t) + np.log(sigma_max) * t)

        sigmas = normalize(sigmas[2:], sigmas[0], sigmas[1]) * sigma_max

        return self.base._sigmas_to_points(sigmas)


@dataclass(frozen=True)
class Beta(ScheduleModifier):
    """Beta continuous distribtuion function. A sort of S-curve.
    https://arxiv.org/abs/2407.12173"""

    alpha: float = 0.6
    beta: float = 0.6

    def _points(self, t: NPSequence) -> NPSchedule:
        from scipy.stats import beta

        sigma_max = self.base.point(1)[1]
        # Always include 1.0 for post-ppf normalize
        probabilities = np.concatenate([[1], t])

        sigmas = beta.ppf(probabilities, self.alpha, self.beta)
        sigmas = normalize(sigmas, sigmas[0])[1:]
        return self.base._sigmas_to_points(sigmas * sigma_max)


@dataclass(frozen=True)
class Hyper(ScheduleModifier):
    "Hyperbolic curve modifier"

    scale: float = 2
    """Sharpness of curve.
    Mathematically this is tanh for positive and sinh negative"""

    tail: bool = True
    "Include the trailing end to make an S curve"

    def _points(self, t: NPSequence) -> NPSchedule:
        if abs(self.scale) <= 1e-8:
            return self.base._points(t)

        points = regularize(np.concatenate([[1], t]), self.scale, -self.scale * self.tail)  # 1..0 -> scale..-scale
        # WARN(beinsezii): sqrt(2) is more or less a magic number afaict
        points = np.sinh(points) if self.scale < 0 else np.tanh(points / math.sqrt(2))
        # don't use -1 because no endcaps
        points = normalize(points[1:], points[0], -points[0] * self.tail)  # hyper..-hyper -> 1..0

        return self.base._points(points)
