import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from skrample.common import FloatSchedule, SigmaTransform, normalize, regularize, sigma_complement, sigma_polar, sigmoid

type NPSchedule = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"[sequence..., timestep:sigma]"

type NPSequence = np.ndarray[tuple[int], np.dtype[np.float64]]


@lru_cache
def np_schedule_lru(schedule: "SkrampleSchedule", steps: int) -> NDArray[np.float64]:
    """Globally cached function for SkrampleSchedule.schedule(steps).
    Prefer moving SkrampleScheudle.schedule() outside of any loops if possible."""
    return schedule.schedule_np(steps)


@lru_cache
def schedule_lru(schedule: "SkrampleSchedule", steps: int) -> FloatSchedule:
    """Globally cached function for SkrampleSchedule.schedule(steps).
    Prefer moving SkrampleScheudle.schedule() outside of any loops if possible."""
    return tuple(map(tuple, np_schedule_lru(schedule, steps).tolist()))


@dataclass(frozen=True)
class SkrampleSchedule(ABC):
    "Abstract class defining the bare minimum for a noise schedule"

    @property
    @abstractmethod
    def sigma_transform(self) -> SigmaTransform:
        "SigmaTransform required for a given noise schedule"

    @abstractmethod
    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        """Return the full noise schedule, timesteps stacked on top of sigmas.
        Excludes the trailing zero"""

    def timesteps_np(self, steps: int) -> NDArray[np.float64]:
        "Just the timesteps component as a 1-d array"
        return self.schedule_np(steps)[:, 0]

    def sigmas_np(self, steps: int) -> NDArray[np.float64]:
        "Just the sigmas component as a 1-d array"
        return self.schedule_np(steps)[:, 1]

    def schedule(self, steps: int) -> FloatSchedule:
        """Return the full noise schedule, [(timestep, sigma), ...)
        Excludes the trailing zero"""
        return tuple(map(tuple, self.schedule_np(steps).tolist()))

    def timesteps(self, steps: int) -> Sequence[float]:
        "Just the timesteps component"
        return self.timesteps_np(steps).tolist()

    def sigmas(self, steps: int) -> Sequence[float]:
        "Just the sigmas component"
        return self.sigmas_np(steps).tolist()

    @abstractmethod
    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        """Sample the schedule along T points in time.
        1.0 is more noise, 0.0 is no noise."""

    def __call__(self, steps: int) -> NDArray[np.float64]:
        return self.schedule_np(steps)


@dataclass(frozen=True)
class ScheduleCommon(SkrampleSchedule):
    "Common attributes for base schedules"

    base_timesteps: int = 1000
    "Original timesteps the model was trained on"

    @abstractmethod
    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def sigmas_to_points(self, sigmas: Sequence[float] | NPSequence) -> NPSchedule:
        pass


@dataclass(frozen=True)
class Scaled(ScheduleCommon):
    "Standard noise schedule for Stable Diffusion and derivatives"

    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_scale: float = 2

    # Let's name this "uniform" instead of trailing since it basically just avoids the truncation.
    # Think that's what ComfyUI does
    uniform: bool = True
    """When this is false, the first timestep is effectively skipped,
    therefore it is recommended to only use this for backward compatibility.
    https://arxiv.org/abs/2305.08891"""

    @property
    def sigma_transform(self) -> SigmaTransform:
        return sigma_polar

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        # it uses full distribution pre-interp
        scaled_sigmas = self.scaled_sigmas()
        log_sigmas = np.log(scaled_sigmas)

        # below here just a copy of diffusers' _sigma_to_t

        # get log sigma
        log_sigma = np.log(np.maximum(sigmas, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=self.base_timesteps - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        return t

    def timesteps_np(self, steps: int) -> NDArray[np.float64]:
        # # https://arxiv.org/abs/2305.08891 Table 2
        if self.uniform:
            return np.linspace(self.base_timesteps - 1, 0, steps, endpoint=False, dtype=np.float64).round()
        else:
            # They use a truncated ratio for ...reasons?
            return np.flip(np.arange(0, steps, dtype=np.float64) * (self.base_timesteps // steps)).round()

    @lru_cache
    def betas(self) -> NDArray[np.float64]:
        return (
            np.linspace(
                self.beta_start ** (1 / self.beta_scale),
                self.beta_end ** (1 / self.beta_scale),
                self.base_timesteps,
                dtype=np.float64,
            )
            ** self.beta_scale
        )

    @lru_cache
    def alphas_cumprod(self) -> NDArray[np.float64]:
        return np.cumprod(1 - self.betas(), axis=0, dtype=np.float64)

    @lru_cache
    def scaled_sigmas(self) -> NDArray[np.float64]:
        alphas_cumprod = self.alphas_cumprod()
        return ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.scaled_sigmas()
        timesteps = self.timesteps_np(steps)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        return np.stack([timesteps, sigmas], axis=1)

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

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        t = np.asarray(t, dtype=np.float64)
        alphas_cumprod = self.continuous_alphas_cumprod(np.asarray(t, dtype=np.float64))
        sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        return np.stack([t * self.base_timesteps, sigmas], 1)

    def sigmas_to_points(self, sigmas: Sequence[float] | NPSequence) -> NPSchedule:
        # TODO (beinsezii): continuous version instead of LRU?
        return np.stack([self.sigmas_to_timesteps(np.asarray(sigmas, dtype=np.float64)), sigmas], axis=1)


@dataclass(frozen=True)
class ZSNR(Scaled):
    "Zero Terminal SNR schedule from https://arxiv.org/abs/2305.08891"

    # Just some funny number I made up when working on the diffusers PR that worked well. F32 smallest subnormal
    epsilon: float = 2**-24
    "Amount to shift the zero value by to keep calculations finite."

    uniform: bool = True
    "ZSNR should always uniform/trailing"

    def alphas_cumprod(self, betas: NDArray[np.float64]) -> NDArray[np.float64]:
        ### from https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        # Convert betas to alphas_bar_sqrt
        alphas_bar_sqrt = np.cumprod(1 - betas, axis=0) ** 0.5

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].item()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].item()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_cumprod = alphas_bar_sqrt**2  # Revert sqrt

        alphas_cumprod[-1] = self.epsilon  # Epsilon to avoid inf
        return alphas_cumprod

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

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        return normalize(sigmas, self.sigma_start) * self.base_timesteps

    def sigmas_np(self, steps: int) -> NDArray[np.float64]:
        return np.linspace(self.sigma_start, 0, steps, endpoint=False, dtype=np.float64)

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.sigmas_np(steps)
        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps, sigmas], axis=1)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        t = np.asarray(t, dtype=np.float64)
        return np.stack([t * self.base_timesteps, t * self.sigma_start], axis=1)

    def sigmas_to_points(self, sigmas: Sequence[float] | NPSequence) -> NPSchedule:
        sigmas = np.asarray(sigmas, dtype=np.float64)
        return np.stack([sigmas * (self.base_timesteps / self.sigma_start), sigmas], axis=1)


@dataclass(frozen=True)
class SigmoidCDF(Linear):
    """Normal cumulative distribution run through sigmoid.
    Produces an S-curve similar to the Beta modifier.
    This is the continuous equivalent of `np.sort(np.randn([steps]))` used in some training schedules"""

    cdf_scale: float = 3
    "Multiply the inverse CDF output before the sigmoid function is applied"

    def sigmas_np(self, steps: int) -> NDArray[np.float64]:
        from scipy.stats import norm

        step_peak = 1 / (steps * math.pi / 2)
        probabilities = np.linspace(1 - step_peak, step_peak, steps, dtype=np.float64)
        sigmas = sigmoid(norm.ppf(probabilities) * self.cdf_scale)
        return regularize(sigmas / sigmas.max(), self.sigma_start)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        from scipy.stats import norm

        # Always include 1.0 for post-sigmoid normalize
        t = np.concatenate([[1], np.asarray(t, dtype=np.float64)])
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

    def sigmas_to_timesteps(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.base.sigmas_to_timesteps(sigmas)

    def sigmas_to_points(self, sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.base.sigmas_to_points(sigmas)

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

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        return self.base.schedule_np(steps)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        return self.base.points(t)


@dataclass(frozen=True)
class FlowShift(ScheduleModifier):
    shift: float = 3.0
    """Amount to shift noise schedule by."""

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.base.sigmas_np(steps)

        start = sigmas.max().item()
        sigmas = self.shift / (self.shift + (start / sigmas - 1)) * start

        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps.flatten(), sigmas], axis=1)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        # t = np.asarray(t)
        # mask = t > 0
        # t[mask] = self.shift / (self.shift + (1 / t[mask] - 1))
        # return self.base.points(t)

        sigmas = self.base.points(t)[:, 1]
        start = self.base.points([1])[0, 1].item()
        mask = sigmas > 0
        sigmas[mask] = self.shift / (self.shift + (start / sigmas[mask] - 1)) * start
        return self.base.sigmas_to_points(sigmas)

        t = np.asarray(t)
        mask = t > 0
        t[mask] = self.shift / (self.shift + (1 / t[mask] - 1))
        return self.base.points(t)


@dataclass(frozen=True)
class Karras(ScheduleModifier):
    "Similar to Exponential, intended for 1st generation Stable Diffusion models"

    rho: float = 7.0
    "Ramp power"

    floor: float = 1 / 20
    "Floor for computing the scale values. Equivalent to 1 / steps for the classic formula."

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.base.sigmas_np(steps)

        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()

        sigmas = np.linspace(sigma_max ** (1 / self.rho), sigma_min ** (1 / self.rho), steps) ** self.rho

        timesteps = self.sigmas_to_timesteps(sigmas)

        return np.stack([timesteps.flatten(), sigmas], axis=1)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        t = np.asarray(t, dtype=np.float64)
        sigma_max = self.base.points([1])[0, 1].item()
        sigmas = ((t + 1) ** self.rho / 2**self.rho) * sigma_max - sigma_max * 2**-self.rho

        sigma_min, sigma_max = self.base.points([self.floor, 1.0])[:, 1].tolist()

        sigmas = ((sigma_min ** (1.0 / self.rho)) * (1 - t) + (sigma_max ** (1.0 / self.rho)) * t) ** self.rho

        sigmas = regularize(normalize(sigmas, sigma_max, sigma_min), sigma_max)

        return self.base.sigmas_to_points(sigmas)


@dataclass(frozen=True)
class Exponential(ScheduleModifier):
    "Also known as 'polyexponential' when rho != 1"

    rho: float = 1.0
    "Ramp power"

    floor: float = 1 / 20
    "Floor for computing the scale values. Equivalent to 1 / steps for the classic formula."

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        sigmas = self.base.sigmas_np(steps)
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()

        ramp = np.linspace(1, 0, steps, dtype=np.float64) ** self.rho
        sigmas = np.exp(regularize(ramp, math.log(sigma_max), math.log(sigma_min)))

        timesteps = self.sigmas_to_timesteps(sigmas)
        return np.stack([timesteps, sigmas], axis=1)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        t = np.asarray(t, dtype=np.float64)
        sigma_max = self.base.points([1])[0, 1].item()
        sigmas = ((t + 1) ** self.rho / 2**self.rho) * sigma_max - sigma_max * 2**-self.rho

        sigma_min, sigma_max = self.base.points([self.floor, 1.0])[:, 1].tolist()

        t = t**self.rho
        sigmas = np.exp(np.log(sigma_min) * (1 - t) + np.log(sigma_max) * t)

        sigmas = regularize(normalize(sigmas, sigma_max, sigma_min), sigma_max)

        return self.base.sigmas_to_points(sigmas)


@dataclass(frozen=True)
class Beta(ScheduleModifier):
    """Beta continuous distribtuion function. A sort of S-curve.
    https://arxiv.org/abs/2407.12173"""

    alpha: float = 0.6
    beta: float = 0.6

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        import scipy

        sigmas = self.base.sigmas_np(steps)
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()

        # WARN(beinsezii): I think this should be endpiont=False and end=0 but I'm not gonna fight diffusers
        pparr = scipy.stats.beta.ppf(np.linspace(1, 0, steps, dtype=np.float64), self.alpha, self.beta)
        sigmas = regularize(pparr, sigma_max, sigma_min)

        timesteps = self.sigmas_to_timesteps(sigmas)
        return np.stack([timesteps.flatten(), sigmas], axis=1)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        from scipy.stats import beta

        sigma_max = self.base.points([1])[0, 1].item()
        # Always include 1.0 for post-ppf normalize
        probabilities = np.concatenate([[1], np.asarray(t, dtype=np.float64)])

        sigmas = beta.ppf(probabilities, self.alpha, self.beta)
        sigmas = normalize(sigmas, sigmas[0])[1:]
        return self.base.sigmas_to_points(sigmas * sigma_max)


@dataclass(frozen=True)
class Hyper(ScheduleModifier):
    "Hyperbolic curve modifier"

    scale: float = 2
    """Sharpness of curve.
    Mathematically this is tanh for positive and sinh negative"""

    tail: bool = True
    "Include the trailing end to make an S curve"

    use_noise: bool = False
    """Use noise scale instead of time scale.
    Does not affect Linaer() and derivatives where these are equivalent"""

    def schedule_np(self, steps: int) -> NDArray[np.float64]:
        if abs(self.scale) <= 1e-8:
            return self.base.schedule_np(steps)

        sigmas = self.base.sigmas_np(steps)
        start = sigmas[0].item()

        sigmas = normalize(sigmas, start)  # Base -> 1..0
        sigmas = regularize(sigmas, self.scale, -self.scale * self.tail)  # 1..0 -> scale..-scale
        # WARN(beinsezii): sqrt(2) is more or less a magic number afaict
        sigmas = np.sinh(sigmas) if self.scale < 0 else np.tanh(sigmas / math.sqrt(2))
        # don't use -1 because no endcaps
        sigmas = normalize(sigmas, sigmas[0], -sigmas[0] * self.tail)  # hyper..-hyper -> 1..0
        sigmas = regularize(sigmas, start)  # 1..0 -> Base

        timesteps = self.sigmas_to_timesteps(sigmas)
        return np.stack([timesteps.flatten(), sigmas], axis=1)

    def points_time(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        points = np.asarray(t, dtype=np.float64)

        points = regularize(points, self.scale, -self.scale * self.tail)  # 1..0 -> scale..-scale
        # WARN(beinsezii): sqrt(2) is more or less a magic number afaict
        points = np.sinh(points) if self.scale < 0 else np.tanh(points / math.sqrt(2))
        # don't use -1 because no endcaps
        points = normalize(points, points[0], -points[0] * self.tail)  # hyper..-hyper -> 1..0

        return self.base.points(points)

    def points_noise(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        sigmas = self.base.points(t)[:, 1]
        start = self.base.points([1])[0, 1].item()

        sigmas = normalize(sigmas, start)  # Base -> 1..0
        sigmas = regularize(sigmas, self.scale, -self.scale * self.tail)  # 1..0 -> scale..-scale
        # WARN(beinsezii): sqrt(2) is more or less a magic number afaict
        sigmas = np.sinh(sigmas) if self.scale < 0 else np.tanh(sigmas / math.sqrt(2))
        # don't use -1 because no endcaps
        sigmas = normalize(sigmas, sigmas[0], -sigmas[0] * self.tail)  # hyper..-hyper -> 1..0
        sigmas = regularize(sigmas, start)  # 1..0 -> Base

        return self.sigmas_to_points(sigmas)

    def points(self, t: Sequence[float] | NPSequence) -> NPSchedule:
        if self.use_noise:
            return self.points_noise(t)
        else:
            return self.points_time(t)
