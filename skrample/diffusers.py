import dataclasses
import math
from collections import OrderedDict
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from skrample import sampling, scheduling
from skrample.common import MergeStrategy, Predictor, predict_epsilon, predict_flow, predict_sample, predict_velocity
from skrample.pytorch.noise import (
    BatchTensorNoise,
    Random,
    TensorNoiseCommon,
    TensorNoiseProps,
    schedule_to_ramp,
)
from skrample.sampling import SkrampleSampler, SKSamples
from skrample.scheduling import ScheduleCommon, ScheduleModifier, SkrampleSchedule

if TYPE_CHECKING:
    from diffusers.configuration_utils import ConfigMixin


DIFFUSERS_CLASS_MAP: dict[str, tuple[type[SkrampleSampler], dict[str, Any]]] = {
    "DDIMScheduler": (sampling.Euler, {}),
    "DDPMScheduler": (sampling.DPM, {"add_noise": True, "order": 1}),
    "DPMSolverMultistepScheduler": (sampling.DPM, {}),
    "DPMSolverSDEScheduler": (sampling.DPM, {"add_noise": True}),
    "EulerAncestralDiscreteScheduler": (sampling.DPM, {"add_noise": True, "order": 1}),
    "EulerDiscreteScheduler": (sampling.Euler, {}),
    "FlowMatchEulerDiscreteScheduler": (sampling.Euler, {}),
    "IPNDMScheduler": (sampling.Adams, {"order": 4}),
    "UniPCMultistepScheduler": (sampling.UniPC, {}),
}

DIFFUSERS_KEY_MAP: dict[str, str] = {
    # DPM and other non-FlowMatch schedulers
    "flow_shift": "shift",
    # sampling.HighOrderSampler
    "solver_order": "order",
    # scheduling.ScheduleCommon.
    "num_train_timesteps": "base_timesteps",
}
"Direct key to key mappings, leaving the values untouched."

DIFFUSERS_KEY_MAP_REV: dict[str, str] = {v: k for k, v in DIFFUSERS_KEY_MAP.items()}
"DIFFUSERS_KEY_MAP with keys and values inverted"

DIFFUSERS_VALUE_MAP: dict[tuple[str, Any], tuple[str, Any]] = {
    # scheduling.Scaled
    ("beta_schedule", "linear"): ("beta_scale", 1),
    ("beta_schedule", "scaled_linear"): ("beta_scale", 2),
    ("timestep_spacing", "leading"): ("uniform", False),
    ("timestep_spacing", "linspace"): ("uniform", True),
    ("timestep_spacing", "trailing"): ("uniform", True),
    # sampling.StochasticSampler
    ("algorithm_type", "dpmsolver"): ("add_noise", False),
    ("algorithm_type", "dpmsolver++"): ("add_noise", False),
    ("algorithm_type", "sde-dpmsolver"): ("add_noise", True),
    ("algorithm_type", "sde-dpmsolver++"): ("add_noise", True),
    # Complex types
    ("prediction_type", "epsilon"): ("skrample_predictor", predict_epsilon),
    ("prediction_type", "flow"): ("skrample_predictor", predict_flow),
    ("prediction_type", "sample"): ("skrample_predictor", predict_sample),
    ("prediction_type", "v_prediction"): ("skrample_predictor", predict_velocity),
    ("use_beta_sigmas", True): ("skrample_modifier", scheduling.Beta),
    ("use_exponential_sigmas", True): ("skrample_modifier", scheduling.Exponential),
    ("use_karras_sigmas", True): ("skrample_modifier", scheduling.Karras),
}
"Key/value to key/value map. For mapping more complex types against diffusers"

DIFFUSERS_VALUE_MAP_REV: dict[tuple[str, Any], tuple[str, Any]] = {v: k for k, v in DIFFUSERS_VALUE_MAP.items()}
"DIFFUSERS_VALUE_MAP with keys and values inverted"


DEFAULT_FAKE_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "use_dynamic_shifting": True,
}
"Baseline fake config presented for pipelines to not raise exceptions"


@dataclasses.dataclass(frozen=True)
class ParsedDiffusersConfig:
    "Values read from a combination of the diffusers config and provided types"

    sampler: type[SkrampleSampler]
    sampler_props: dict[str, Any]
    schedule: type[SkrampleSchedule]
    schedule_props: dict[str, Any]
    schedule_modifiers: list[tuple[type[ScheduleModifier], dict[str, Any]]]
    predictor: Predictor


def parse_diffusers_config(
    config: "dict[str, Any] | ConfigMixin",
    sampler: type[SkrampleSampler] | None = None,
    schedule: type[SkrampleSchedule] | None = None,
) -> ParsedDiffusersConfig:
    """Reads a diffusers scheduler or config as a set of skrample classes and properties.
    Input sampler/schedule types may or may not influence output"""

    diffusers_class = config.get("_class_name", "") if isinstance(config, dict) else type(config).__name__
    if not isinstance(config, dict):
        config = dict(config.config)

    remapped = (
        dict(config)  # ensure not ordered/frozen
        | {DIFFUSERS_KEY_MAP[k]: v for k, v in config.items() if k in DIFFUSERS_KEY_MAP}
        | {
            DIFFUSERS_VALUE_MAP[(k, v)][0]: DIFFUSERS_VALUE_MAP[(k, v)][1]
            for k, v in config.items()
            if isinstance(v, Hashable) and (k, v) in DIFFUSERS_VALUE_MAP
        }
    )

    if "skrample_predictor" in remapped:
        predictor: Predictor = remapped.pop("skrample_predictor")
    elif "shift" in remapped:  # should only be flow
        predictor = predict_flow
    else:
        predictor = predict_epsilon

    if not sampler:
        sampler, sampler_props = DIFFUSERS_CLASS_MAP.get(diffusers_class, (sampling.DPM, {}))
    else:
        sampler_props = {}

    if not schedule:
        if predictor is predict_flow:
            schedule = scheduling.Linear
        elif remapped.get("rescale_betas_zero_snr", False):
            schedule = scheduling.ZSNR
        else:
            schedule = scheduling.Scaled

    # Adjust sigma_start to match scaled beta for sd1/xl
    if "sigma_start" not in remapped and predictor is not predict_flow and issubclass(schedule, scheduling.Linear):
        scaled_keys = [f.name for f in dataclasses.fields(scheduling.Scaled)]
        # non-uniform misses a whole timestep
        scaled = scheduling.Scaled(**{k: v for k, v in remapped.items() if k in scaled_keys} | {"uniform": True})
        sigma_start: float = scaled.sigmas(1).item()
        remapped["sigma_start"] = math.sqrt(sigma_start)

    schedule_modifiers: list[tuple[type[ScheduleModifier], dict[str, Any]]] = []

    if predictor is predict_flow:
        flow_keys = [f.name for f in dataclasses.fields(scheduling.FlowShift)]
        schedule_modifiers.append((scheduling.FlowShift, {k: v for k, v in remapped.items() if k in flow_keys}))

    if "skrample_modifier" in remapped:
        schedule_modifier: type[ScheduleModifier] = remapped.pop("skrample_modifier")
        modifier_keys = [f.name for f in dataclasses.fields(schedule_modifier)]
        schedule_modifiers.append((schedule_modifier, {k: v for k, v in remapped.items() if k in modifier_keys}))

    # feels cleaner than inspect.signature().parameters
    sampler_keys = [f.name for f in dataclasses.fields(sampler)]
    schedule_keys = [f.name for f in dataclasses.fields(schedule)]

    return ParsedDiffusersConfig(
        sampler=sampler,
        sampler_props=sampler_props | {k: v for k, v in (remapped).items() if k in sampler_keys},
        schedule=schedule,
        schedule_props={k: v for k, v in remapped.items() if k in schedule_keys},
        schedule_modifiers=schedule_modifiers,
        predictor=predictor,
    )


def attr_dict[T: Any](**kwargs: T) -> OrderedDict[str, T]:
    od = OrderedDict(**kwargs)
    for k, v in od.items():
        setattr(od, k, v)
    return od


def as_diffusers_config(sampler: SkrampleSampler, schedule: SkrampleSchedule, predictor: Predictor) -> dict[str, Any]:
    "Converts skrample classes back into a diffusers-readable config. Not comprehensive"
    skrample_config = dataclasses.asdict(sampler)
    skrample_config["skrample_predictor"] = predictor

    if isinstance(schedule, ScheduleModifier):
        for modifier in schedule.all:
            skrample_config |= dataclasses.asdict(modifier)
        skrample_config["skrample_modifier"] = type(schedule)
    else:
        skrample_config |= dataclasses.asdict(schedule)

    return (
        skrample_config
        | {DIFFUSERS_KEY_MAP_REV[k]: v for k, v in skrample_config.items() if k in DIFFUSERS_KEY_MAP_REV}
        | {
            DIFFUSERS_VALUE_MAP_REV[(k, v)][0]: DIFFUSERS_VALUE_MAP_REV[(k, v)][1]
            for k, v in skrample_config.items()
            if isinstance(v, Hashable) and (k, v) in DIFFUSERS_VALUE_MAP_REV
        }
    )


@dataclasses.dataclass
class SkrampleWrapperScheduler[T: TensorNoiseProps | None]:
    """Wrapper class to present skrample types in a way that diffusers' DiffusionPipelines can handle.
    Best effort approach. Most of the items presented in .config are fake, and many function inputs are ignored.
    A general rule of thumb is it will always prioritize the skrample properties over the incoming properties."""

    sampler: SkrampleSampler
    schedule: SkrampleSchedule
    predictor: Predictor[Tensor] = predict_epsilon
    noise_type: type[TensorNoiseCommon[T]] = Random  # type: ignore  # Unsure why?
    noise_props: T | None = None
    compute_scale: torch.dtype | None = torch.float32
    allow_dynamic: bool = True
    """Whether or not classes can be overridden during sampling.
    Currently only applies to FlowShift when `mu` is provided, IE "use_dynamic_shifting" in diffusers."""
    fake_config: dict[str, Any] = dataclasses.field(default_factory=lambda: DEFAULT_FAKE_CONFIG.copy())
    """Extra items presented in scheduler.config to the pipeline.
    It is recommended to use an actual diffusers scheduler config if one is available."""

    def __post_init__(self) -> None:
        # State
        self._steps: int = 50
        self._device: torch.device = torch.device("cpu")
        self._previous: list[SKSamples[Tensor]] = []
        self._noise_generator: BatchTensorNoise | None = None
        self._schedule = self.schedule  # copy of original for restoration in set_timesteps

    @classmethod
    def from_diffusers_config[N: TensorNoiseProps | None](  # pyright fails if you use the outer generic
        cls,
        config: "dict[str, Any] | ConfigMixin",
        sampler: type[SkrampleSampler] | None = None,
        schedule: type[SkrampleSchedule] | None = None,
        schedule_modifiers: list[tuple[type[ScheduleModifier], dict[str, Any]]] = [],
        predictor: Predictor[Tensor] | None = None,
        noise_type: type[TensorNoiseCommon[N]] = Random,
        compute_scale: torch.dtype | None = torch.float32,
        sampler_props: dict[str, Any] = {},
        noise_props: N | None = None,
        schedule_props: dict[str, Any] = {},
        modifier_merge_strategy: MergeStrategy = MergeStrategy.UniqueBefore,
        allow_dynamic: bool = True,
    ) -> "SkrampleWrapperScheduler[N]":
        "Thin sugar over `parse_diffusers_config` to make a complete wrapper with arbitrary customizations"
        parsed = parse_diffusers_config(config=config, sampler=sampler, schedule=schedule)

        built_sampler = (sampler or parsed.sampler)(**parsed.sampler_props | sampler_props)
        built_schedule = (schedule or parsed.schedule)(**parsed.schedule_props | schedule_props)

        if isinstance(built_schedule, ScheduleCommon | ScheduleModifier):
            for modifier, modifier_props in modifier_merge_strategy.merge(
                ours=schedule_modifiers,
                theirs=parsed.schedule_modifiers,
                cmp=lambda a, b: a[0] is b[0],
            ):
                built_schedule = modifier(base=built_schedule, **modifier_props)

        return cls(
            built_sampler,
            built_schedule,
            predictor or parsed.predictor,
            noise_type=noise_type,  # type: ignore  # think these are weird because of the defaults?
            noise_props=noise_props,  # type: ignore
            compute_scale=compute_scale,
            fake_config=config.copy() if isinstance(config, dict) else dict(config.config),
            allow_dynamic=allow_dynamic,
        )

    @property
    def schedule_np(self) -> NDArray[np.float64]:
        return scheduling.schedule_lru(self.schedule, self._steps)

    @property
    def schedule_pt(self) -> Tensor:
        return torch.from_numpy(self.schedule_np).to(self._device)

    @property
    def timesteps(self) -> Tensor:
        return torch.from_numpy(self.schedule_np[:, 0]).to(self._device)

    @property
    def sigmas(self) -> Tensor:
        sigmas = torch.from_numpy(self.schedule_np[:, 1]).to(self._device)
        # diffusers expects the extra zero
        return torch.cat([sigmas, torch.zeros([1], device=sigmas.device, dtype=sigmas.dtype)])

    @property
    def init_noise_sigma(self) -> float:
        return self.sampler.scale_input(1, self.schedule_np[0, 1].item(), sigma_transform=self.schedule.sigma_transform)

    @property
    def order(self) -> int:
        return 1  # for multistep this is always 1

    @property
    def config(self) -> OrderedDict[str, Any]:
        # Diffusers expects the frozen shift value
        return attr_dict(**(self.fake_config | as_diffusers_config(self.sampler, self._schedule, self.predictor)))

    def time_shift(self, mu: float, sigma: float, t: Tensor) -> Tensor:
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_begin_index(self, begin_index: int = 0) -> None:
        self.fake_config["begin_index"] = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        timesteps: Tensor | list[int] | None = None,
        sigmas: Tensor | list[float] | None = None,
        mu: float | None = None,
    ) -> None:
        self.schedule = self._schedule  # Restore any replaced props

        if num_inference_steps is None:
            if timesteps is not None:
                num_inference_steps = len(timesteps)
            elif sigmas is not None:
                num_inference_steps = len(sigmas)
            else:
                return

        self._steps = num_inference_steps

        if (
            self.allow_dynamic
            and mu is not None
            and isinstance(self.schedule, scheduling.ScheduleModifier)
            and (found := self.schedule.find_split(scheduling.FlowShift)) is not None
        ):
            before, flow, after, base = found
            self.schedule = self.schedule.stack([*before, dataclasses.replace(flow, shift=math.exp(mu)), *after], base)

        self._previous = []
        self._noise_generator = None

        if device is not None:
            self._device = torch.device(device)

    def scale_noise(self, sample: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep.item())  # type: ignore  # np v2 Number
        sigma = schedule[step, 1].item()
        return self.sampler.merge_noise(sample, noise, sigma, sigma_transform=self.schedule.sigma_transform)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.scale_noise(original_samples, timesteps[0], noise)

    def scale_model_input(self, sample: Tensor, timestep: float | Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int | float)) else timestep.item())  # type: ignore  # np v2 Number
        sigma = schedule[step, 1].item()
        return self.sampler.scale_input(sample, sigma, sigma_transform=self.schedule.sigma_transform)

    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
    ) -> tuple[Tensor, Tensor] | OrderedDict[str, Tensor]:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, int | float) else timestep.item())  # type: ignore  # np v2 Number

        if self.sampler.require_noise:
            if self._noise_generator is None:
                if isinstance(generator, list) and len(generator) == sample.shape[0]:
                    seeds = generator
                elif isinstance(generator, torch.Generator) and sample.shape[0] == 1:
                    seeds = [generator]
                else:
                    # use median element +4 decimals as seed for a balance of determinism without lacking variety
                    # multiply by step index to spread the values and minimize clash
                    # does not work across batch sizes but at least Flux will have something mostly deterministic
                    seeds = [
                        torch.Generator().manual_seed(
                            int(b.reshape(b.numel())[b.numel() // 2].item() * 1e4) * (step + 1)
                        )
                        for b in sample
                    ]

                self._noise_generator = BatchTensorNoise.from_batch_inputs(
                    self.noise_type,
                    unit_shape=sample.shape[1:],
                    seeds=seeds,
                    props=self.noise_props,
                    ramp=schedule_to_ramp(schedule),
                    dtype=torch.float32,
                )

            noise = self._noise_generator.generate().to(dtype=self.compute_scale, device=model_output.device)
        else:
            noise = None

        sample_cast = sample.to(dtype=self.compute_scale)
        prediction = self.predictor(
            sample_cast,
            model_output.to(dtype=self.compute_scale),
            schedule[step, 1].item(),
            self.schedule.sigma_transform,
        )
        sampled = self.sampler.sample(
            sample=sample_cast,
            prediction=prediction,
            sigma_schedule=schedule[:, 1],
            step=step,
            noise=noise,
            previous=tuple(self._previous),
            sigma_transform=self.schedule.sigma_transform,
        )
        self._previous.append(sampled)
        self._previous = self._previous[max(len(self._previous) - self.sampler.require_previous, 0) :]

        if return_dict:
            return attr_dict(
                prev_sample=sampled.final.to(device=model_output.device, dtype=model_output.dtype),
                pred_original_sample=sampled.prediction.to(device=model_output.device, dtype=model_output.dtype),
            )
        else:
            return (
                sampled.final.to(device=model_output.device, dtype=model_output.dtype),
                sampled.prediction.to(device=model_output.device, dtype=model_output.dtype),
            )
