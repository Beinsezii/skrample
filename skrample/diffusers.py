import dataclasses
import math
from collections import OrderedDict
from typing import Any, Hashable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from skrample import sampling, scheduling
from skrample.sampling import PREDICTOR, SkrampleSampler, SKSamples, StochasticSampler
from skrample.scheduling import ScheduleModifier, SkrampleSchedule

DIFFUSERS_KEY_MAP: dict[str, str] = {
    # DPM and other non-FlowMatch schedulers
    "flow_shift": "shift",
    # sampling.HighOrderSampler
    "solver_order": "order",
}

DIFFUSERS_KEY_MAP_REV: dict[str, str] = {v: k for k, v in DIFFUSERS_KEY_MAP.items()}

DIFFUSERS_VALUE_MAP: dict[tuple[str, Any], tuple[str, Any]] = {
    # scheduling.Scaled
    ("beta_schedule", "linear"): ("beta_scale", 1),
    ("beta_schedule", "scaled_linear"): ("beta_scale", 2),
    ("timestep_spacing", "leading"): ("uniform", False),
    ("timestep_spacing", "linspace"): ("uniform", True),
    ("timestep_spacing", "trailing"): ("uniform", True),
    # Complex types
    ("prediction_type", "epsilon"): ("skrample_predictor", sampling.EPSILON),
    ("prediction_type", "flow"): ("skrample_predictor", sampling.FLOW),
    ("prediction_type", "sample"): ("skrample_predictor", sampling.SAMPLE),
    ("prediction_type", "v_prediction"): ("skrample_predictor", sampling.VELOCITY),
    ("use_beta_sigmas", True): ("skrample_modifier", scheduling.Beta),
    ("use_exponential_sigmas", True): ("skrample_modifier", scheduling.Exponential),
    ("use_karras_sigmas", True): ("skrample_modifier", scheduling.Karras),
}

DIFFUSERS_VALUE_MAP_REV: dict[tuple[str, Any], tuple[str, Any]] = {v: k for k, v in DIFFUSERS_VALUE_MAP.items()}


@dataclasses.dataclass(frozen=True)
class ParsedDiffusersConfig:
    sampler: type[SkrampleSampler]
    sampler_props: dict[str, Any]
    predictor: PREDICTOR
    schedule: type[SkrampleSchedule]
    schedule_props: dict[str, Any]
    modifier: type[ScheduleModifier] | None
    modifier_props: dict[str, Any]


def parse_diffusers_config(
    # really don't wanna make a huge manual map.
    # all our samplers work everywhere so let's just require it
    sampler: type[SkrampleSampler],
    schedule: type[SkrampleSchedule] | None = None,
    schedule_modifier: type[ScheduleModifier] | None = None,
    **config,
) -> ParsedDiffusersConfig:
    remapped = (
        config
        | {DIFFUSERS_KEY_MAP[k]: v for k, v in config.items() if k in DIFFUSERS_KEY_MAP}
        | {
            DIFFUSERS_VALUE_MAP[(k, v)][0]: DIFFUSERS_VALUE_MAP[(k, v)][1]
            for k, v in config.items()
            if isinstance(v, Hashable) and (k, v) in DIFFUSERS_VALUE_MAP
        }
    )

    if "skrample_predictor" in remapped:
        pop: PREDICTOR = remapped.pop("skrample_predictor")
        predictor = pop
    elif "shift" in remapped:  # should only be flow
        predictor = sampling.FLOW
    else:
        predictor = sampling.EPSILON

    if not schedule:
        if predictor is sampling.FLOW:
            schedule = scheduling.Flow
        elif remapped.get("rescale_betas_zero_snr", False):
            schedule = scheduling.ZSNR
        else:
            schedule = scheduling.Scaled

    if not schedule_modifier:
        schedule_modifier = remapped.pop("skrample_modifier", None)

    # feels cleaner than inspect.signature().parameters
    sampler_keys = [f.name for f in dataclasses.fields(sampler)]
    schedule_keys = [f.name for f in dataclasses.fields(schedule)]

    if schedule_modifier:
        modifier_keys = [f.name for f in dataclasses.fields(schedule_modifier)]
        modifier_props = {k: v for k, v in remapped.items() if k in modifier_keys}
    else:
        modifier_props = {}

    return ParsedDiffusersConfig(
        sampler=sampler,
        sampler_props={k: v for k, v in remapped.items() if k in sampler_keys},
        predictor=predictor,
        schedule=schedule,
        schedule_props={k: v for k, v in remapped.items() if k in schedule_keys},
        modifier=schedule_modifier,
        modifier_props=modifier_props,
    )


def as_diffusers_config(sampler: SkrampleSampler, schedule: SkrampleSchedule) -> dict[str, Any]:
    skrample_config = dataclasses.asdict(sampler)
    skrample_config["skrample_predictor"] = sampler.predictor

    if isinstance(schedule, ScheduleModifier):
        skrample_config |= dataclasses.asdict(schedule.base) | dataclasses.asdict(schedule)
        skrample_config["skrample_modifier"] = type(schedule)
    else:
        skrample_config |= dataclasses.asdict(schedule)

    return (
        skrample_config
        | {DIFFUSERS_KEY_MAP_REV[k]: v for k, v in skrample_config.items() if k in DIFFUSERS_KEY_MAP_REV}
        | {
            DIFFUSERS_VALUE_MAP_REV[(k, v)][0]: DIFFUSERS_VALUE_MAP_REV[(k, v)][1]
            for k, v in skrample_config.items()
            if (k, v) in DIFFUSERS_VALUE_MAP_REV
        }
    )


class SkrampleWrapperScheduler:
    sampler: SkrampleSampler
    schedule: SkrampleSchedule
    fake_config: dict[str, Any]
    compute_scale: torch.dtype | None

    _steps: int = 50
    _device: torch.device = torch.device("cpu")
    _previous: list[SKSamples] = []

    def __init__(
        self,
        sampler: SkrampleSampler,
        schedule: SkrampleSchedule,
        compute_scale: torch.dtype | None = torch.float32,
        fake_config: dict[str, Any] = {  # Required for FluxPipeline to not die
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "use_dynamic_shifting": True,
        },
    ):
        self.sampler = sampler
        self.schedule = schedule
        self.compute_scale = compute_scale
        self.fake_config = fake_config

    @classmethod
    def from_diffusers_config(
        cls,
        sampler: type[SkrampleSampler],
        schedule: type[SkrampleSchedule] | None = None,
        schedule_modifier: type[ScheduleModifier] | None = None,
        predictor: PREDICTOR | None = None,
        compute_scale: torch.dtype | None = None,
        sampler_props: dict[str, Any] = {},
        schedule_props: dict[str, Any] = {},
        schedule_modifier_props: dict[str, Any] = {},
        **config,
    ):
        parsed = parse_diffusers_config(
            sampler=sampler,
            schedule=schedule,
            schedule_modifier=schedule_modifier,
            **config,
        )

        sampler_props = parsed.sampler_props | sampler_props
        schedule_props = parsed.schedule_props | schedule_props
        modifier_props = parsed.modifier_props | schedule_modifier_props

        built_sampler = parsed.sampler(**sampler_props)
        built_schedule = parsed.schedule(**schedule_props)
        if parsed.modifier:
            built_schedule = parsed.modifier(base=built_schedule, **modifier_props)

        built_sampler.predictor = predictor or parsed.predictor

        return cls(
            built_sampler,
            built_schedule,
            compute_scale=compute_scale,
            fake_config=config.copy(),
        )

    @property
    def schedule_np(self) -> NDArray[np.float64]:
        return self.schedule(steps=self._steps)

    @property
    def schedule_pt(self) -> Tensor:
        return torch.from_numpy(self.schedule_np).to(self._device)

    @property
    def timesteps(self) -> Tensor:
        return torch.from_numpy(self.schedule.timesteps(steps=self._steps)).to(self._device)

    @property
    def sigmas(self) -> Tensor:
        sigmas = torch.from_numpy(self.schedule.sigmas(steps=self._steps)).to(self._device)
        # diffusers expects the extra zero
        return torch.cat([sigmas, torch.zeros([1], device=sigmas.device, dtype=sigmas.dtype)])

    @property
    def init_noise_sigma(self) -> float:
        # idk why tf diffusers uses this instead of add_noise() for some shit
        return self.sampler.merge_noise(0, 1, self.schedule_np[0, 1].item(), subnormal=self.schedule.subnormal)  # type: ignore

    @property
    def order(self) -> int:
        return 1  # for multistep this is always 1

    @property
    def config(self):
        fake_config_object = OrderedDict(self.fake_config | as_diffusers_config(self.sampler, self.schedule))

        for k, v in fake_config_object.items():
            setattr(fake_config_object, k, v)

        return fake_config_object

    def time_shift(self, mu: float, sigma: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        timesteps: Tensor | list[int] | None = None,
        sigmas: Tensor | list[float] | None = None,
        mu: float | None = None,
    ):
        if num_inference_steps is None:
            if timesteps is not None:
                num_inference_steps = len(timesteps)
            elif sigmas is not None:
                num_inference_steps = len(sigmas)
            else:
                return

        self._steps = num_inference_steps
        if isinstance(self.schedule, scheduling.Flow):
            self.schedule.mu = mu
        elif isinstance(self.schedule, ScheduleModifier) and isinstance(self.schedule.base, scheduling.Flow):
            self.schedule.base.mu = mu
        self._previous = []

        if device is not None:
            self._device = torch.device(device)

    def scale_noise(self, sample: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep.item())  # type: ignore  # np v2 Number
        sigma = schedule[step, 1].item()
        return self.sampler.merge_noise(sample, noise, sigma, subnormal=self.schedule.subnormal)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return self.scale_noise(original_samples, timesteps[0], noise)

    def scale_model_input(self, sample: Tensor, timestep: float | Tensor) -> Tensor:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int | float)) else timestep.item())  # type: ignore  # np v2 Number
        sigma = schedule[step, 1].item()
        return self.sampler.scale_input(sample, sigma, subnormal=self.schedule.subnormal)

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
    ) -> tuple[Tensor, Tensor]:
        schedule = self.schedule_np
        step = schedule[:, 0].tolist().index(timestep if isinstance(timestep, (int, float)) else timestep.item())  # type: ignore  # np v2 Number

        if isinstance(self.sampler, StochasticSampler) and self.sampler.add_noise:
            if isinstance(generator, list) and len(generator) == sample.shape[0]:
                seeds = generator
            elif isinstance(generator, torch.Generator) and sample.shape[0] == 1:
                seeds = [generator]
            else:
                # use median element +4 decimals as seed for a balance of determinism without lacking variety
                # multiply by step index to spread the values and minimize clash
                # does not work across batch sizes but at least Flux will have something mostly deterministic
                seeds = [
                    torch.Generator().manual_seed(int(b.view(b.numel())[b.numel() // 2].item() * 1e4) * (step + 1))
                    for b in sample
                ]

            noise = torch.stack(
                [
                    torch.randn(
                        sample.shape[1:],  # exclude batch
                        generator=g,  # should we even use the generators?
                        dtype=torch.float32,  # lock to cpu f32 for consistent seeds
                        device="cpu",
                    ).to(device=sample.device, dtype=self.compute_scale)
                    for g in seeds
                ]
            )
        else:
            noise = None

        if return_dict:
            raise ValueError
        else:
            sampled = self.sampler.sample(
                sample=sample.to(dtype=self.compute_scale),
                output=model_output.to(dtype=self.compute_scale),
                sigma_schedule=schedule[:, 1],
                step=step,
                noise=noise,
                previous=self._previous,
                subnormal=self.schedule.subnormal,
            )
            self._previous.append(sampled)
            return (
                sampled.final.to(device=model_output.device, dtype=model_output.dtype),
                sampled.prediction.to(device=model_output.device, dtype=model_output.dtype),
            )
