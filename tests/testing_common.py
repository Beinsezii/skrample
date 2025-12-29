import dataclasses
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from skrample.common import SigmaTransform, sigma_complement, sigma_polar
from skrample.sampling.models import (
    DataModel,
    DiffusionModel,
    FlowModel,
    NoiseModel,
    ScaleX,
    VelocityModel,
)
from skrample.sampling.structured import (
    DPM,
    SPC,
    Adams,
    Euler,
    StructuredSampler,
    UniP,
    UniPC,
)
from skrample.scheduling import (
    Beta,
    Exponential,
    FlowShift,
    Hyper,
    Karras,
    Linear,
    NoMod,
    NoSub,
    Scaled,
    ScheduleCommon,
    ScheduleModifier,
    Siggauss,
    Sinner,
    SubSchedule,
)


@dataclasses.dataclass(frozen=True)
class ScaledB1(Scaled):  # INFO: So can pass raw types to parametrize()
    beta_scale: float = 1


ALL_STRUCTURED: list[type[StructuredSampler]] = [
    Adams,
    DPM,
    Euler,
    SPC,
    UniPC,
    UniP,
]


ALL_SCHEDULES: list[type[ScheduleCommon]] = [
    Linear,
    Scaled,
    ScaledB1,
]

ALL_MODIFIERS: list[type[ScheduleModifier | SubSchedule]] = [
    NoSub,
    NoMod,
    Beta,
    FlowShift,
    Karras,
    Exponential,
    Siggauss,
    Hyper,
    Sinner,
]
ALL_MODIFIERS_OPTION: list[type[ScheduleModifier | SubSchedule] | None] = [None, *ALL_MODIFIERS]

ALL_MODELS: list[type[DiffusionModel]] = [
    DataModel,
    NoiseModel,
    FlowModel,
    VelocityModel,
]

ALL_FAKE_MODELS: list[type[DiffusionModel]] = [
    ScaleX,
]

ALL_TRANSFROMS: list[SigmaTransform] = [
    sigma_complement,
    sigma_polar,
]


FLOW_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "flow_shift": 3.0,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "prediction_type": "flow_prediction",
    "shift": 3.0,
    "use_dynamic_shifting": True,
}
SCALED_CONFIG = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "trailing",
    "trained_betas": None,
    "use_karras_sigmas": False,
}


def hf_scheduler_config(
    hf_repo: str, filename: str = "scheduler_config.json", subfolder: str | None = "scheduler"
) -> dict:
    with open(hf_hub_download(hf_repo, filename, subfolder=subfolder), mode="r") as jfile:
        return json.load(jfile)


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    message: str | None = "",
    margin: float = 1e-8,
) -> None:
    assert a.isfinite().all(), message
    assert b.isfinite().all(), message
    delta = (a - b).abs().square().mean().item()
    assert delta <= margin, f"{delta} <= {margin}" + (" | " + message if message is not None else "")


def compare_pp[T: np.typing.NDArray[np.floating]](a: T, b: T, tolerance: float = 0.5) -> None:
    """Compare arrays `a` and `b`
    `tolerance` is applied as a percentage (0..=100) of the `b` tensor.
    Similar to allclose() but with more debugging information."""
    assert np.isfinite(a).all()
    assert np.isfinite(b).all()
    deviation = abs(a - b)
    relative_tolerance = (tolerance / 100) * abs(b)

    def message() -> str:
        error_percent = np.nan_to_num(deviation / abs(b), nan=0, posinf=None, neginf=None) * 100
        return (
            f"\tMIN {round(error_percent.min().item(), 2)}%\t"
            f"MEAN {round(error_percent.mean().item(), 2)}%\t"
            f"MAX {round(error_percent.max().item(), 2)}%"
        )

    assert (deviation <= relative_tolerance).all(), message()
