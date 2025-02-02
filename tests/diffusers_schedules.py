import json

import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

from skrample.scheduling import ScaledSchedule, ScheduleTrait
from tests.common import compare_tensors


def hf_scheduler_config(
    hf_repo: str, filename: str = "scheduler_config.json", subfolder: str | None = "scheduler"
) -> dict:
    with open(hf_hub_download(hf_repo, filename, subfolder=subfolder), mode="r") as jfile:
        return json.load(jfile)


def compare_schedules(
    a: ScheduleTrait,
    b: EulerDiscreteScheduler,
    ts_margin: float = 1.0,
    sig_margin: float = 1e-3,
):
    for steps in range(1, 12):
        b.set_timesteps(num_inference_steps=steps)

        compare_tensors(torch.from_numpy(a.timesteps(steps)), b.timesteps, f"TIMESTEPS @ {steps}", margin=ts_margin)
        compare_tensors(torch.from_numpy(a.sigmas(steps)), b.sigmas[:-1], f"SIGMAS @ {steps}", margin=sig_margin)


def test_scaled():
    compare_schedules(
        ScaledSchedule(uniform=False),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
        ),
    )


def test_scaled_uniform():
    compare_schedules(
        ScaledSchedule(),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
            timestep_spacing="trailing",
        ),
    )
