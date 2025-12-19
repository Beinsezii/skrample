from collections.abc import Iterable

import pytest
import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from testing_common import FLOW_CONFIG, SCALED_CONFIG, compare_pp

from skrample.scheduling import ZSNR, FlowShift, Linear, NPSchedule, Scaled, SkrampleSchedule

STEPS: Iterable[int] = [*range(1, 12)]


def get_diffusers_schedule(
    diffusers_scheduler: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler, steps: int
) -> NPSchedule:
    b = diffusers_scheduler
    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        # b.set_timesteps(num_inference_steps=steps, mu=mu)
        # # flux pipe hardcodes sigmas to this...
        b.set_timesteps(sigmas=torch.linspace(1.0, 1 / steps, steps))
    else:
        b.set_timesteps(num_inference_steps=steps)

    return torch.stack([b.timesteps, b.sigmas[:-1]], 1).to(dtype=torch.float64).numpy()


def compare_timesteps(
    a: SkrampleSchedule,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    steps: int,
    tolerance: float = 0.5,
) -> None:
    compare_pp(a.timesteps_np(steps), get_diffusers_schedule(b, steps)[:, 0], tolerance)


def compare_sigmas(
    a: SkrampleSchedule,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    steps: int,
    tolerance: float = 0.5,
) -> None:
    compare_pp(a.sigmas_np(steps), get_diffusers_schedule(b, steps)[:, 1], tolerance)


@pytest.mark.parametrize("steps", STEPS)
def test_scaled_timesteps(steps: int) -> None:
    compare_timesteps(Scaled(), EulerDiscreteScheduler.from_config(SCALED_CONFIG), steps, 2)


@pytest.mark.parametrize("steps", STEPS)
def test_scaled_sigmas(steps: int) -> None:
    compare_sigmas(Scaled(), EulerDiscreteScheduler.from_config(SCALED_CONFIG), steps)


@pytest.mark.parametrize("steps", STEPS)
def test_zsnr_timesteps(steps: int) -> None:
    compare_timesteps(
        ZSNR(), EulerDiscreteScheduler.from_config(SCALED_CONFIG | {"rescale_betas_zero_snr": True}), steps, 2
    )


@pytest.mark.parametrize("steps", STEPS)
def test_zsnr_sigmas(steps: int) -> None:
    compare_sigmas(ZSNR(), EulerDiscreteScheduler.from_config(SCALED_CONFIG | {"rescale_betas_zero_snr": True}), steps)


@pytest.mark.parametrize("steps", STEPS)
def test_flow_timesteps(steps: int) -> None:
    compare_timesteps(
        FlowShift(Linear()),
        FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG | {"use_dynamic_shifting": False}),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_flow_sigmas(steps: int) -> None:
    compare_sigmas(
        FlowShift(Linear()),
        FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG | {"use_dynamic_shifting": False}),
        steps,
    )
