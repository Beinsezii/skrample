import math

import pytest
import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from testing_common import FLOW_CONFIG, SCALED_CONFIG, compare_tensors

from skrample.scheduling import ZSNR, Beta, Exponential, FlowShift, Karras, Linear, Scaled, SkrampleSchedule

STEPS = range(1, 12)


def compare_schedules(
    a: SkrampleSchedule,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    steps: int,
    mu: float | None = None,
    ts_margin: float = 1.0,
    sig_margin: float = 1e-3,
) -> None:
    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        # b.set_timesteps(num_inference_steps=steps, mu=mu)
        # # flux pipe hardcodes sigmas to this...
        b.set_timesteps(sigmas=torch.linspace(1.0, 1 / steps, steps), mu=mu)
    else:
        b.set_timesteps(num_inference_steps=steps)

    compare_tensors(
        torch.from_numpy(a.timesteps_np(steps)),
        b.timesteps,
        f"TIMESTEPS @ {steps}",
        margin=ts_margin,
    )
    compare_tensors(
        torch.from_numpy(a.sigmas_np(steps)),
        b.sigmas[:-1],
        f"SIGMAS @ {steps}",
        margin=sig_margin,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_scaled(steps: int) -> None:
    compare_schedules(
        Scaled(uniform=False),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_scaled_uniform(steps: int) -> None:
    compare_schedules(
        Scaled(),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_scaled_beta(steps: int) -> None:
    compare_schedules(
        Beta(Scaled()),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_beta_sigmas=True,
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_scaled_exponential(steps: int) -> None:
    compare_schedules(
        Exponential(Scaled()),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_exponential_sigmas=True,
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_scaled_karras(steps: int) -> None:
    compare_schedules(
        Karras(Scaled()),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_karras_sigmas=True,
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_zsnr(steps: int) -> None:
    compare_schedules(
        ZSNR(),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG | {"timestep_spacing": "trailing", "rescale_betas_zero_snr": True}
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_flow_dynamic(steps: int) -> None:
    compare_schedules(
        FlowShift(Linear(), shift=math.exp(0.7)),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG,
        ),
        steps,
        mu=0.7,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_flow(steps: int) -> None:
    compare_schedules(
        FlowShift(Linear()),
        FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG | {"use_dynamic_shifting": False}),
        steps,
        mu=None,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_flow_beta(steps: int) -> None:
    compare_schedules(
        Beta(FlowShift(Linear())),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_beta_sigmas=True,
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_flow_exponential(steps: int) -> None:
    compare_schedules(
        Exponential(FlowShift(Linear())),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_exponential_sigmas=True,
        ),
        steps,
    )


@pytest.mark.parametrize("steps", STEPS)
def test_flow_karras(steps: int) -> None:
    compare_schedules(
        Karras(FlowShift(Linear())),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_karras_sigmas=True,
        ),
        steps,
    )
