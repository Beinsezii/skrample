import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from testing_common import compare_tensors, hf_scheduler_config

from skrample.scheduling import ZSNR, Flow, Scaled, SkrampleSchedule


def compare_schedules(
    a: SkrampleSchedule,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    mu: float | None = None,
    ts_margin: float = 1.0,
    sig_margin: float = 1e-3,
):
    for steps in range(1, 12):
        if isinstance(b, FlowMatchEulerDiscreteScheduler):
            # b.set_timesteps(num_inference_steps=steps, mu=mu)
            # # flux pipe hardcodes sigmas to this...
            b.set_timesteps(sigmas=torch.linspace(1.0, 1 / steps, steps), mu=mu)
        else:
            b.set_timesteps(num_inference_steps=steps)

        compare_tensors(
            torch.from_numpy(a.timesteps(steps)),
            b.timesteps,
            f"TIMESTEPS @ {steps}",
            margin=ts_margin,
        )
        compare_tensors(
            torch.from_numpy(a.sigmas(steps)),
            b.sigmas[:-1],  # type: ignore  # FloatTensor
            f"SIGMAS @ {steps}",
            margin=sig_margin,
        )


def test_scaled():
    compare_schedules(
        Scaled(uniform=False),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
        ),
    )


def test_scaled_uniform():
    compare_schedules(
        Scaled(),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
            timestep_spacing="trailing",
        ),
    )


def test_zsnr():
    compare_schedules(
        ZSNR(),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("bghira/terminus-xl-velocity-v2"),
        ),
    )


def test_flow_dynamic():
    compare_schedules(
        Flow(mu=0.7),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("black-forest-labs/FLUX.1-dev"),
        ),
        mu=0.7,
    )


def test_flow():
    compare_schedules(
        Flow(),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("stabilityai/stable-diffusion-3-medium-diffusers"),
        ),
        mu=None,
    )
