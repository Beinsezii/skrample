import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from testing_common import FLOW_CONFIG, SCALED_CONFIG, compare_tensors

from skrample.scheduling import ZSNR, Beta, Exponential, Flow, Karras, Scaled, SkrampleSchedule


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
            b.set_timesteps(sigmas=torch.linspace(1.0, 1 / steps, steps), mu=mu)  # type: ignore
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
            SCALED_CONFIG,
        ),
    )


def test_scaled_uniform():
    compare_schedules(
        Scaled(),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            SCALED_CONFIG,
            timestep_spacing="trailing",
        ),
    )


def test_scaled_beta():
    compare_schedules(
        Beta(Scaled()),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_beta_sigmas=True,
        ),
    )


def test_scaled_exponential():
    compare_schedules(
        Exponential(Scaled()),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_exponential_sigmas=True,
        ),
    )


def test_scaled_karras():
    compare_schedules(
        Karras(Scaled()),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_karras_sigmas=True,
        ),
    )


def test_zsnr():
    compare_schedules(
        ZSNR(),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            SCALED_CONFIG | {"timestep_spacing": "trailing", "rescale_betas_zero_snr": True}
        ),
    )


def test_flow_dynamic():
    compare_schedules(
        Flow(mu=0.7),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            FLOW_CONFIG,
        ),
        mu=0.7,
    )


def test_flow():
    compare_schedules(
        Flow(),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            FLOW_CONFIG | {"use_dynamic_shifting": False}
        ),
        mu=None,
    )


def test_flow_beta():
    compare_schedules(
        Beta(Flow()),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_beta_sigmas=True,
        ),
    )


def test_flow_exponential():
    compare_schedules(
        Exponential(Flow()),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_exponential_sigmas=True,
        ),
    )


def test_flow_karras():
    compare_schedules(
        Karras(Flow()),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_karras_sigmas=True,
        ),
    )
