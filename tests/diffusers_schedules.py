import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from testing_common import FLOW_CONFIG, SCALED_CONFIG, compare_tensors

from skrample.scheduling import ZSNR, Beta, Exponential, FlowShift, Karras, Linear, Scaled, SkrampleSchedule


def compare_schedules(
    a: SkrampleSchedule,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    mu: float | None = None,
    ts_margin: float = 1.0,
    sig_margin: float = 1e-3,
) -> None:
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
            b.sigmas[:-1],
            f"SIGMAS @ {steps}",
            margin=sig_margin,
        )


def test_scaled() -> None:
    compare_schedules(
        Scaled(uniform=False),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
        ),
    )


def test_scaled_uniform() -> None:
    compare_schedules(
        Scaled(),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
        ),
    )


def test_scaled_beta() -> None:
    compare_schedules(
        Beta(Scaled()),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_beta_sigmas=True,
        ),
    )


def test_scaled_exponential() -> None:
    compare_schedules(
        Exponential(Scaled()),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_exponential_sigmas=True,
        ),
    )


def test_scaled_karras() -> None:
    compare_schedules(
        Karras(Scaled()),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG,
            timestep_spacing="trailing",
            use_karras_sigmas=True,
        ),
    )


def test_zsnr() -> None:
    compare_schedules(
        ZSNR(),
        EulerDiscreteScheduler.from_config(
            SCALED_CONFIG | {"timestep_spacing": "trailing", "rescale_betas_zero_snr": True}
        ),
    )


def test_flow_dynamic() -> None:
    compare_schedules(
        FlowShift(Linear(), mu=0.7),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG,
        ),
        mu=0.7,
    )


def test_flow() -> None:
    compare_schedules(
        FlowShift(Linear()),
        FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG | {"use_dynamic_shifting": False}),
        mu=None,
    )


def test_flow_beta() -> None:
    compare_schedules(
        Beta(FlowShift(Linear())),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_beta_sigmas=True,
        ),
    )


def test_flow_exponential() -> None:
    compare_schedules(
        Exponential(FlowShift(Linear())),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_exponential_sigmas=True,
        ),
    )


def test_flow_karras() -> None:
    compare_schedules(
        Karras(FlowShift(Linear())),
        FlowMatchEulerDiscreteScheduler.from_config(
            FLOW_CONFIG | {"use_dynamic_shifting": False},
            use_karras_sigmas=True,
        ),
    )
