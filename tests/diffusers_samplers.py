import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from skrample.sampling import VELOCITY, Euler, EulerFlow, SkrampleSampler
from tests.common import compare_tensors, hf_scheduler_config


def dual_sample(
    a: SkrampleSampler,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    steps: range,
    mu: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Inputs
    a_sample = torch.zeros([1, 4, 128, 128], dtype=torch.float32)
    b_sample = a_sample.clone()
    noise = torch.randn(a_sample.shape, generator=torch.manual_seed(0), dtype=a_sample.dtype)

    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        b.set_timesteps(steps.stop, mu=mu)
    else:
        b.set_timesteps(steps.stop)

    # Use the same exact schedule for both to reduce variables
    schedule = torch.stack([b.timesteps, b.sigmas[:-1]], dim=1)  # type: ignore  # FloatTensor
    timestep, sigma = schedule[steps.start]

    a_sample = a.merge_noise(a_sample, noise, sigma.item())

    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        b_sample = b.scale_noise(sample=b_sample, timestep=timestep.unsqueeze(0), noise=noise)  # type: ignore  # FloatTensor
    else:
        b_sample = b.add_noise(original_samples=b_sample, noise=noise, timesteps=timestep.unsqueeze(0))

    for step in steps:
        # Just some pseud-random transform that shouldn't blow up the values
        model = torch.randn([128, 128], generator=torch.manual_seed(step), dtype=a_sample.dtype)

        timestep, sigma = schedule[step]

        a_output = a.scale_input(a_sample, sigma.item()) * model
        a_sample = a.sample(a_sample, a_output, schedule.numpy(), step)

        if isinstance(b, FlowMatchEulerDiscreteScheduler):
            b_output = b_sample * model
        else:
            b_output = b.scale_model_input(sample=b_sample, timestep=timestep) * model

        b_sample = b.step(model_output=b_output, sample=b_sample, timestep=timestep)[0]  # type: ignore  # FloatTensor

    return a_sample, b_sample


def compare_samplers(
    a: SkrampleSampler,
    b: EulerDiscreteScheduler,
    mu: float | None = None,
    margin: float = 1e-4,
):
    for step_range in [range(0, 11), range(7, 16), range(2, 22)]:
        compare_tensors(*dual_sample(a, b, step_range, mu), message=str(step_range), margin=margin)


def test_euler():
    compare_samplers(
        Euler(),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0")
        ),
    )


def test_euler_velocity():
    compare_samplers(
        Euler(predictor=VELOCITY),
        EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
            prediction_type="v_prediction",
        ),
    )


def test_euler_flow():
    compare_samplers(
        EulerFlow(),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("black-forest-labs/FLUX.1-dev")
        ),
        mu=0.7,
    )
