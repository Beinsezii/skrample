import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from testing_common import compare_tensors, hf_scheduler_config

from skrample.sampling import DPM, EPSILON, FLOW, VELOCITY, Euler, EulerFlow, SkrampleSampler, SKSamples, UniPC


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

    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        b_sample = b.scale_noise(sample=b_sample, timestep=timestep.unsqueeze(0), noise=noise)  # type: ignore  # FloatTensor
        subnormal = True
    else:
        b_sample = b.add_noise(original_samples=b_sample, noise=noise, timesteps=timestep.unsqueeze(0))
        subnormal = False

    a_sample = a.merge_noise(a_sample, noise, sigma.item(), subnormal=subnormal)

    prior_steps: list[SKSamples] = []
    for step in steps:
        # Just some pseud-random transform that shouldn't blow up the values
        model = torch.randn([128, 128], generator=torch.manual_seed(step), dtype=a_sample.dtype)

        timestep, sigma = schedule[step]

        a_output = a.scale_input(a_sample, sigma.item(), subnormal=subnormal) * model
        sampled = a.sample(a_sample, a_output, schedule.numpy(), step, prior_steps, subnormal)
        a_sample = sampled.sampled
        prior_steps.append(sampled)

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
    message: str = "",
):
    for step_range in [range(0, 11), range(7, 16), range(2, 22)]:
        compare_tensors(
            *dual_sample(a, b, step_range, mu),
            message=str(step_range) + (" | " + message if message else ""),
            margin=margin,
        )


def test_euler():
    for predictor in [(EPSILON, "epsilon"), (VELOCITY, "v_prediction")]:
        compare_samplers(
            Euler(predictor=predictor[0]),
            EulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
                hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
                prediction_type=predictor[1],
            ),
            message=predictor[0].__name__,
        )


def test_euler_flow():
    compare_samplers(
        EulerFlow(),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            hf_scheduler_config("black-forest-labs/FLUX.1-dev")
        ),
        mu=0.7,
    )


def test_dpm():
    for predictor in [(EPSILON, "epsilon"), (VELOCITY, "v_prediction"), (FLOW, "flow_prediction")]:
        for order in range(1, 3):
            compare_samplers(
                DPM(predictor=predictor[0], order=order),
                DPMSolverMultistepScheduler.from_config(  # type: ignore  # Diffusers return BS
                    hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
                    algorithm_type="dpmsolver++",
                    final_sigmas_type="zero",
                    solver_order=order,
                    prediction_type=predictor[1],
                ),
                message=f"{predictor[0].__name__} o{order}",
            )


def test_unipc():
    for predictor in [(EPSILON, "epsilon"), (VELOCITY, "v_prediction"), (FLOW, "flow_prediction")]:
        for order in range(1, 5):  # technically it can do N order? Let's just test till 4th now
            compare_samplers(
                UniPC(predictor=predictor[0], order=order),
                UniPCMultistepScheduler.from_config(  # type: ignore  # Diffusers return BS
                    hf_scheduler_config("stabilityai/stable-diffusion-xl-base-1.0"),
                    final_sigmas_type="zero",
                    solver_order=order,
                    prediction_type=predictor[1],
                ),
                message=f"{predictor[0].__name__} o{order}",
            )
