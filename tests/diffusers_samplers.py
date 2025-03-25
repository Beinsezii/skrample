from inspect import signature

import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from testing_common import FLOW_CONFIG, SCALED_CONFIG, compare_tensors

from skrample.sampling import DPM, EPSILON, FLOW, VELOCITY, Euler, SkrampleSampler, SKSamples, UniPC

DiffusersScheduler = (
    EulerDiscreteScheduler | DPMSolverMultistepScheduler | FlowMatchEulerDiscreteScheduler | UniPCMultistepScheduler
)


def dual_sample(
    a: SkrampleSampler,
    b: DiffusersScheduler,
    steps: range,
    mu: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Inputs
    a_sample = torch.zeros([1, 4, 128, 128], dtype=torch.float32)
    b_sample = a_sample.clone()
    seed = torch.manual_seed(0)
    initial_noise = torch.randn(a_sample.shape, generator=seed, dtype=a_sample.dtype)

    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        b.set_timesteps(steps.stop, mu=mu)
    else:
        b.set_timesteps(steps.stop)

    # Use the same exact schedule for both to reduce variables
    schedule = torch.stack([b.timesteps, b.sigmas[:-1]], dim=1)  # type: ignore  # FloatTensor
    timestep, sigma = schedule[steps.start]

    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        b_sample = b.scale_noise(sample=b_sample, timestep=timestep.unsqueeze(0), noise=initial_noise)  # type: ignore  # FloatTensor
        subnormal = True
    else:
        b_sample = b.add_noise(original_samples=b_sample, noise=initial_noise, timesteps=timestep.unsqueeze(0))  # type: ignore  # IntTensor
        subnormal = False

    a_sample = a.merge_noise(a_sample, initial_noise, sigma.item(), subnormal=subnormal)

    prior_steps: list[SKSamples] = []
    for step in steps:
        # Just some pseud-random transform that shouldn't blow up the values
        model = torch.randn([128, 128], generator=seed, dtype=a_sample.dtype)
        noise = torch.randn(a_sample.shape, generator=seed.clone_state(), dtype=a_sample.dtype)

        timestep, sigma = schedule[step]

        a_output = a.scale_input(a_sample, sigma.item(), subnormal=subnormal) * model
        sampled = a.sample(a_sample, a_output, schedule[:, 1].numpy(), step, noise, prior_steps, subnormal)
        a_sample = sampled.final
        prior_steps.append(sampled)

        if isinstance(b, FlowMatchEulerDiscreteScheduler):
            b_output = b_sample * model
        else:
            b_output = b.scale_model_input(sample=b_sample, timestep=timestep) * model

        if "generator" in signature(b.step).parameters:  # why, diffusers, why
            b_sample = b.step(model_output=b_output, sample=b_sample, timestep=timestep, generator=seed)[0]  # type: ignore  # FloatTensor
        else:
            b_sample = b.step(model_output=b_output, sample=b_sample, timestep=timestep)[0]  # type: ignore  # FloatTensor

    return a_sample, b_sample


def compare_samplers(
    a: SkrampleSampler,
    b: DiffusersScheduler,
    mu: float | None = None,
    margin: float = 1e-8,
    message: str = "",
):
    for step_range in [range(0, 2), range(0, 11), range(0, 201), range(3, 6), range(2, 23), range(31, 200)]:
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
                SCALED_CONFIG,
                prediction_type=predictor[1],
            ),
            message=predictor[0].__name__,
        )


def test_euler_ancestral():
    for predictor in [(EPSILON, "epsilon"), (VELOCITY, "v_prediction")]:
        compare_samplers(
            Euler(add_noise=True, predictor=predictor[0]),
            EulerAncestralDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
                SCALED_CONFIG,
                prediction_type=predictor[1],
            ),
            message=predictor[0].__name__,
        )


def test_euler_flow():
    compare_samplers(
        Euler(predictor=FLOW),
        FlowMatchEulerDiscreteScheduler.from_config(  # type: ignore  # Diffusers return BS
            FLOW_CONFIG
        ),
        mu=0.7,
    )


def test_dpm():
    for predictor in [(EPSILON, "epsilon"), (VELOCITY, "v_prediction"), (FLOW, "flow_prediction")]:
        for order in range(1, 3):  # Their third order is fucked up. Turns into barf @ super high steps
            for stochastic in [False, True]:
                compare_samplers(
                    DPM(predictor=predictor[0], order=order, add_noise=stochastic),
                    DPMSolverMultistepScheduler.from_config(  # type: ignore  # Diffusers return BS
                        SCALED_CONFIG,
                        algorithm_type="sde-dpmsolver++" if stochastic else "dpmsolver++",
                        final_sigmas_type="zero",
                        solver_order=order,
                        prediction_type=predictor[1],
                    ),
                    message=f"{predictor[0].__name__} o{order} s{stochastic}",
                )


# # Diffusers ipndm doesnt support anything really. It even explodes in their own pipeline.
# def test_ipndm():
#     compare_samplers(
#         IPNDM(),
#         IPNDMScheduler.from_config(  # type: ignore  # Diffusers return BS
#             SCALED_CONFIG
#         ),
#     )


def test_unipc():
    for predictor in [(EPSILON, "epsilon"), (VELOCITY, "v_prediction"), (FLOW, "flow_prediction")]:
        # technically it can do N order, but diffusers actually breaks down super hard with high order + steps
        # They use torch scalars for everything which accumulates error faster as steps and order increase
        # Considering Diffusers just NaNs out in like half the order as mine, I'm fine with fudging the margins
        for order, margin in zip(range(1, 4), (1e-8, 1e-7, 1e-3)):
            compare_samplers(
                UniPC(predictor=predictor[0], order=order),
                UniPCMultistepScheduler.from_config(  # type: ignore  # Diffusers return BS
                    SCALED_CONFIG,
                    final_sigmas_type="zero",
                    solver_order=order,
                    prediction_type=predictor[1],
                ),
                message=f"{predictor[0].__name__} o{order}",
                margin=margin,
            )
