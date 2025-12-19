import itertools
from inspect import signature

import pytest
import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from testing_common import FLOW_CONFIG, SCALED_CONFIG, compare_tensors

from skrample.common import SigmaTransform, sigma_complement, sigma_polar
from skrample.sampling.functional import RKUltra
from skrample.sampling.models import DiffusionModel, FlowModel, NoiseModel, VelocityModel
from skrample.sampling.structured import DPM, Euler, SKSamples, StructuredSampler, UniPC
from skrample.sampling.tableaux import RK2
from skrample.scheduling import FixedSchedule

# TODO (beinsezii): no idea why this is touchy???
SCALED_CONFIG = SCALED_CONFIG | {"timestep_spacing": "leading"}

DiffusersScheduler = (
    EulerDiscreteScheduler
    | DPMSolverMultistepScheduler
    | FlowMatchEulerDiscreteScheduler
    | FlowMatchHeunDiscreteScheduler
    | UniPCMultistepScheduler
)

EPSILON = NoiseModel()
FLOW = FlowModel()
VELOCITY = VelocityModel()


def fake_model(t: torch.Tensor) -> torch.Tensor:
    t @= torch.randn(t.shape, generator=torch.Generator(t.device).manual_seed(-1), dtype=t.dtype)
    return t / t.std()  # keep values in sane range


def dual_sample(
    a: StructuredSampler,
    b: DiffusersScheduler,
    model_transform: DiffusionModel,
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
    schedule = torch.stack([b.timesteps, b.sigmas[:-1]], dim=1)
    timestep, sigma = schedule[steps.start]

    if isinstance(b, FlowMatchEulerDiscreteScheduler):
        b_sample = b.scale_noise(sample=b_sample, timestep=timestep.unsqueeze(0), noise=initial_noise)
    else:
        b_sample = b.add_noise(original_samples=b_sample, noise=initial_noise, timesteps=timestep.unsqueeze(0))

    sigma_transform = sigma_complement if isinstance(model_transform, FlowModel) else sigma_polar

    a_sample = a.merge_noise(a_sample, initial_noise, sigma.item(), sigma_transform)

    prior_steps: list[SKSamples] = []
    for step in steps:
        # Just some pseud-random transform that shouldn't blow up the values
        noise = torch.randn(a_sample.shape, generator=seed.clone_state(), dtype=a_sample.dtype)

        timestep, sigma = schedule[step]

        a_output = model_transform.to_x(
            a_sample, fake_model(a.scale_input(a_sample, sigma.item(), sigma_transform)), sigma.item(), sigma_transform
        )
        sampled = a.sample(a_sample, a_output, step, schedule.numpy().tolist(), sigma_transform, noise, prior_steps)
        a_sample = sampled.final
        prior_steps.append(sampled)

        if isinstance(b, FlowMatchEulerDiscreteScheduler):
            b_output = fake_model(b_sample)
        else:
            b_output = fake_model(b.scale_model_input(sample=b_sample, timestep=timestep))

        if "generator" in signature(b.step).parameters:  # why, diffusers, why
            b_sample = b.step(model_output=b_output, sample=b_sample, timestep=timestep, generator=seed)[0]
        else:
            b_sample = b.step(model_output=b_output, sample=b_sample, timestep=timestep)[0]

    return a_sample, b_sample


STEP_RANGES = [range(0, 2), range(0, 11), range(0, 201), range(3, 6), range(2, 23), range(31, 200)]


@pytest.mark.parametrize(
    ("predictor", "steps"),
    itertools.product([(EPSILON, "epsilon"), (VELOCITY, "v_prediction")], STEP_RANGES),
)
def test_euler(predictor: tuple[DiffusionModel, str], steps: range) -> None:
    compare_tensors(
        *dual_sample(
            Euler(),
            EulerDiscreteScheduler.from_config(
                SCALED_CONFIG,
                prediction_type=predictor[1],
            ),
            predictor[0],
            steps,
        )
    )


@pytest.mark.parametrize(
    ("predictor", "steps"),
    itertools.product([(EPSILON, "epsilon"), (VELOCITY, "v_prediction")], STEP_RANGES),
)
def test_euler_ancestral(predictor: tuple[DiffusionModel, str], steps: range) -> None:
    compare_tensors(
        *dual_sample(
            DPM(order=1, add_noise=True),
            EulerAncestralDiscreteScheduler.from_config(
                SCALED_CONFIG,
                prediction_type=predictor[1],
            ),
            predictor[0],
            steps,
        )
    )


@pytest.mark.parametrize("steps", STEP_RANGES)
def test_euler_flow(steps: range) -> None:
    compare_tensors(
        *dual_sample(
            Euler(),
            FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG),
            FLOW,
            steps,
            mu=0.7,
        )
    )


@pytest.mark.parametrize(
    ("predictor", "order", "stochastic", "steps"),
    itertools.product(
        [(EPSILON, "epsilon"), (VELOCITY, "v_prediction"), (FLOW, "flow_prediction")],
        range(1, 3),  # Their third order is fucked up. Turns into barf @ super high steps
        (False, True),
        STEP_RANGES,
    ),
)
def test_dpm(predictor: tuple[DiffusionModel, str], order: int, stochastic: bool, steps: range) -> None:
    compare_tensors(
        *dual_sample(
            DPM(order=order, add_noise=stochastic),
            DPMSolverMultistepScheduler.from_config(
                SCALED_CONFIG,
                algorithm_type="sde-dpmsolver++" if stochastic else "dpmsolver++",
                final_sigmas_type="zero",
                solver_order=order,
                prediction_type=predictor[1],
                use_flow_sigmas=predictor[0] == FLOW,
            ),
            predictor[0],
            steps,
        )
    )


@pytest.mark.parametrize(
    ("predictor", "order", "steps"),
    itertools.product(
        [(EPSILON, "epsilon"), (VELOCITY, "v_prediction"), (FLOW, "flow_prediction")],
        # technically it can do N order, but diffusers actually breaks down super hard with high order + steps
        # They use torch scalars for everything which accumulates error faster as steps and order increase
        # Considering Diffusers just NaNs out in like half the order as mine, I'm fine with fudging the margins
        range(1, 5),
        STEP_RANGES,
    ),
)
def test_unipc(predictor: tuple[DiffusionModel, str], order: int, steps: range) -> None:
    compare_tensors(
        *dual_sample(
            UniPC(order=order, fast_solve=True),
            UniPCMultistepScheduler.from_config(
                SCALED_CONFIG,
                final_sigmas_type="zero",
                solver_order=order,
                prediction_type=predictor[1],
                use_flow_sigmas=predictor[0] == FLOW,
            ),
            predictor[0],
            steps,
        )
    )


@pytest.mark.parametrize(
    ("model_transform", "derivative_transform", "sigma_transform", "diffusers_scheduler", "steps"),
    (
        (mt, dt, st, ds, s)
        for (mt, dt, st, ds), s in itertools.product(
            (
                (
                    NoiseModel(),
                    NoiseModel(),
                    sigma_polar,
                    HeunDiscreteScheduler.from_config(SCALED_CONFIG, prediction_type="epsilon"),
                ),
                (
                    VelocityModel(),
                    NoiseModel(),
                    sigma_polar,
                    HeunDiscreteScheduler.from_config(SCALED_CONFIG, prediction_type="v_prediction"),
                ),
                (
                    FlowModel(),
                    FlowModel(),
                    sigma_complement,
                    FlowMatchHeunDiscreteScheduler.from_config(FLOW_CONFIG),
                ),
            ),
            (2, 3, 30, 31, 200, 201),
        )
    ),
)
def test_heun(
    model_transform: DiffusionModel,
    derivative_transform: DiffusionModel,
    sigma_transform: SigmaTransform,
    diffusers_scheduler: HeunDiscreteScheduler | FlowMatchHeunDiscreteScheduler,
    steps: int,
) -> None:
    diffusers_scheduler.set_timesteps(steps)

    skrample_schedule = FixedSchedule(
        list(zip(diffusers_scheduler.timesteps.tolist(), diffusers_scheduler.sigmas.tolist())),
        sigma_transform,
    )
    skrample_sampler = RKUltra(
        order=2,
        providers={2: RK2.Heun},
        derivative_transform=derivative_transform,
    )

    sk_sample = torch.zeros([1, 4, 128, 128], dtype=torch.float32)
    seed = torch.manual_seed(0)

    df_noise = torch.randn(sk_sample.shape, generator=seed.clone_state(), dtype=sk_sample.dtype)
    # df_sample = df.add_noise(sk_sample.clone(), df_noise, df.timesteps[0:1])

    df_sample = df_noise.clone()
    if isinstance(diffusers_scheduler, HeunDiscreteScheduler):
        df_sample *= diffusers_scheduler.init_noise_sigma

    for t in diffusers_scheduler.timesteps:
        model_input: torch.Tensor = df_sample
        if isinstance(diffusers_scheduler, HeunDiscreteScheduler):
            model_input = diffusers_scheduler.scale_model_input(df_sample, timestep=t)

        df_sample: torch.Tensor = diffusers_scheduler.step(
            fake_model(model_input),  # pyright: ignore [reportArgumentType]
            sample=df_sample,  # pyright: ignore [reportArgumentType]
            timestep=t,  # pyright: ignore [reportArgumentType]
        )[0]

    sk_sample = skrample_sampler.generate_model(
        lambda x, t, s: fake_model(x),
        model_transform,
        skrample_schedule,
        lambda: torch.randn(sk_sample.shape, generator=seed, dtype=sk_sample.dtype),
        steps,
        initial=sk_sample,
    )

    compare_tensors(df_sample, sk_sample, margin=1e-8)
