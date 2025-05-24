import math
import random
from dataclasses import replace

import numpy as np
import torch
from testing_common import compare_tensors

from skrample.common import sigma_complement
from skrample.diffusers import SkrampleWrapperScheduler
from skrample.sampling import (
    DPM,
    SPC,
    Adams,
    Euler,
    HighOrderSampler,
    SkrampleSampler,
    SKSamples,
    StochasticSampler,
    UniPC,
)
from skrample.scheduling import Beta, FlowShift, Karras, Linear, Scaled, SigmoidCDF

ALL_SAMPLERS = [
    Adams,
    DPM,
    Euler,
    SPC,
    UniPC,
]

ALL_SCHEDULES = [
    Linear,
    Scaled,
    SigmoidCDF,
]

ALL_MODIFIERS = [
    Beta,
    FlowShift,
    Karras,
]


def test_sigmas_to_timesteps() -> None:
    for schedule in [*(cls() for cls in ALL_SCHEDULES), Scaled(beta_scale=1)]:  # base schedules
        timesteps = schedule.timesteps(123)
        timesteps_inv = schedule.sigmas_to_timesteps(schedule.sigmas(123))
        compare_tensors(torch.tensor(timesteps), torch.tensor(timesteps_inv), margin=0)  # shocked this rounds good


def test_sampler_generics() -> None:
    eps = 1e-12
    for sampler in [
        *(cls() for cls in ALL_SAMPLERS),
        *(cls(order=cls.max_order()) for cls in ALL_SAMPLERS if issubclass(cls, HighOrderSampler)),
    ]:
        for schedule in Scaled(), FlowShift(Linear()):
            i, o = random.random(), random.random()
            prev = [SKSamples(random.random(), random.random(), random.random()) for _ in range(9)]

            scalar = sampler.sample(i, o, 4, schedule.sigmas(10), schedule.sigma_transform, previous=prev).final

            # Enforce FP64 as that should be equivalent to python scalar
            ndarr = sampler.sample(
                np.array([i], dtype=np.float64),
                np.array([o], dtype=np.float64),
                4,
                schedule.sigmas(10),
                schedule.sigma_transform,
                previous=prev,  # type: ignore
            ).final.item()

            tensor = sampler.sample(
                torch.tensor([i], dtype=torch.float64),
                torch.tensor([o], dtype=torch.float64),
                4,
                schedule.sigmas(10),
                schedule.sigma_transform,
                previous=prev,  # type: ignore
            ).final.item()

            assert abs(tensor - scalar) < eps
            assert abs(tensor - ndarr) < eps
            assert abs(scalar - ndarr) < eps


def test_mu_set() -> None:
    mu = 1.2345
    a = SkrampleWrapperScheduler(DPM(), Beta(FlowShift(Karras(Linear()))))
    b = SkrampleWrapperScheduler(DPM(), Beta(FlowShift(Karras(Linear()), shift=math.exp(mu))))
    a.set_timesteps(1, mu=mu)
    assert a.schedule == b.schedule


def test_require_previous() -> None:
    samplers: list[SkrampleSampler] = []
    for cls in ALL_SAMPLERS:
        if issubclass(cls, HighOrderSampler):
            samplers.extend([cls(order=o + 1) for o in range(cls.min_order(), cls.max_order())])
        else:
            samplers.append(cls())

    for o1 in range(1, 4):
        for o2 in range(1, 4):
            samplers.append(UniPC(order=o1, solver=Adams(order=o2)))
            samplers.append(SPC(predictor=Adams(order=o1), corrector=Adams(order=o2)))

    for sampler in samplers:
        sample = 1.5
        prediction = 0.5
        previous = tuple(SKSamples(n / 2, n * 2, n * 1.5) for n in range(100))

        a = sampler.sample(
            sample,
            prediction,
            31,
            Linear().sigmas(100),
            sigma_complement,
            None,
            previous,
        )
        b = sampler.sample(
            sample,
            prediction,
            31,
            Linear().sigmas(100),
            sigma_complement,
            None,
            previous[len(previous) - sampler.require_previous :],
        )

        assert a == b, (sampler, sampler.require_previous)


def test_require_noise() -> None:
    samplers: list[SkrampleSampler] = []
    for cls in ALL_SAMPLERS:
        if issubclass(cls, StochasticSampler):
            samplers.extend([cls(add_noise=n) for n in (False, True)])
        else:
            samplers.append(cls())

    for n1 in (False, True):
        for n2 in (False, True):
            samplers.append(UniPC(solver=DPM(add_noise=n2)))
            samplers.append(SPC(predictor=DPM(add_noise=n1), corrector=DPM(add_noise=n2)))

    for sampler in samplers:
        sample = 1.5
        prediction = 0.5
        previous = tuple(SKSamples(n / 2, n * 2, n * 1.5) for n in range(100))
        noise = -0.5

        a = sampler.sample(
            sample,
            prediction,
            31,
            Linear().sigmas(100),
            sigma_complement,
            noise,
            previous,
        )
        b = sampler.sample(
            sample,
            prediction,
            31,
            Linear().sigmas(100),
            sigma_complement,
            noise if sampler.require_noise else None,
            previous,
        )

        # Don't compare stored noise since it's expected diff
        b = replace(b, noise=a.noise)

        assert a == b, (sampler, sampler.require_noise)
