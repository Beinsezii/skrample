import random

import numpy as np
import torch
from testing_common import compare_tensors

from skrample.diffusers import SkrampleWrapperScheduler
from skrample.sampling import DPM, Adams, Euler, SKSamples, UniPC
from skrample.scheduling import Beta, FlowShift, Karras, Linear, Scaled


def test_sigmas_to_timesteps() -> None:
    for schedule in [Scaled(), Scaled(beta_scale=1), FlowShift(Linear())]:  # base schedules
        timesteps = schedule.timesteps(123)
        timesteps_inv = schedule.sigmas_to_timesteps(schedule.sigmas(123))
        compare_tensors(torch.tensor(timesteps), torch.tensor(timesteps_inv), margin=0)  # shocked this rounds good


def test_sampler_generics() -> None:
    eps = 1e-12
    for sampler in Euler(), DPM(order=2), Adams(), UniPC(order=3):
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
    b = SkrampleWrapperScheduler(DPM(), Beta(FlowShift(Karras(Linear()), mu=mu)))
    a.set_timesteps(1, mu=mu)
    assert a.schedule == b.schedule
