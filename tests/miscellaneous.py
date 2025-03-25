import random

import numpy as np
import torch
from testing_common import compare_tensors

from skrample.sampling import DPM, IPNDM, Euler, SKSamples, UniPC
from skrample.scheduling import Flow, Scaled


def test_sigmas_to_timesteps():
    for schedule in [Scaled(), Scaled(beta_scale=1), Flow()]:  # base schedules
        timesteps = schedule.timesteps(123)
        timesteps_inv = schedule.sigmas_to_timesteps(schedule.sigmas(123))
        compare_tensors(torch.tensor(timesteps), torch.tensor(timesteps_inv), margin=0)  # shocked this rounds good


def test_sampler_generics():
    eps = 1e-16
    for sampler in Euler(), DPM(order=2), IPNDM(), UniPC(order=3):
        for schedule in Scaled(), Flow():
            i, o = random.random(), random.random()
            prev = [SKSamples(random.random(), random.random(), random.random()) for _ in range(9)]

            scalar = sampler.sample(i, o, schedule.sigmas(10), 4, previous=prev).final

            # Enforce FP64 as that should be equivalent to python scalar
            ndarr = sampler.sample(
                np.array([i], dtype=np.float64),
                np.array([o], dtype=np.float64),
                schedule.sigmas(10),
                4,
                previous=prev,  # type: ignore
            ).final.item()

            tensor = sampler.sample(
                torch.tensor([i], dtype=torch.float64),
                torch.tensor([o], dtype=torch.float64),
                schedule.sigmas(10),
                4,
                previous=prev,  # type: ignore
            ).final.item()

            assert abs(tensor - scalar) < eps
            assert abs(tensor - ndarr) < eps
            assert abs(scalar - ndarr) < eps
