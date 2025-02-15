import torch
from testing_common import compare_tensors

from skrample.scheduling import Flow, Scaled


def test_sigmas_to_timesteps():
    for schedule in [Scaled(), Scaled(scale=1), Flow()]:  # base schedules
        timesteps = schedule.timesteps(123)
        timesteps_inv = schedule.sigmas_to_timesteps(schedule.sigmas(123))
        compare_tensors(torch.tensor(timesteps), torch.tensor(timesteps_inv), margin=0)  # shocked this rounds good
