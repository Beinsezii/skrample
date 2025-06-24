#! /usr/bin/env python

from time import perf_counter_ns

import torch

from skrample.diffusers import SkrampleWrapperScheduler
from skrample.sampling import DPM
from skrample.scheduling import Beta, FlowShift, SigmoidCDF


def bench_wrapper() -> int:
    wrapper = SkrampleWrapperScheduler(DPM(), Beta(FlowShift(SigmoidCDF())))
    wrapper.set_timesteps(1000)

    clock = perf_counter_ns()
    for timestep in wrapper.timesteps:
        output, sample = torch.rand([1]), torch.rand([1])
        wrapper.step(output, timestep, sample, return_dict=False)

    return perf_counter_ns() - clock


if __name__ == "__main__":
    for run in range(5):
        print(bench_wrapper())
