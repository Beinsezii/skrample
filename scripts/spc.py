#! /usr/bin/env python

import dataclasses
import math
from argparse import ArgumentParser, FileType
from dataclasses import replace
from random import random

import numpy as np
from numpy.typing import NDArray

import skrample.sampling as sampling
import skrample.scheduling as scheduling
from skrample.common import SigmaTransform, sigma_complement, sigma_polar

parser = ArgumentParser()
parser.add_argument("out", type=FileType("w"))
parser.add_argument("--steps", "-s", type=int, default=[10, 20, 30], nargs="+")
parser.add_argument("--curves", "-k", type=int, default=[10, 20, 30], nargs="+")

args = parser.parse_args()


@dataclasses.dataclass(frozen=True)
class Row:
    pe: str
    ce: str
    t: str
    k: int
    h: int
    e: float
    e2: float


def sample_model(
    sampler: sampling.SkrampleSampler, schedule: NDArray[np.float64], curve: int, transform: SigmaTransform
) -> NDArray:
    previous: list[sampling.SKSamples] = []
    sample = 1.0
    sampled_values = [sample]
    for step, sigma in enumerate(schedule):
        result = sampler.sample(
            sample=sample,
            prediction=math.sin(sigma * curve),
            step=step,
            sigma_schedule=schedule,
            sigma_transform=transform,
            previous=tuple(previous),
            noise=random(),
        )
        previous.append(result)
        sample = result.final
        sampled_values.append(sample)
    return np.array(sampled_values)


samplers: set[sampling.SkrampleSampler] = {sampling.Euler(), sampling.Adams(order=2), sampling.DPM(order=2)}
for v in samplers.copy():
    if isinstance(v, sampling.HighOrderSampler):
        for o in range(2, v.max_order() + 1):
            samplers.add(replace(v, order=o))

schedule = scheduling.Linear(base_timesteps=10_000)

table: list[Row] = []
for t in [sigma_polar, sigma_complement]:
    for k in args.curves:
        reference = sample_model(sampling.Euler(), schedule.sigmas(schedule.base_timesteps), k, t)
        for h in args.steps:
            reference_aliased = np.interp(np.linspace(0, 1, h + 1), np.linspace(0, 1, len(reference)), reference)
            for pe in samplers:
                for ce in samplers:
                    spc = sampling.SPC(predictor=pe, corrector=ce)
                    sampled = sample_model(spc, schedule.sigmas(h), k, t)
                    table.append(
                        Row(
                            type(pe).__name__ + (str(pe.order) if isinstance(pe, sampling.HighOrderSampler) else ""),
                            type(ce).__name__ + (str(ce.order) if isinstance(ce, sampling.HighOrderSampler) else ""),
                            t.__name__,
                            k,
                            h,
                            abs(sampled - reference_aliased).mean().item(),
                            (abs(sampled - reference_aliased + 1) ** 2 - 1).mean().item(),
                        )
                    )

table = sorted(
    sorted(sorted(sorted(table, key=lambda r: r.e), key=lambda r: r.h), key=lambda r: r.k), key=lambda r: r.t
)

args.out.write(",".join(f.name for f in dataclasses.fields(Row)) + "\n")
for row in table:
    args.out.write(",".join(str(v) for v in dataclasses.astuple(row)) + "\n")
