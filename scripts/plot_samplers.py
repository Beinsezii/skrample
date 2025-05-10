#! /usr/bin/env python

import math
from argparse import ArgumentParser
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from random import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import skrample.sampling as sampling
from skrample.common import sigma_complement
from skrample.scheduling import Linear

OKLAB_XYZ_M1 = np.array(
    [
        [0.41217385, 0.21187214, 0.08831541],
        [0.53629746, 0.68074768, 0.28186631],
        [0.05146303, 0.10740646, 0.63026345],
    ]
)
OKLAB_M2 = np.array(
    [
        [0.2104542553, 1.9779984951, 0.0259040371],
        [0.7936177850, -2.4285922050, 0.7827717662],
        [-0.0040720468, 0.4505937099, -0.8086757660],
    ]
)


def spowf(array: NDArray[np.float64], power: int | float | list[int | float]) -> NDArray[np.float64]:
    return np.copysign(np.abs(array) ** power, array)


def oklch_to_srgb(array: NDArray[np.float64]) -> list[float]:
    oklab = np.stack(
        [array[0], array[1] * np.cos(np.deg2rad(array[2])), array[1] * np.sin(np.deg2rad(array[2]))],
        axis=0,
    )
    lrgb = spowf((oklab @ np.linalg.inv(OKLAB_M2)), 3) @ np.linalg.inv(OKLAB_XYZ_M1)
    srgb = spowf(lrgb, 1 / 2.2)
    return srgb.clip(0, 1).tolist()  # type: ignore


def colors(hue_steps: int) -> Generator[list[float]]:
    for offset in 0, 1:
        for lightness, chroma in [
            (0.6, 0.6),
            (0.8, 0.4),
            (0.4, 0.4),
            (0.8, 0.8),
            (0.4, 0.8),
        ]:
            lighness_actual = lightness * (0.9 - 0.25) + 0.25  # offset by approximate quantiles of srgb clip
            chroma_actual = chroma * 0.25
            for hue in range(15 + int(offset * 360 / hue_steps / 2), 360, int(360 / hue_steps)):
                yield oklch_to_srgb(np.array([lighness_actual, chroma_actual, hue], dtype=np.float64))


COLORS = colors(6)
width, height = 12, 6
plt.figure(figsize=(width, height), facecolor="black", edgecolor="white")

###

plt.xlim(1, 0)
plt.xlabel("Schedule")
plt.ylabel("Sample")

SAMPLERS: dict[str, sampling.SkrampleSampler] = {
    "euler": sampling.Euler(),
    "adams": sampling.Adams(),
    "dpm": sampling.DPM(),
    "unipc": sampling.UniPC(),
}
for k, v in list(SAMPLERS.items()):
    if isinstance(v, sampling.HighOrderSampler):
        for o in range(v.max_order):
            if o != v.order:
                SAMPLERS[k + str(o)] = replace(v, order=o)

parser = ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("-s", "--steps", type=int, default=20)
parser.add_argument("-k", "--scale", type=int, default=10)
parser.add_argument(
    "--sampler",
    "-S",
    type=str,
    choices=list(SAMPLERS.keys()),
    nargs="+",
    default=["euler", "adams", "dpm2", "unipc"],
)

args = parser.parse_args()


def sample_model(sampler: sampling.SkrampleSampler, schedule: NDArray[np.float64]) -> list[float]:
    previous: list[sampling.SKSamples] = []
    sample = 1.0
    sampled_values = [sample]
    for step, sigma in enumerate(schedule):
        # Run sampler
        result = sampler.sample(
            sample=sample,
            prediction=math.sin(sigma * args.scale),
            step=step,
            sigma_schedule=schedule,
            sigma_transform=sigma_complement,
            previous=previous,
            noise=random(),
        )
        previous.append(result)
        sample = result.final
        sampled_values.append(sample)
    return sampled_values


schedule = Linear(base_timesteps=10_000)

plt.plot(
    [*schedule.sigmas(schedule.base_timesteps), 0],
    sample_model(sampling.Euler(), schedule.sigmas(schedule.base_timesteps)),
    label="Reference",
    color=next(COLORS),
)

for sampler in [SAMPLERS[s] for s in args.sampler]:
    sigmas = schedule.sigmas(args.steps)
    label = type(sampler).__name__
    if isinstance(sampler, sampling.HighOrderSampler) and sampler.order != type(sampler).order:
        label += " " + str(sampler.order)
    plt.plot([*sigmas, 0], sample_model(sampler, sigmas), label=label, color=next(COLORS), linestyle="--")


###

ax = plt.gca()
ax.set(facecolor="black")
ax.grid(color="white")

ax.tick_params(axis="both", which="both", color="white")

ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")

[v.set_color("white") for v in list(ax.spines.values()) + ax.get_xticklabels() + ax.get_yticklabels()]

ax.legend(facecolor="black", labelcolor="white", edgecolor="gray")

plt.savefig(args.file, dpi=max(1920 / width, 1080 / height))
