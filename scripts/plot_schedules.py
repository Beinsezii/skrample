#! /usr/bin/env python

from argparse import ArgumentParser
from collections.abc import Generator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import skrample.scheduling as scheduling

SCHEDULES: dict[str, scheduling.ScheduleCommon | scheduling.ScheduleModifier] = {
    "scaled": scheduling.Scaled(uniform=False),
    "scaled_uniform": scheduling.Scaled(),
    "zsnr": scheduling.ZSNR(),
    "linear": scheduling.Linear(),
    "sigcdf": scheduling.SigmoidCDF(),
}

MODIFIERS: dict[str, tuple[type[scheduling.ScheduleModifier], dict[str, Any]] | None] = {
    "beta": (scheduling.Beta, {}),
    "exponential": (scheduling.Exponential, {}),
    "karras": (scheduling.Karras, {}),
    "flow": (scheduling.FlowShift, {}),
    "flow_mu": (scheduling.FlowShift, {"mu": 1}),
    "none": None,
}

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


parser = ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument(
    "--schedule",
    "-s",
    type=str,
    choices=list(SCHEDULES.keys()),
    nargs="+",
    default=["scaled_uniform", "sigcdf"],
)
parser.add_argument(
    "--modifier",
    "-m",
    type=str,
    choices=list(MODIFIERS.keys()),
    nargs="+",
    default=["none", "flow"],
)
parser.add_argument(
    "--modifier_2",
    "-m2",
    type=str,
    choices=list(MODIFIERS.keys()),
    nargs="+",
    default=["none"],
)

args = parser.parse_args()

width, height = 12, 6
plt.figure(figsize=(width, height), facecolor="black", edgecolor="white")

COLORS = colors(6)
for mod1 in args.modifier:
    for mod2 in args.modifier_2:
        for sched_name in args.schedule:
            schedule = SCHEDULES[sched_name]

            composed = schedule
            label: str = sched_name

            for mod_label, (mod_type, mod_props) in [  # type: ignore # Destructure
                m for m in [(mod1, MODIFIERS[mod1]), (mod2, MODIFIERS[mod2])] if m[1]
            ]:
                composed = mod_type(schedule, **mod_props)
                label += "_" + mod_label

            label = " ".join([s.capitalize() for s in label.split("_")])

            data = np.concatenate([composed.schedule(args.steps), [[0, 0]]], dtype=np.float64)

            timesteps = data[:, 0] / composed.base_timesteps
            sigmas = data[:, 1] / data[:, 1].max()

            plt.plot(timesteps, label=label + " Timesteps", marker="+", color=next(COLORS))
            if not np.allclose(timesteps, sigmas, atol=1e-2):
                plt.plot(sigmas, label=label + " Sigmas", marker="+", color=next(COLORS))

plt.xlabel("Step")
plt.ylabel("Normalized Values")
plt.title("Skrample Schedules")
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
