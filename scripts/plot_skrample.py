#! /usr/bin/env python

import math
from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from random import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from skrample import scheduling
from skrample.common import SigmaTransform, sigma_complement, sigma_polar, spowf
from skrample.sampling import functional, models, structured
from skrample.sampling.interface import StructuredFunctionalAdapter

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


TRANSFORMS: dict[str, tuple[float, SigmaTransform, models.DiffusionModel]] = {
    "polar": (1.0, sigma_polar, models.NoiseModel()),
    "complement": (1.0, sigma_complement, models.FlowModel()),
}
SAMPLERS: dict[str, structured.StructuredSampler | functional.FunctionalSampler] = {
    "euler": structured.Euler(),
    "adams": structured.Adams(),
    "dpm": structured.DPM(),
    "unip": structured.UniP(),
    "unipc": structured.UniPC(),
    "spc": structured.SPC(),
    "rku": functional.RKUltra(),
    "rkm": functional.RKMoire(),
    "fheun": functional.FastHeun(),
}
for k, v in list(SAMPLERS.items()):
    if isinstance(v, structured.StructuredMultistep | functional.FunctionalHigher):
        for o in range(v.min_order(), min(v.max_order() + 1, 9)):
            if o != v.order:
                SAMPLERS[k + str(o)] = replace(v, order=o)

SCHEDULES: dict[str, scheduling.ScheduleCommon] = {
    "scaled": scheduling.Scaled(),
    "zsnr": scheduling.ZSNR(),
    "linear": scheduling.Linear(),
}
SUBSCHEDULES: dict[str, tuple[type[scheduling.SubSchedule], dict[str, Any]] | None] = {
    "beta": (scheduling.Beta, {}),
    "exponential": (scheduling.Exponential, {}),
    "karras": (scheduling.Karras, {}),
    "sigauss": (scheduling.Siggauss, {}),
    "none": None,
}
MODIFIERS: dict[str, tuple[type[scheduling.ScheduleModifier], dict[str, Any]] | None] = {
    "flow": (scheduling.FlowShift, {}),
    "hyper": (scheduling.Hyper, {}),
    "vyper": (scheduling.Hyper, {"scale": -2}),
    "hype": (scheduling.Hyper, {"tail": False}),
    "vype": (scheduling.Hyper, {"scale": -2, "tail": False}),
    "sinner": (scheduling.Sinner, {}),
    "pinner": (scheduling.Sinner, {"scale": -scheduling.Sinner.scale}),
    "none": None,
}


# Common
parser = ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("--steps", "-s", type=int, default=25)
subparsers = parser.add_subparsers(dest="command")

# Samplers
parser_sampler = subparsers.add_parser("samplers")
parser_sampler.add_argument("--adjust", type=bool, default=True, action=BooleanOptionalAction)
parser_sampler.add_argument("--curve", "-k", type=int, default=30)
parser_sampler.add_argument("--transform", "-t", type=str, choices=list(TRANSFORMS.keys()), default="polar")
parser_sampler.add_argument(
    "--sampler",
    "-S",
    type=str,
    nargs="+",
    choices=list(SAMPLERS.keys()),
    default=["euler", "adams", "dpm", "unipc", "spc"],
)

# Schedules
parser_schedule = subparsers.add_parser("schedules")
parser_schedule.add_argument(
    "--schedule",
    "-S",
    type=str,
    choices=list(SCHEDULES.keys()),
    nargs="+",
    default=["linear"],
)
parser_schedule.add_argument(
    "--subschedule",
    "-S2",
    type=str,
    choices=list(SUBSCHEDULES.keys()),
    nargs="+",
    default=["none"],
)
parser_schedule.add_argument(
    "--modifier",
    "-m",
    type=str,
    choices=list(MODIFIERS.keys()),
    nargs="+",
    default=["none", "flow", "hyper"],
)
parser_schedule.add_argument(
    "--modifier_2",
    "-m2",
    type=str,
    choices=list(MODIFIERS.keys()),
    nargs="+",
    default=["none"],
)

args = parser.parse_args()
COLORS = colors(6)
width, height = 12, 6
plt.figure(figsize=(width, height), facecolor="black", edgecolor="white")

if args.command == "samplers":
    plt.xlim(1, 0)
    plt.xlabel("Schedule")
    plt.ylabel("Sample")
    plt.title("Skrample Samplers")

    schedule = scheduling.Hyper(
        scheduling.Linear(
            sigma_start=TRANSFORMS[args.transform][0],
            base_timesteps=10_000,
            custom_transform=TRANSFORMS[args.transform][1],
        ),
        -2,
        False,
    )

    def sample_model(
        sampler: structured.StructuredSampler | functional.FunctionalSampler, steps: int
    ) -> tuple[list[float], list[float]]:
        if isinstance(sampler, structured.StructuredSampler):
            sampler = StructuredFunctionalAdapter(sampler)

        sample = 1.0
        sampled_values = [sample]
        timesteps = [0.0]

        def callback(x: float, n: int, t: float, s: float) -> None:
            nonlocal sampled_values, timesteps
            sampled_values.append(x)
            timesteps.insert(-1, t / schedule.base_timesteps)

        if isinstance(sampler, functional.RKMoire) and args.adjust:
            adjusted = schedule.base_timesteps
        elif isinstance(sampler, functional.FunctionalHigher) and args.adjust:
            adjusted = sampler.adjust_steps(steps)
        else:
            adjusted = steps

        sampler.sample_model(
            sample=sample,
            model=lambda x, t, s: x - math.sin(t / schedule.base_timesteps * args.curve),
            model_transform=TRANSFORMS[args.transform][2],
            schedule=schedule,
            steps=adjusted,
            rng=random,
            callback=callback,
        )

        return timesteps, sampled_values

    plt.plot(*sample_model(structured.Euler(), schedule.base_timesteps), label="Reference", color=next(COLORS))

    for sampler in [SAMPLERS[s] for s in args.sampler]:
        label = type(sampler).__name__
        if (
            isinstance(sampler, structured.StructuredMultistep | functional.FunctionalHigher)
            and sampler.order != type(sampler).order
        ):
            label += " " + str(sampler.order)
        plt.plot(*sample_model(sampler, args.steps), label=label, color=next(COLORS), linestyle="--")

elif args.command == "schedules":
    plt.xlabel("Step")
    plt.ylabel("Normalized Values")
    plt.title("Skrample Schedules")

    for sched_name in args.schedule:
        for sub in args.subschedule:
            for mod1 in args.modifier:
                for mod2 in args.modifier_2:
                    schedule = SCHEDULES[sched_name]

                    composed = schedule
                    label: str = sched_name

                    if (subschedule := SUBSCHEDULES[sub]) and sub is not None:
                        composed = subschedule[0](composed, **subschedule[1])
                        label += "_" + subschedule[0].__name__.lower()

                    for mod_label, (mod_type, mod_props) in [  # type: ignore # Destructure
                        m for m in [(mod1, MODIFIERS[mod1]), (mod2, MODIFIERS[mod2])] if m[1]
                    ]:
                        composed = mod_type(composed, **mod_props)
                        label += "_" + mod_label

                    label = " ".join([s.capitalize() for s in label.split("_")])

                    data = composed.ipoints(np.linspace(0, 1, args.steps + 1))

                    timesteps = data[:, 0] / composed.base_timesteps
                    sigmas = data[:, 1] / data[:, 1].max()

                    marker = "+" if args.steps <= 50 else ""
                    plt.plot(timesteps, label=label + " Timesteps", marker=marker, color=next(COLORS))
                    if not np.allclose(timesteps, sigmas, atol=1e-2):
                        plt.plot(sigmas, label=label + " Sigmas", marker=marker, color=next(COLORS))

else:
    raise NotImplementedError


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
