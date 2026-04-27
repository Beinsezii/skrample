#! /usr/bin/env python

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from skrample import scheduling
from skrample.analytics import plotting
from skrample.analytics.common import OscDecay
from skrample.sampling import functional, models, structured, traits

SPACES: dict[str, tuple[float, scheduling.SigmaSpace, models.DiffusionModel]] = {
    "vp": (1.0, scheduling.VariancePreserving(), models.NoiseModel()),
    "fm": (1.0, scheduling.FlowMatching(), models.FlowModel()),
}
SAMPLERS: dict[str, structured.StructuredSampler | functional.FunctionalSampler] = {
    "euler": structured.Euler(),
    "adams": structured.Adams(),
    "dpm": structured.DPM(),
    "unip": structured.UniP(),
    "unipc": structured.UniPC(),
    "spc": structured.SPC(),
    "rku": functional.RKUltra(),
    "ssprk": functional.RKUltra(providers={**functional.DEFAULT_PROVIDERS, **functional.STABLE_PROVIDERS}),
    "dynrk": functional.DynasauRK(),
    "rkm": functional.RKMoire(),
}
for k, v in list(SAMPLERS.items()):
    if isinstance(v, traits.HigherOrder):
        for o in range(v.min_order(), min(v.max_order() + 1, 9)):
            if o != v.order:
                SAMPLERS[k + str(o)] = replace(v, order=o)

SCHEDULES: dict[str, scheduling.ScheduleCommon] = {
    "scaled": scheduling.Scaled(),
    "zsnr": scheduling.ZSNR(),
    "linear": scheduling.Linear(),
}
SUBSCHEDULES: dict[str, tuple[type[scheduling.SubSchedule], dict[str, Any]]] = {
    "beta": (scheduling.Beta, {}),
    "exponential": (scheduling.Exponential, {}),
    "karras": (scheduling.Karras, {}),
    "probit": (scheduling.Probit, {}),
    "none": (scheduling.NoSub, {}),
}
MODIFIERS: dict[str, tuple[type[scheduling.ScheduleModifier], dict[str, Any]]] = {
    "flow": (scheduling.FlowShift, {}),
    "hyper": (scheduling.Hyper, {}),
    "vyper": (scheduling.Hyper, {"scale": -2}),
    "hype": (scheduling.Hyper, {"tail": False}),
    "vype": (scheduling.Hyper, {"scale": -2, "tail": False}),
    "sinner": (scheduling.Sinner, {}),
    "pinner": (scheduling.Sinner, {"scale": -scheduling.Sinner.scale}),
    "none": (scheduling.NoMod, {}),
}


# Common
parser = ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("--steps", "-s", type=int, default=25)
subparsers = parser.add_subparsers(dest="command")

# Samplers
parser_sampler = subparsers.add_parser("samplers")
parser_sampler.add_argument("--adjust", type=bool, default=True, action=BooleanOptionalAction)
parser_sampler.add_argument("--curve", "-k", type=int, default=OscDecay.scale)
parser_sampler.add_argument("--transform", "-t", type=str, choices=list(SPACES.keys()), default="fm")
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
parser_schedule.add_argument("--alphas", type=bool, default=False, action=BooleanOptionalAction)
parser_schedule.add_argument("--timesteps", type=bool, default=False, action=BooleanOptionalAction)
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

if args.command == "samplers":
    base_steps: int = 10_000

    Image.fromarray(
        plotting.draw(
            plotting.plot_samplers(
                [SAMPLERS[s] for s in args.sampler],
                scheduling.Hyper(
                    scheduling.Linear(
                        sigma_start=SPACES[args.transform][0],
                        base_timesteps=base_steps,
                        custom_space=SPACES[args.transform][1],
                    ),
                    tail=False,
                ),
                SPACES[args.transform][2],
                OscDecay(scale=args.curve),
                steps=args.steps,
                reference_steps=base_steps,
                adjust_steps=args.adjust,
            )
        )
    ).save(args.file, format="PNG", compress_level=1)


elif args.command == "schedules":
    schedules: list[tuple[scheduling.SkrampleSchedule, str]] = []
    for sched_name in args.schedule:
        for sub in args.subschedule:
            for mod1 in args.modifier:
                for mod2 in args.modifier_2:
                    schedule = SCHEDULES[sched_name]

                    composed = schedule
                    label: str = sched_name

                    if (subschedule := SUBSCHEDULES[sub]) and not issubclass(subschedule[0], scheduling.NoSub):
                        composed = subschedule[0](composed, **subschedule[1])
                        label += "_" + subschedule[0].__name__.lower()

                    for mod_label, (mod_type, mod_props) in [  # pyright: ignore # Destructure
                        m
                        for m in [(mod1, MODIFIERS[mod1]), (mod2, MODIFIERS[mod2])]
                        if not issubclass(m[1][0], scheduling.NoMod)
                    ]:
                        composed = mod_type(composed, **mod_props)
                        label += "_" + mod_label

                    label = " ".join([s.capitalize() for s in label.split("_")])

                    schedules.append((composed, label))

    Image.fromarray(
        plotting.draw(
            plotting.plot_schedules(
                schedules,
                args.steps,
                args.alphas,
                args.timesteps,
            )
        )
    ).save(args.file, format="PNG", compress_level=1)
