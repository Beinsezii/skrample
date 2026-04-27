#! /usr/bin/env python

import contextlib
import random
from collections.abc import Generator, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from skrample import scheduling
from skrample.common import DeltaPoint, spowf
from skrample.sampling import functional, models, structured, tableaux, traits
from skrample.sampling.interface import StructuredFunctionalAdapter
from skrample.sampling.models import DiffusionModel

from .common import OscDecay

type PlottableSampler = structured.StructuredSampler | functional.FunctionalSampler

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
    return srgb.clip(0, 1).tolist()


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


@contextlib.contextmanager
def common_figure(
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 1920,
    height: int = 1920 // 2,
    dpi: float = 180,
) -> Generator[tuple[Figure, Axes], None, None]:
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black", edgecolor="white")
    ax = fig.add_subplot(1, 1, 1)

    yield fig, ax

    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.set_title(title, color="white")

    ax.set_facecolor("black")
    ax.grid(color="white")

    ax.tick_params(axis="both", which="both", color="white")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    [v.set_color("white") for v in list(ax.spines.values()) + ax.get_xticklabels() + ax.get_yticklabels()]

    ax.legend(facecolor="black", labelcolor="white", edgecolor="gray")


def plot_samplers(
    samplers: Sequence[PlottableSampler | tuple[PlottableSampler, str]],
    schedule: scheduling.SkrampleSchedule = scheduling.Hyper(scheduling.Linear(), tail=False),
    model: DiffusionModel = models.FlowModel(),
    ode: functional.SampleableModel = OscDecay(),
    steps: int = 30,
    reference_steps: int = 1000,
    reference_sampler: PlottableSampler = functional.RKUltra(order=4, providers={4: tableaux.providers.RK4.Kutta}),
    downsample_reference: bool = True,
    adjust_steps: bool = True,
) -> Figure:
    x0 = 1.0

    def sample_model(
        sampler: structured.StructuredSampler | functional.FunctionalSampler,
        steps: int,
        adjust: bool,
    ) -> tuple[list[float], list[float]]:
        if isinstance(sampler, structured.StructuredSampler):
            sampler = StructuredFunctionalAdapter(sampler)

        sample = x0
        sampled_values = [sample]
        timesteps = [0.0]

        def callback(x: float, n: int, d: DeltaPoint) -> None:
            nonlocal sampled_values, timesteps
            sampled_values.append(x)
            timesteps.insert(-1, d.point_from.timestep / schedule.point_1.timestep)

        if isinstance(sampler, functional.RKMoire) and adjust:
            adjusted = reference_steps
        elif isinstance(sampler, functional.FunctionalHigher) and adjust:
            adjusted = sampler.adjust_steps(steps)
        else:
            adjusted = steps

        sampler.sample_model(
            sample=sample,
            model=ode,
            model_transform=model,
            schedule=schedule,
            steps=adjusted,
            rng=lambda _: random.random(),
            callback=callback,
        )

        return timesteps, sampled_values

    with common_figure("Skrample Samplers", "Schedule", "Sample") as (fig, ax):
        ax.set_xlim(1, 0)

        ground_points, ground_truth = sample_model(reference_sampler, reference_steps, False)
        if downsample_reference and reference_steps > steps:
            ground_points = np.interp(
                np.linspace(0, 1, steps + 1), np.linspace(0, 1, reference_steps + 1), ground_points
            ).tolist()
            ground_truth = np.interp(
                np.linspace(0, 1, steps + 1), np.linspace(0, 1, reference_steps + 1), ground_truth
            ).tolist()

        gen = colors(6)
        ax.plot(ground_points, ground_truth, label="Reference", color=next(gen))
        ymin, ymax = min(ground_truth), max(ground_truth)
        ax.set_ylim(ymin - abs(0.1 * ymin), ymax + abs(0.1 * ymax))

        for packed in samplers:
            if isinstance(packed, tuple):
                sampler, label = packed
            else:
                sampler = packed
                label = type(sampler).__name__
                if isinstance(sampler, traits.HigherOrder) and sampler.order != type(sampler).order:
                    label += " " + str(sampler.order)
            ax.plot(*sample_model(sampler, steps, adjust_steps), label=label, color=next(gen), linestyle="--")

        return fig


def plot_schedules(
    schedules: Sequence[scheduling.SkrampleSchedule | tuple[scheduling.SkrampleSchedule, str]],
    steps: int = 30,
    alphas: bool = False,
    timesteps: bool = False,
) -> Figure:
    with common_figure("Skrample Schedules", "Step", "Noise") as (fig, ax):
        ax.set_ylim(0, 1)
        ax.set_xlim(0, steps)

        gen = colors(6)

        for packed in schedules:
            if isinstance(packed, tuple):
                schedule, label = packed
            else:
                schedule = packed
                label = " ".join(
                    reversed(
                        [
                            type(s).__name__
                            for s in (
                                schedule.all
                                if isinstance(schedule, scheduling.ScheduleModifier | scheduling.SubSchedule)
                                else [schedule]
                            )
                            if not isinstance(s, scheduling.NoSub | scheduling.NoMod)
                        ]
                    )
                )

            marker = "+" if steps <= 50 else ""
            data = schedule.ipoints_np(np.linspace(0, 1, steps + 1))

            if timesteps:
                ax.plot(
                    data[:, 0] / schedule.point_1.timestep,
                    label=label + " Timesteps",
                    marker=marker,
                    color=next(gen),
                )
            ax.plot(
                data[:, 1],
                label=label + (" Sigmas" if timesteps or alphas else ""),
                marker=marker,
                color=(color := next(gen)),
            )
            if alphas:
                ax.plot(
                    data[:, 2],
                    label=label + " Alphas",
                    marker=marker,
                    color=color,
                )

        return fig


def draw(fig: Figure) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
