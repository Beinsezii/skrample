from collections.abc import Sequence

import numpy as np
import pytest
from matplotlib.lines import Line2D
from testing_common import ALL_FUNCTIONAL, ALL_SCHEDULES, ALL_STRUCTURED

from skrample import scheduling
from skrample.analytics import plotting
from skrample.analytics.equations import Derivative, Exponential, Fourier, OscDecay
from skrample.sampling import functional, structured

# A representative sampler from each structured and functional family.
SAMPLERS: list[structured.StructuredSampler | functional.FunctionalSampler] = [
    s() for s in (ALL_STRUCTURED + ALL_FUNCTIONAL)
]

# A representative schedule from each base / subschedule / modifier family.
SCHEDULES: list[scheduling.SkrampleSchedule] = [
    scheduling.Beta(scheduling.Linear()),
    scheduling.Hyper(scheduling.Linear()),
    scheduling.Sinner(scheduling.Linear()),
] + [s() for s in ALL_SCHEDULES]

STEPS: Sequence[int] = [1, 3, 8, 30, 999, 1000, 1001, 10_001]

# The reference trajectory is always plotted first, before the user samplers.
REFERENCE_LINE: int = 0
FIRST_SAMPLER_LINE: int = 1


def lines(fig: plotting.Figure) -> Sequence[Line2D]:
    return fig.axes[0].lines


def xdata(line: Line2D) -> np.ndarray:
    return np.asarray(line.get_xdata())


def ydata(line: Line2D) -> np.ndarray:
    return np.asarray(line.get_ydata())


def labels(fig: plotting.Figure) -> list[str]:
    legend = fig.axes[0].get_legend()
    assert legend is not None
    return [str(t.get_text()) for t in legend.get_texts()]


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_plot_samplers_structure(
    steps: int,
    sampler: structured.StructuredSampler | functional.FunctionalSampler,
) -> None:
    fig = plotting.plot_samplers([sampler], steps=steps, reference_steps=max(steps, 1000), adjust_steps=False)
    all_lines = lines(fig)

    # One reference line + one line per sampler.
    assert len(all_lines) == 2

    # Reference trajectory.
    reference_x, reference_y = xdata(all_lines[REFERENCE_LINE]), ydata(all_lines[REFERENCE_LINE])
    assert str(all_lines[REFERENCE_LINE].get_label()) == "Reference"
    assert len(reference_x) == steps + 1
    assert np.isfinite(reference_y).all()
    # Schedule timesteps are normalized to the unit range, noise -> clean.
    assert reference_x[0] == 1.0
    assert reference_x[-1] == 0.0

    # Sampler trajectory.
    sample_x, sample_y = xdata(all_lines[FIRST_SAMPLER_LINE]), ydata(all_lines[FIRST_SAMPLER_LINE])
    if isinstance(sampler, functional.FunctionalAdaptive):
        # Adaptive step sizing yields a variable number of points.
        assert len(sample_x) >= 2
    else:
        assert len(sample_x) == steps + 1
    assert np.isfinite(sample_y).all()
    assert sample_x[0] == 1.0
    assert sample_x[-1] == 0.0

    # Legend reflects the reference and the sampler type name.
    assert labels(fig) == ["Reference", type(sampler).__name__]


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_plot_samplers_deterministic(sampler: structured.StructuredSampler | functional.FunctionalSampler) -> None:
    a = plotting.plot_samplers([sampler], steps=6, adjust_steps=False)
    b = plotting.plot_samplers([sampler], steps=6, adjust_steps=False)

    for line_a, line_b in zip(lines(a), lines(b)):
        assert np.array_equal(xdata(line_a), xdata(line_b))
        assert np.array_equal(ydata(line_a), ydata(line_b))


def test_plot_samplers_custom_label() -> None:
    fig = plotting.plot_samplers([(structured.Euler(), "My Sampler")], steps=4)
    assert labels(fig) == ["Reference", "My Sampler"]


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_plot_schedules_structure(steps: int, schedule: scheduling.SkrampleSchedule) -> None:
    fig = plotting.plot_schedules([schedule], steps=steps)
    all_lines = lines(fig)

    # One sigma line per schedule by default.
    assert len(all_lines) == 1

    x, y = xdata(all_lines[0]), ydata(all_lines[0])
    assert len(x) == steps + 1
    assert np.isfinite(y).all()
    # Sigmas span the unit noise range.
    assert np.min(y) >= -1e-12
    assert np.max(y) <= 1.0 + 1e-12


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_plot_schedules_modes(steps: int, schedule: scheduling.SkrampleSchedule) -> None:
    fig = plotting.plot_schedules([schedule], steps=steps, timesteps=True, alphas=True)
    all_lines = lines(fig)

    # timesteps + sigmas + alphas per schedule.
    assert len(all_lines) == 3

    for line in all_lines:
        x, y = xdata(line), ydata(line)
        assert len(x) == steps + 1
        assert np.isfinite(y).all()
        assert np.min(y) >= -1e-12
        assert np.max(y) <= 1.0 + 1e-12

    # Labels distinguish the three series (suffix is stable across schedules).
    line_labels = [str(line.get_label()) for line in all_lines]
    assert line_labels[0].endswith("Timesteps")
    assert line_labels[1].endswith("Sigmas")
    assert line_labels[2].endswith("Alphas")


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_plot_schedules_deterministic(schedule: scheduling.SkrampleSchedule) -> None:
    a = plotting.plot_schedules([schedule], steps=6)
    b = plotting.plot_schedules([schedule], steps=6)

    for line_a, line_b in zip(lines(a), lines(b)):
        assert np.array_equal(ydata(line_a), ydata(line_b))


@pytest.mark.parametrize("steps", STEPS)
def test_draw(steps: int) -> None:
    fig = plotting.plot_samplers([structured.Euler()], steps=steps)
    arr = plotting.draw(fig)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 4  # RGBA
    assert arr.shape[0] > 0
    assert arr.shape[1] > 0


@pytest.mark.parametrize("equation", [Exponential(), OscDecay(), Fourier()])
def test_equations(equation: Derivative) -> None:
    result = equation(0.5, 0.5, 0.5, 0.5)

    assert isinstance(result, float)
    assert np.isfinite(result)
