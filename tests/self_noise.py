import itertools

import numpy as np
import pytest
import scipy.fft as fft
import torch
from scipy.stats import linregress

from skrample.common import Step
from skrample.pytorch.noise import Colored, ColoredProps


def measure_noise_color(data: np.ndarray) -> float:
    """
    Measures the spectral exponent (beta) of an n-dimensional noise array.
    Implemented by Gemini. Was intentionally given no reference to Colored or skrample.
    """
    ndim = data.ndim
    shape = data.shape

    # 1. Compute the n-dimensional FFT and shift the DC component to the center
    fft_dims = tuple(range(ndim))
    F = fft.fftn(data, axes=fft_dims)
    F_shifted = fft.fftshift(F)
    psd = np.abs(F_shifted) ** 2

    # 2. Generate frequency grids for each dimension
    freqs = [fft.fftshift(fft.fftfreq(s)) for s in shape]
    mesh = np.meshgrid(*freqs, indexing="ij")

    # 3. Calculate the radial frequency (Euclidean distance from center)
    radial_freq = np.sqrt(sum(m**2 for m in mesh))

    # 4. Mask out the DC component (frequency = 0) to avoid log(0)
    mask = radial_freq > 0
    radial_freq_flat = radial_freq[mask]
    psd_flat = psd[mask]

    # 5. Bin the radial frequencies
    num_bins = min(shape) // 2
    bin_edges = np.linspace(radial_freq_flat.min(), radial_freq_flat.max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_indices = np.digitize(radial_freq_flat, bin_edges) - 1
    bin_powers = np.zeros(num_bins)

    # Calculate the MEAN power per radial bin
    for i in range(num_bins):
        bin_powers[i] = np.mean(psd_flat[bin_indices == i])

    # 6. Filter out empty or unrepresentative bins
    valid = (bin_powers > 0) & (bin_centers > 0)
    log_f = np.log(bin_centers[valid])
    log_p = np.log(bin_powers[valid])

    # 7. Linear regression on log-log scale: log(P) = -beta * log(f) + C
    slope, _intercept, _r_value, _p_value, _std_err = linregress(log_f, log_p)

    beta = -slope.item()  # pyright: ignore
    return beta


@pytest.mark.parametrize(
    ("exponent", "shape"),
    (itertools.product([-3, -1.5, 0, 1.5, 3], [(65536,), (1024, 1024), (128, 128, 128)])),
)
def test_noise_color(exponent: float, shape: tuple[int, ...]) -> None:
    generator = Colored(
        shape,
        torch.Generator("cpu"),
        torch.float32,
        ColoredProps(color_curve=0, color_start=exponent, color_end=-exponent),
    )
    n0 = generator.generate(None)
    color0 = measure_noise_color(n0.numpy())
    assert abs(exponent - color0) < 0.1, f"{exponent=}, {color0=}"

    n1 = generator.generate(Step(0, 1))
    color1 = measure_noise_color(n1.numpy())
    assert abs(-exponent - color1) < 0.1, f"{-exponent=}, {color1=}"


@pytest.mark.parametrize(
    ("energy", "shape"),
    (itertools.product([None, -3, -1.5, 0, 1.5, 3], [(65536,), (1024, 1024), (128, 128, 128)])),
)
def test_noise_energy(energy: float | None, shape: tuple[int, ...]) -> None:
    generator = Colored(
        shape,
        torch.Generator("cpu"),
        torch.float32,
        ColoredProps(energy=energy, color_start=torch.randn(1).item(), color_end=torch.randn(1).item()),
    )

    std0 = generator.generate(None).std().item()
    std1 = generator.generate(Step(0, 1)).std().item()

    if energy is None:
        assert abs(1 - std0) < 1e-2, f"{energy=}, {std0=}"
        assert abs(1 - std1) < 1e-2, f"{energy=}, {std1=}"
    else:
        assert abs(abs(energy) - std0) < 1e-6, f"{energy=}, {std0=}"
        assert abs(abs(energy) - std1) < 1e-6, f"{energy=}, {std1=}"
