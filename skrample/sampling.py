import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

    Sample = Tensor
else:
    # Avoid pulling all of torch as the code doesn't explicitly depend on it.
    Sample = NDArray


def sigma_normal(sigma: float, subnormal: bool = False) -> tuple[float, float]:
    if subnormal:
        return sigma, 1 - sigma
    else:
        alpha = 1 / ((sigma**2 + 1) ** 0.5)
        return sigma * alpha, alpha


def EPSILON(sample: Sample, output: Sample, sigma: float) -> Sample:
    sigma, alpha = sigma_normal(sigma)
    return (sample - sigma * output) / alpha


def SAMPLE(sample: Sample, output: Sample, sigma: float) -> Sample:
    return output


def VELOCITY(sample: Sample, output: Sample, sigma: float) -> Sample:
    sigma, alpha = sigma_normal(sigma)
    return alpha * sample - sigma * output


def FLOW(sample: Sample, output: Sample, sigma: float) -> Sample:
    return sample - sigma * output


@dataclass(frozen=True)
class SKSamples:
    prediction: Sample
    sampled: Sample


@dataclass
class SkrampleSampler(ABC):
    predictor: Callable[[Sample, Sample, float], Sample] = EPSILON

    @staticmethod
    def get_sigma(step: int, schedule: NDArray) -> float:
        return schedule[step, 1].item() if step < len(schedule) else 0

    @abstractmethod
    def sample(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
    ) -> SKSamples:
        pass

    def scale_input(self, sample: Sample, sigma: float) -> Sample:
        return sample

    def merge_noise(self, sample: Sample, noise: Sample, sigma: float) -> Sample:
        sigma, alpha = sigma_normal(sigma)
        return sample * alpha + noise * sigma


@dataclass
class Euler(SkrampleSampler):
    def sample(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
    ) -> SKSamples:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        prediction = self.predictor(self.scale_input(sample, sigma), output, sigma)

        return SKSamples(
            prediction=prediction,
            sampled=sample + ((sample - prediction) / sigma) * (sigma_n1 - sigma),
        )

    def scale_input(self, sample: Sample, sigma: float) -> Sample:
        return sample / ((sigma**2 + 1) ** 0.5)

    def merge_noise(self, sample: Sample, noise: Sample, sigma: float) -> Sample:
        return sample + noise * sigma


@dataclass
class EulerFlow(SkrampleSampler):
    def sample(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
    ) -> SKSamples:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        return SKSamples(prediction=output, sampled=sample + (sigma_n1 - sigma) * output)

    def merge_noise(self, sample: Sample, noise: Sample, sigma: float) -> Sample:
        sigma, alpha = sigma_normal(sigma, subnormal=True)
        return sample * alpha + noise * sigma


@dataclass
class DPM(SkrampleSampler):
    """https://arxiv.org/abs/2211.01095
    Page 4 Algo 2 for order=2
    Section 5 for SDE"""

    order: int = 1

    def sample(
        self,
        sample: Sample,
        output: Sample,
        schedule: NDArray,
        step: int,
        previous: list[SKSamples] = [],
    ) -> SKSamples:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        signorm, alpha = sigma_normal(sigma)
        signorm_n1, alpha_n1 = sigma_normal(sigma_n1)

        lambda_ = math.log(alpha) - math.log(signorm)
        if sigma_n1 == 0:
            h = float("inf")
        else:
            lambda_n1 = math.log(alpha_n1) - math.log(signorm_n1)
            h = lambda_n1 - lambda_

        prediction = self.predictor(sample, output, sigma)
        # 1st order
        sampled = (signorm_n1 / signorm) * sample - (alpha_n1 * (math.exp(-h) - 1.0)) * prediction

        effective_order = min(
            step + 1,
            self.order,
            len(previous) + 1,
            len(schedule) - step,  # lower for final is the default
        )

        if effective_order >= 2:
            sigma_p1 = self.get_sigma(step - 1, schedule)
            signorm_p1, alpha_p1 = sigma_normal(sigma_p1)

            if math.isfinite(h):
                lambda_p1 = math.log(alpha_p1) - math.log(signorm_p1)
                h_p1 = lambda_ - lambda_p1
                r = h_p1 / h  # math people and their var names...
            else:
                r = float("inf")

            # Calculate previous predicton from sample, output
            prediction_p1 = previous[-1].prediction
            prediction_p1 = (1.0 / r) * (prediction - prediction_p1)

            # 2nd order
            sampled -= 0.5 * (alpha_n1 * (math.exp(-h) - 1.0)) * prediction_p1

        return SKSamples(prediction=prediction, sampled=sampled)
