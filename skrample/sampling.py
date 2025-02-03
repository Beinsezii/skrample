import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray
from torch import Tensor


def sigma_normal(sigma: float, subnormal: bool = False) -> tuple[float, float]:
    if subnormal:
        return sigma, 1 - sigma
    else:
        alpha = 1 / ((sigma**2 + 1) ** 0.5)
        return sigma * alpha, alpha


def EPSILON(sample: Tensor, output: Tensor, sigma: float) -> Tensor:
    sigma, alpha = sigma_normal(sigma)
    return (sample - sigma * output) / alpha


def SAMPLE(sample: Tensor, output: Tensor, sigma: float) -> Tensor:
    return output


def VELOCITY(sample: Tensor, output: Tensor, sigma: float) -> Tensor:
    sigma, alpha = sigma_normal(sigma)
    return alpha * sample - sigma * output


def FLOW(sample: Tensor, output: Tensor, sigma: float) -> Tensor:
    return sample - sigma * output


@dataclass
class SkrampleSampler(ABC):
    predictor: Callable[[Tensor, Tensor, float], Tensor] = EPSILON

    @staticmethod
    def get_sigma(step: int, schedule: NDArray) -> float:
        return schedule[step, 1].item() if step < len(schedule) else 0

    @abstractmethod
    def sample(self, sample: Tensor, output: Tensor, schedule: NDArray, step: int) -> Tensor:
        pass

    def scale_input(self, sample: Tensor, sigma: float) -> Tensor:
        return sample

    def merge_noise(self, sample: Tensor, noise: Tensor, sigma: float) -> Tensor:
        sigma, alpha = sigma_normal(sigma)
        return sample * alpha + noise * sigma


@dataclass
class Euler(SkrampleSampler):
    def sample(self, sample: Tensor, output: Tensor, schedule: NDArray, step: int) -> Tensor:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        prediction = self.predictor(self.scale_input(sample, sigma), output, sigma)

        return sample + ((sample - prediction) / sigma) * (sigma_n1 - sigma)

    def scale_input(self, sample: Tensor, sigma: float) -> Tensor:
        return sample / ((sigma**2 + 1) ** 0.5)

    def merge_noise(self, sample: Tensor, noise: Tensor, sigma: float) -> Tensor:
        return sample + noise * sigma


@dataclass
class EulerFlow(SkrampleSampler):
    def sample(self, sample: Tensor, output: Tensor, schedule: NDArray, step: int) -> Tensor:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        return sample + (sigma_n1 - sigma) * output

    def merge_noise(self, sample: Tensor, noise: Tensor, sigma: float) -> Tensor:
        sigma, alpha = sigma_normal(sigma, subnormal=True)
        return sample * alpha + noise * sigma


@dataclass
class DPM(SkrampleSampler):
    def sample(self, sample: Tensor, output: Tensor, schedule: NDArray, step: int) -> Tensor:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        sigma_t, sigma_s = sigma_n1, sigma
        sigma_s, alpha_s = sigma_normal(sigma_s)
        sigma_t, alpha_t = sigma_normal(sigma_t)

        prediction = self.predictor(sample, output, sigma)

        if sigma_n1 == 0:
            h = float("inf")
        else:
            lambda_t = math.log(alpha_t) - math.log(sigma_t)
            lambda_s = math.log(alpha_s) - math.log(sigma_s)
            h = lambda_t - lambda_s

        # 1st order non-sde
        prediction = (sigma_t / sigma_s) * sample - (alpha_t * (math.exp(-h) - 1.0)) * prediction
        return prediction
