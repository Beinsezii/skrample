from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray
from torch import Tensor


def EPSILON(sample: Tensor, output: Tensor, sigma: float) -> Tensor:
    return sample - sigma * output


def SAMPLE(sample: Tensor, output: Tensor, sigma: float) -> Tensor:
    return output


def VELOCITY(sample: Tensor, output: Tensor, sigma: float) -> Tensor:
    return output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))


@dataclass
class SkrampleSampler(ABC):
    predictor: Callable[[Tensor, Tensor, float], Tensor] = EPSILON

    @staticmethod
    def get_sigma(step: int, schedule: NDArray) -> float:
        return schedule[step, 1] if step < len(schedule) else 0

    @abstractmethod
    def sample(self, sample: Tensor, output: Tensor, schedule: NDArray, step: int) -> Tensor:
        pass

    def scale_input(self, sample: Tensor, sigma: float) -> Tensor:
        return sample

    def merge_noise(self, sample: Tensor, noise: Tensor, sigma: float) -> Tensor:
        return sample + noise * sigma


@dataclass
class Euler(SkrampleSampler):
    def sample(self, sample: Tensor, output: Tensor, schedule: NDArray, step: int) -> Tensor:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        prediction = self.predictor(sample, output, sigma)

        return sample + ((sample - prediction) / sigma) * (sigma_n1 - sigma)

    def scale_input(self, sample: Tensor, sigma: float) -> Tensor:
        return sample / ((sigma**2 + 1) ** 0.5)


@dataclass
class EulerFlow(SkrampleSampler):
    def sample(self, sample: Tensor, output: Tensor, schedule: NDArray, step: int) -> Tensor:
        sigma = self.get_sigma(step, schedule)
        sigma_n1 = self.get_sigma(step + 1, schedule)

        return sample + (sigma_n1 - sigma) * output

    def merge_noise(self, sample: Tensor, noise: Tensor, sigma: float) -> Tensor:
        return sigma * noise + (1.0 - sigma) * sample
