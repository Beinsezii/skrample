import abc
import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class Derivative(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: float, timestep: float, sigma: float, alpha: float) -> float:
        pass


@dataclasses.dataclass(frozen=True)
class Exponential(Derivative):
    scale: float = 6

    def __call__(self, x: float, timestep: float, sigma: float, alpha: float) -> float:
        return x * alpha * self.scale


@dataclasses.dataclass(frozen=True)
class OscDecay(Derivative):
    scale: float = 10
    frequency: float = 15
    decay: float = 3

    def __call__(self, x: float, timestep: float, sigma: float, alpha: float) -> float:
        return -x * math.sin(sigma * self.frequency) * math.exp(alpha * -self.decay) * self.scale
