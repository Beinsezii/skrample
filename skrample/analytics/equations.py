import abc
import dataclasses
import math
from collections.abc import Sequence
from typing import NamedTuple


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


@dataclasses.dataclass(frozen=True)
class Fourier(Derivative):
    class Wave(NamedTuple):
        freq: float = 2
        amp: float = 10
        phase: float = 0

    waves: Sequence[Wave] = (
        Wave(50, 1 / 2, 4 / 8),
        Wave(40, 1 / 2, 4 / 8),
        Wave(6, 2, -8 / 8),
        Wave(4, 4, 6 / 8),
        Wave(3, 4, 0),
        Wave(1, 4, -4 / 8),
    )

    def __call__(self, x: float, timestep: float, sigma: float, alpha: float) -> float:
        return -sum((math.sin(sigma * w.freq * math.pi + w.phase * math.pi) * w.amp for w in self.waves), start=x)
