import dataclasses

from skrample import scheduling
from skrample.common import RNG, FloatSchedule, Sample, SigmaTransform

from . import functional, models, structured


@dataclasses.dataclass(frozen=True)
class StructuredFunctionalAdapter(functional.FunctionalSampler):
    sampler: structured.StructuredSampler

    def merge_noise[T: Sample](self, sample: T, noise: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return self.sampler.merge_noise(sample, noise, sigma, sigma_transform)

    def sample_model[T: Sample](
        self,
        sample: T,
        model: functional.SampleableModel[T],
        model_transform: models.DiffusionModel,
        schedule: scheduling.SkrampleSchedule,
        steps: int,
        include: slice = slice(None),
        rng: RNG[T] | None = None,
        callback: functional.SampleCallback | None = None,
    ) -> T:
        previous: list[structured.SKSamples[T]] = []
        float_schedule: FloatSchedule = schedule.schedule(steps)

        for n in list(range(len(float_schedule)))[include]:
            timestep, sigma = float_schedule[n]

            output = model(self.sampler.scale_input(sample, sigma, schedule.sigma_transform), timestep, sigma)
            prediction = model_transform.to_x(sample, output, sigma, schedule.sigma_transform)

            sksamples = self.sampler.sample(
                sample,
                prediction,
                n,
                float_schedule,
                schedule.sigma_transform,
                noise=rng() if rng and self.sampler.require_noise else None,
                previous=tuple(previous),
            )

            if self.sampler.require_previous > 0:
                previous.append(sksamples)
                previous = previous[max(len(previous) - self.sampler.require_previous, 0) :]

            sample = sksamples.final

            if callback:
                callback(sample, n, *float_schedule[n] if n < len(float_schedule) else (0, 0))

        return sample
