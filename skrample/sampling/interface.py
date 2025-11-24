import dataclasses

from skrample.common import RNG, FloatSchedule, Sample

from . import functional, models, structured


@dataclasses.dataclass(frozen=True)
class StructuredFunctionalAdapter(functional.FunctionalSampler):
    sampler: structured.StructuredSampler

    def merge_noise[T: Sample](self, sample: T, noise: T, steps: int, start: int) -> T:
        sigmas = self.schedule.sigmas(steps)
        sigma = sigmas[start] if start < len(sigmas) else 0
        return self.sampler.merge_noise(sample, noise, sigma, self.schedule.sigma_transform)

    def sample_model[T: Sample](
        self,
        sample: T,
        model: functional.SampleableModel[T],
        model_transform: models.DiffusionModel,
        steps: int,
        include: slice = slice(None),
        rng: RNG[T] | None = None,
        callback: functional.SampleCallback | None = None,
    ) -> T:
        previous: list[structured.SKSamples[T]] = []
        schedule: FloatSchedule = self.schedule.schedule(steps)

        for n in list(range(len(schedule)))[include]:
            timestep, sigma = schedule[n]

            output = model(self.sampler.scale_input(sample, sigma, self.schedule.sigma_transform), timestep, sigma)
            prediction = model_transform.to_x(sample, output, sigma, self.schedule.sigma_transform)

            sksamples = self.sampler.sample(
                sample,
                prediction,
                n,
                schedule,
                self.schedule.sigma_transform,
                noise=rng() if rng and self.sampler.require_noise else None,
                previous=tuple(previous),
            )

            if self.sampler.require_previous > 0:
                previous.append(sksamples)
                previous = previous[max(len(previous) - self.sampler.require_previous, 0) :]

            sample = sksamples.final

            if callback:
                callback(sample, n, *schedule[n] if n < len(schedule) else (0, 0))

        return sample
