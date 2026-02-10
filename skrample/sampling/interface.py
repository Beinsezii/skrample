import dataclasses

from skrample import scheduling
from skrample.common import RNG, FloatSchedule, Sample, SigmaTransform, Step

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

        for n, point in list(enumerate(float_schedule))[include]:
            sksamples = self.sampler.sample_packed(
                structured.SampleInput(
                    sample=sample,
                    prediction=model(self.sampler.scale_input(sample, point.sigma, schedule.sigma_transform), *point),
                    step=Step.from_int(n, len(float_schedule)),
                    noise=rng() if rng and self.sampler.require_noise else None,
                ),
                model_transform,
                schedule,
                previous=previous,
            )

            if self.sampler.require_previous > 0:
                previous.append(sksamples)
                previous = previous[max(len(previous) - self.sampler.require_previous, 0) :]

            sample = sksamples.final

            if callback:
                callback(sample, n, *float_schedule[n] if n < len(float_schedule) else (0, 0))

        return sample
