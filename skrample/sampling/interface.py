import dataclasses

from skrample.common import Sample, SigmaTransform
from skrample.sampling import functional, structured


@dataclasses.dataclass(frozen=True)
class StructuredFunctionalAdapter(functional.FunctionalSampler):
    sampler: structured.StructuredSampler

    def merge_noise[T: Sample](self, sample: T, noise: T, sigma: float, sigma_transform: SigmaTransform) -> T:
        return self.sampler.merge_noise(sample, noise, sigma, sigma_transform)

    def sample_model[T: Sample](
        self,
        sample: T,
        model: functional.FunctionalSampler.SampleableModel[T],
        steps: int,
        include: slice = slice(None),
        rng: functional.FunctionalSampler.RNG[T] | None = None,
        callback: functional.FunctionalSampler.SampleCallback | None = None,
    ) -> T:
        previous: list[structured.SKSamples[T]] = []
        schedule_np = self.schedule.schedule(steps)
        schedule: list[tuple[float, float]] = schedule_np.tolist()
        sigmas = schedule_np[:, 1]
        del schedule_np

        for n in list(range(len(schedule)))[include]:
            timestep, sigma = schedule[n]

            prediction = model(self.sampler.scale_input(sample, sigma, self.schedule.sigma_transform), timestep, sigma)

            sksamples = self.sampler.sample(
                sample,
                prediction,
                n,
                sigmas,
                self.schedule.sigma_transform,
                noise=rng() if rng else None,
                previous=tuple(previous),
            )

            if self.sampler.require_previous > 0:
                previous.append(sksamples)
                previous = previous[max(len(previous) - self.sampler.require_previous, 0) :]

            sample = sksamples.final

            if callback:
                callback(sample, n, *schedule[n] if n < len(schedule) else (0, 0))

        return sample
