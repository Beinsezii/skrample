import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import torch
from testing_common import compare_tensors

from skrample.common import (
    MergeStrategy,
    SigmaTransform,
    bashforth,
    euler,
    sigma_complement,
    sigma_polar,
    sigmoid,
    softmax,
    spowf,
)
from skrample.diffusers import SkrampleWrapperScheduler
from skrample.sampling import tableaux
from skrample.sampling.interface import StructuredFunctionalAdapter
from skrample.sampling.models import (
    DiffusionModel,
    EpsilonModel,
    FlowModel,
    ModelConvert,
    ModelTransform,
    ScaleX,
    VelocityModel,
)
from skrample.sampling.structured import (
    DPM,
    SPC,
    Adams,
    Euler,
    SKSamples,
    StructuredMultistep,
    StructuredSampler,
    StructuredStochastic,
    UniPC,
)
from skrample.scheduling import Beta, FlowShift, Karras, Linear, Scaled, ScheduleCommon, ScheduleModifier, SigmoidCDF

ALL_STRUCTURED: Sequence[type[StructuredSampler]] = [
    Adams,
    DPM,
    Euler,
    SPC,
    UniPC,
]

ALL_SCHEDULES: Sequence[type[ScheduleCommon]] = [
    Linear,
    Scaled,
    SigmoidCDF,
]

ALL_MODIFIERS: Sequence[type[ScheduleModifier]] = [
    Beta,
    FlowShift,
    Karras,
]

ALL_MODELS: Sequence[type[ModelTransform]] = [
    DiffusionModel,
    EpsilonModel,
    FlowModel,
    VelocityModel,
]

ALL_FAKE_MODELS: Sequence[type[ModelTransform]] = [
    ScaleX,
]

ALL_TRANSFROMS: Sequence[SigmaTransform] = [
    sigma_complement,
    sigma_polar,
]


def test_sigmas_to_timesteps() -> None:
    for schedule in [*(cls() for cls in ALL_SCHEDULES), Scaled(beta_scale=1)]:  # base schedules
        timesteps = schedule.timesteps_np(123)
        timesteps_inv = schedule.sigmas_to_timesteps(schedule.sigmas_np(123))
        compare_tensors(torch.tensor(timesteps), torch.tensor(timesteps_inv), margin=0)  # shocked this rounds good


@pytest.mark.parametrize(
    ("model_type", "sigma_transform"),
    itertools.product(ALL_MODELS, ALL_TRANSFROMS),
)
def test_model_transforms(model_type: type[ModelTransform], sigma_transform: SigmaTransform) -> None:
    model_transform = model_type()
    sample = 0.8
    output = 0.3
    sigma = 0.2

    x = model_transform.to_x(sample, output, sigma, sigma_transform)
    o = model_transform.from_x(sample, x, sigma, sigma_transform)
    assert abs(output - o) < 1e-12

    sigma_next = 0.05
    for sigma_next in 0.05, 0:  # extra 0 to validate X̂
        snr = euler(
            sample, model_transform.to_x(sample, output, sigma, sigma_transform), sigma, sigma_next, sigma_transform
        )
        df = model_transform.forward(sample, output, sigma, sigma_next, sigma_transform)
        assert abs(snr - df) < 1e-12

        ob = model_transform.backward(sample, df, sigma, sigma_next, sigma_transform)
        assert abs(o - ob) < 1e-12


@pytest.mark.parametrize(
    ("model_from", "model_to", "sigma_transform", "sigma_to"),
    itertools.product(ALL_MODELS, ALL_MODELS + ALL_FAKE_MODELS, ALL_TRANSFROMS, (0.05, 0.0)),
)
def test_model_convert(
    model_from: type[ModelTransform],
    model_to: type[ModelTransform],
    sigma_transform: SigmaTransform,
    sigma_to: float,
) -> None:
    convert = ModelConvert(model_from(), model_to())
    sample = 0.8
    output = 0.3
    sigma_from = 0.2

    def model(x: float, t: float, s: float) -> float:
        return output

    x_from = convert.transform_from.forward(
        sample,
        model(sample, sigma_from, sigma_from),
        sigma_from,
        sigma_to,
        sigma_transform,
    )
    x_to = convert.transform_to.forward(
        sample,
        convert.wrap_model_call(model, sigma_transform)(sample, sigma_from, sigma_from),
        sigma_from,
        sigma_to,
        sigma_transform,
    )

    assert abs(x_from - x_to) < 1e-12


def test_sampler_generics() -> None:
    eps = 1e-12
    for sampler in [
        *(cls() for cls in ALL_STRUCTURED),
        *(cls(order=cls.max_order()) for cls in ALL_STRUCTURED if issubclass(cls, StructuredMultistep)),
    ]:
        for schedule in Scaled(), FlowShift(Linear()):
            i, o = random.random(), random.random()
            prev = [SKSamples(random.random(), random.random(), random.random()) for _ in range(9)]

            scalar = sampler.sample(i, o, 4, schedule.schedule(10), schedule.sigma_transform, previous=prev).final

            # Enforce FP64 as that should be equivalent to python scalar
            ndarr = sampler.sample(
                np.array([i], dtype=np.float64),
                np.array([o], dtype=np.float64),
                4,
                schedule.schedule(10),
                schedule.sigma_transform,
                previous=prev,  # type: ignore
            ).final.item()

            tensor = sampler.sample(
                torch.tensor([i], dtype=torch.float64),
                torch.tensor([o], dtype=torch.float64),
                4,
                schedule.schedule(10),
                schedule.sigma_transform,
                previous=prev,  # type: ignore
            ).final.item()

            assert abs(tensor - scalar) < eps
            assert abs(tensor - ndarr) < eps
            assert abs(scalar - ndarr) < eps


def test_mu_set() -> None:
    mu = 1.2345
    a = SkrampleWrapperScheduler(DPM(), Beta(FlowShift(Karras(Linear()))))
    b = SkrampleWrapperScheduler(DPM(), Beta(FlowShift(Karras(Linear()), shift=math.exp(mu))))
    a.set_timesteps(1, mu=mu)
    assert a.schedule == b.schedule


def test_require_previous() -> None:
    samplers: list[StructuredSampler] = []
    for cls in ALL_STRUCTURED:
        if issubclass(cls, StructuredMultistep):
            samplers.extend([cls(order=o + 1) for o in range(cls.min_order(), cls.max_order())])
        else:
            samplers.append(cls())

    for o1 in range(1, 4):
        for o2 in range(1, 4):
            samplers.append(UniPC(order=o1, solver=Adams(order=o2)))
            samplers.append(SPC(predictor=Adams(order=o1), corrector=Adams(order=o2)))

    for sampler in samplers:
        sample = 1.5
        prediction = 0.5
        previous = tuple(SKSamples(n / 2, n * 2, n * 1.5) for n in range(100))

        a = sampler.sample(
            sample,
            prediction,
            31,
            Linear().schedule(100),
            sigma_complement,
            None,
            previous,
        )
        b = sampler.sample(
            sample,
            prediction,
            31,
            Linear().schedule(100),
            sigma_complement,
            None,
            previous[len(previous) - sampler.require_previous :],
        )

        assert a == b, (sampler, sampler.require_previous)


def test_require_noise() -> None:
    samplers: list[StructuredSampler] = []
    for cls in ALL_STRUCTURED:
        if issubclass(cls, StructuredStochastic):
            samplers.extend([cls(add_noise=n) for n in (False, True)])
        else:
            samplers.append(cls())

    for n1 in (False, True):
        for n2 in (False, True):
            samplers.append(UniPC(solver=DPM(add_noise=n2)))
            samplers.append(SPC(predictor=DPM(add_noise=n1), corrector=DPM(add_noise=n2)))

    for sampler in samplers:
        sample = 1.5
        prediction = 0.5
        previous = tuple(SKSamples(n / 2, n * 2, n * 1.5) for n in range(100))
        noise = -0.5

        a = sampler.sample(
            sample,
            prediction,
            31,
            Linear().schedule(100),
            sigma_complement,
            noise,
            previous,
        )
        b = sampler.sample(
            sample,
            prediction,
            31,
            Linear().schedule(100),
            sigma_complement,
            noise if sampler.require_noise else None,
            previous,
        )

        # Don't compare stored noise since it's expected diff
        b = replace(b, noise=a.noise)

        assert a == b, (sampler, sampler.require_noise)


def test_functional_adapter() -> None:
    def fake_model(x: float, _: float, s: float) -> float:
        return x + math.sin(x) * s

    samplers: list[StructuredSampler] = [DPM(n, o) for o in range(1, 4) for n in [False, True]]
    for schedule in Linear(), Scaled():
        for sampler in samplers:
            for steps in [1, 3, 4, 9, 512, 999]:
                sample = 1.5
                adapter = StructuredFunctionalAdapter(schedule, sampler)
                noise = [random.random() for _ in range(steps)]

                rng = iter(noise)
                model_transform = FlowModel()
                sample_f = adapter.sample_model(sample, fake_model, model_transform, steps, rng=lambda: next(rng))

                rng = iter(noise)
                float_schedule = schedule.schedule(steps)
                sample_s = sample
                previous: list[SKSamples[float]] = []
                for n, (t, s) in enumerate(float_schedule):
                    results = sampler.sample(
                        sample_s,
                        model_transform.to_x(sample_s, fake_model(sample_s, t, s), s, schedule.sigma_transform),
                        n,
                        float_schedule,
                        schedule.sigma_transform,
                        next(rng),
                        tuple(previous),
                    )
                    previous.append(results)
                    sample_s = results.final

                assert sample_s == sample_f, (sample_s, sample_f, sampler, schedule, steps)


def test_bashforth() -> None:
    for n, coeffs in enumerate(
        np.array(c) for c in ((1,), (3 / 2, -1 / 2), (23 / 12, -4 / 3, 5 / 12), (55 / 24, -59 / 24, 37 / 24, -3 / 8))
    ):
        assert np.allclose(coeffs, np.array(bashforth(n + 1)), atol=1e-12, rtol=1e-12)


def test_tableau_providers() -> None:
    for provider in [tableaux.RK2, tableaux.RK3, tableaux.RK4, tableaux.RK5, tableaux.RKE2, tableaux.RKE5]:
        for variant in provider:
            if error := tableaux.validate_tableau(variant.tableau()):
                raise error


def flat_tableau(t: tuple[float | tuple[float | tuple[float | tuple[float, ...], ...], ...], ...]) -> tuple[float, ...]:
    return tuple(z for y in (flat_tableau(x) if isinstance(x, tuple) else (x,) for x in t) for z in y)


def tableau_distance(a: tableaux.Tableau, b: tableaux.Tableau) -> float:
    return abs(np.subtract(flat_tableau(a), flat_tableau(b))).max().item()


def test_rk2_tableau() -> None:
    assert (
        tableau_distance(
            (  # Ralston
                (
                    (0.0, ()),
                    (2 / 3, (2 / 3,)),
                ),
                (1 / 4, 3 / 4),
            ),
            tableaux.rk2_tableau(2 / 3),
        )
        < 1e-20
    )


def test_rk3_tableau() -> None:
    assert (
        tableau_distance(
            (  # Wray
                (
                    (0.0, ()),
                    (8 / 15, (8 / 15,)),
                    (2 / 3, (1 / 4, 5 / 12)),
                ),
                (1 / 4, 0.0, 3 / 4),
            ),
            tableaux.rk3_tableau(8 / 15, 2 / 3),
        )
        < 1e-15
    )


def test_sigmoid() -> None:
    items = spowf(torch.linspace(-2, 2, 9, dtype=torch.float64), 2)
    a = torch.sigmoid(items)
    b = sigmoid(items)
    assert torch.allclose(a, b, rtol=0, atol=1e-12), (a.tolist(), b.tolist())


def test_softmax() -> None:
    items = spowf(torch.linspace(-2, 2, 9, dtype=torch.float64), 2)
    a = torch.softmax(items, 0)
    b = torch.tensor(softmax(tuple(items)), dtype=torch.float64)
    assert torch.allclose(a, b, rtol=0, atol=1e-12), (a.tolist(), b.tolist())


def test_merge() -> None:
    array_deltas: list[tuple[list[int], list[int], list[int], list[int]]] = [
        (list(range(0, 11)), list(range(0, 15, 2)), list(range(1, 10, 2)), list(range(12, 15, 2))),
        (list(range(4, 15)), list(range(0, 11, 2)), list(range(5, 11, 2)) + list(range(11, 15)), list(range(0, 4, 2))),
    ]
    for a, b, aX, bX in array_deltas:
        tests: list[tuple[list[int], list[int], MergeStrategy, list[int]]] = [
            (a, b, MergeStrategy.Ours, a),
            (b, a, MergeStrategy.Ours, b),
            (a, b, MergeStrategy.Theirs, b),
            (b, a, MergeStrategy.Theirs, a),
            (a, b, MergeStrategy.After, a + b),
            (b, a, MergeStrategy.After, b + a),
            (a, b, MergeStrategy.Before, b + a),
            (b, a, MergeStrategy.Before, a + b),
            (a, b, MergeStrategy.UniqueBefore, b + aX),
            (b, a, MergeStrategy.UniqueBefore, a + bX),
            (a, b, MergeStrategy.UniqueAfter, a + bX),
            (b, a, MergeStrategy.UniqueAfter, b + aX),
        ]
        for ours, theirs, ms, merged in tests:
            assert ms.merge(ours, theirs) == merged, f"{ours} {ms} {theirs} : {merged}"
