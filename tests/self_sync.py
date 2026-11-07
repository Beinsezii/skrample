import random

import pytest
import torch
from testing_common import ALL_FUNCTIONAL, ALL_STRUCTURED, ALL_WRAPPERS, FLOW_CONFIG
from testing_generated import sync_detection

from skrample import scheduling
from skrample.common import Step
from skrample.diffusers import BuiltinSkrampleWrapper
from skrample.sampling import functional, interface, models, structured


@pytest.mark.parametrize("wrapper_cls", ALL_WRAPPERS)
@pytest.mark.parametrize("steps", [1, 2, 5, 99])
def test_wrapper_sync(wrapper_cls: type[BuiltinSkrampleWrapper], steps: int) -> None:
    wrapper = wrapper_cls.from_diffusers_config(FLOW_CONFIG)
    with sync_detection.SyncCounter() as counter:
        x: torch.Tensor = torch.randn([4, 4], dtype=torch.float32, device="cpu")
        wrapper.set_timesteps(steps)
        for t in wrapper.timesteps:
            counter.mark()
            x = wrapper.step(torch.randn_like(x), t, x, return_dict=False)[0]

        assert counter.total == steps * wrapper.order
        assert all(ps == 1 for ps in counter.per_step), counter.per_step


@pytest.mark.parametrize("sampler", ALL_FUNCTIONAL + ALL_STRUCTURED)
@pytest.mark.parametrize("steps", [1, 2, 5, 99])
def test_sampler_sync(
    sampler: type[functional.FunctionalSampler] | type[structured.StructuredSampler],
    steps: int,
) -> None:

    if issubclass(sampler, structured.StructuredSampler):
        solver: functional.FunctionalSampler = interface.StructuredFunctionalAdapter(sampler())
    elif issubclass(sampler, functional.FunctionalAdaptive):
        solver = sampler(evaluator=lambda a, b: random.random())  # Fake evaluator that doesn't need syncs
    else:
        solver = sampler()

    with sync_detection.SyncCounter() as counter:

        def rng(_: Step | None) -> torch.Tensor:
            return torch.randn([4, 4], dtype=torch.float32, device="cpu")

        def model(x: torch.Tensor, t: float, s: float, a: float) -> torch.Tensor:
            return x @ rng(None)

        solver.generate_model(model, models.FlowModel(), scheduling.Linear(), rng, steps)

        assert counter.total == 0
