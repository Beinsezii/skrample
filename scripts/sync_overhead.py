#! /usr/bin/env python

import statistics
from time import perf_counter_ns

import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

from skrample import scheduling
from skrample.diffusers import SkrampleWrapperScheduler
from skrample.sampling import models, structured

with torch.inference_mode():
    pipe: Flux2KleinPipeline = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
    ).to(dtype=torch.bfloat16, device=torch.accelerator.current_accelerator())

    pipe.transformer.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)

    pipe.scheduler = SkrampleWrapperScheduler(
        structured.Euler(),
        scheduling.Linear(),
        model=models.FlowModel(),
    )

    prompt_embeds, _text_ids = pipe.encode_prompt(
        "bright high resolution dslr photograph of a kitten on a beach of rainbow pebbles"
    )

    def bench_wrapper(steps: int) -> int:
        torch.accelerator.synchronize()

        now = perf_counter_ns()
        pipe(
            width=16,
            height=16,
            num_inference_steps=steps,
            output_type="latent",
            guidance_scale=1,
            prompt_embeds=prompt_embeds,
        )
        end = perf_counter_ns() - now

        torch.accelerator.synchronize()
        return end

    bench_wrapper(2)
    sync_times: list[int] = [bench_wrapper(5) for _ in range(50)]
    print(f"{round(statistics.median(sync_times) / 1e6, 2)} ±{round(statistics.stdev(sync_times) / 1e6, 2)} ms")
