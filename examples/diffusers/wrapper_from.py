#! /usr/bin/env python

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

import skrample.pytorch.noise as sknoise
import skrample.sampling as sampling
from skrample.diffusers import SkrampleWrapperScheduler

pipe: FluxPipeline = FluxPipeline.from_pretrained(  # type: ignore
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = SkrampleWrapperScheduler.from_diffusers_config(
    # Schedule, prediction, etc is auto detected
    pipe.scheduler.config,
    sampler=sampling.DPM,
    sampler_props={"order": 2, "add_noise": True},
    noise_type=sknoise.Brownian,
)

pipe.enable_model_cpu_offload()

imgs = pipe(
    "bright high resolution dslr photograph of a kitten on a beach of rainbow pebbles",
    generator=torch.Generator("cpu").manual_seed(42),
)
imgs.images[0].save("wrapper_from.png")  # type: ignore
