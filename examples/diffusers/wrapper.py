#! /usr/bin/env python

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

import skrample.pytorch.noise as sknoise
import skrample.sampling as sampling
import skrample.scheduling as scheduling
from skrample.common import predict_flow
from skrample.diffusers import SkrampleWrapperScheduler

pipe: FluxPipeline = FluxPipeline.from_pretrained(  # type: ignore
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = scheduler = SkrampleWrapperScheduler(
    sampler=sampling.DPM(order=2, add_noise=True),
    schedule=scheduling.FlowShift(scheduling.Linear(), shift=2.0),
    predictor=predict_flow,
    noise_type=sknoise.Brownian,
    allow_dynamic=False,
)

pipe.enable_model_cpu_offload()

imgs = pipe(
    "bright high resolution dslr photograph of a kitten on a beach of rainbow pebbles",
    generator=torch.Generator("cpu").manual_seed(42),
)
imgs.images[0].save("wrapper.png")  # type: ignore
