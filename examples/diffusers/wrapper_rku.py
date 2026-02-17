#! /usr/bin/env python

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

from skrample.diffusers import RKUltraWrapperScheduler

pipe: ZImagePipeline = ZImagePipeline.from_pretrained(  # type: ignore
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = RKUltraWrapperScheduler.from_diffusers_config(
    # Schedule, prediction, etc is auto detected
    pipe.scheduler.config,
    rk_order=2,
)

pipe.enable_model_cpu_offload()

imgs = pipe(
    prompt="bright high resolution dslr photograph of a kitten on a beach of rainbow pebbles",
    generator=torch.Generator("cpu").manual_seed(42),
    guidance_scale=1,
    num_inference_steps=4,
)
imgs.images[0].save("wrapper_rku.png")  # type: ignore
