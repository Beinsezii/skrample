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
    rk_order=3,
)

pipe.enable_model_cpu_offload()

imgs = pipe(
    prompt="Analogue portrait photograph of a woman in a stained glass church. "
    "She is wearing gothic plate armor and has short, curly blonde hair. "
    "The photo is softly lit, with the light in the image being provided "
    "by multicolored rays coming from the church windows.",
    generator=torch.Generator("cpu").manual_seed(42),
    guidance_scale=1,
    num_inference_steps=4,
)
imgs.images[0].save("wrapper_rku.png")  # type: ignore
