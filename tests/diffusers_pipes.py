import gc

import torch
from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from skrample.diffusers import SkrampleScheduler


def compare_latents(a: torch.Tensor, b: torch.Tensor, margin: float = 1e-8):
    assert a.isfinite().all()
    assert b.isfinite().all()
    assert (a - b).abs().square().mean().item() < margin


def compare_schedulers(
    pipe: StableDiffusionXLImg2ImgPipeline | FluxImg2ImgPipeline,
    a: SkrampleScheduler,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    margin: float = 1e-8,
    **kwargs,
):
    original = pipe.scheduler

    pipe.scheduler = a
    a_o = pipe(output_type="latent", return_dict=False, generator=torch.Generator("cpu").manual_seed(0), **kwargs)[0]
    assert isinstance(a_o, torch.Tensor)

    pipe.scheduler = b
    b_o = pipe(output_type="latent", return_dict=False, generator=torch.Generator("cpu").manual_seed(0), **kwargs)[0]
    assert isinstance(b_o, torch.Tensor)

    pipe.scheduler = original

    print("TIMESTEPS", "\n", a.timesteps, "\n", b.timesteps)
    print("SIGMAS", "\n", a.sigmas, "\n", b.sigmas)

    compare_latents(a_o, b_o, margin=margin)


@torch.inference_mode()
def test_sdxl_i2i():
    gc.collect()
    dt, dv = torch.float32, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dt)
    pipe.enable_model_cpu_offload(device=dv)
    assert isinstance(pipe, StableDiffusionXLImg2ImgPipeline)

    a = SkrampleScheduler()
    b = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    assert isinstance(b, EulerDiscreteScheduler)

    compare_schedulers(
        pipe,
        a,
        b,
        image=torch.zeros([1, 4, 32, 32], dtype=dt, device=dv),
        num_inference_steps=8,
        prompt_embeds=torch.zeros([1, 77, 2048], dtype=dt, device=dv),
        negative_prompt_embeds=torch.zeros([1, 77, 2048], dtype=dt, device=dv),
        pooled_prompt_embeds=torch.zeros([1, 1280], dtype=dt, device=dv),
        negative_pooled_prompt_embeds=torch.zeros([1, 1280], dtype=dt, device=dv),
    )


@torch.inference_mode()
def test_flux_i2i():
    gc.collect()
    dt, dv = torch.float16, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dt)
    pipe.enable_model_cpu_offload(device=dv)
    assert isinstance(pipe, FluxImg2ImgPipeline)

    a = SkrampleScheduler(flow=True)
    b = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    assert isinstance(b, FlowMatchEulerDiscreteScheduler)

    fn, pipe._encode_vae_image = (
        pipe._encode_vae_image,
        lambda *_, **__: torch.zeros([1, 16, 32, 32], dtype=dt, device=dv),
    )

    compare_schedulers(
        pipe,
        a,
        b,
        5e-3,  # close enough for now
        height=256,
        width=256,
        image=torch.zeros([1, 1, 1], dtype=dt, device=dv),
        num_inference_steps=8,
        prompt_embeds=torch.zeros([1, 512, 4096], dtype=dt, device=dv),
        pooled_prompt_embeds=torch.zeros([1, 768], dtype=dt, device=dv),
    )

    pipe._encode_vae_image = fn
