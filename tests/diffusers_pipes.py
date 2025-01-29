import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.schedulers import SchedulerMixin
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

from skrample.diffusers import SkrampleScheduler


def compare_latents(a: torch.Tensor, b: torch.Tensor, margin: float = 1e-8):
    assert (a-b).abs().square().mean().item() < margin

def compare_schedulers(
    pipe: StableDiffusionXLImg2ImgPipeline,
    a: SkrampleScheduler,
    b: SchedulerMixin,
    margin: float = 1e-8,
    **kwargs,
):
    original = pipe.scheduler

    # pipe.scheduler = a
    # a_o = pipe(output_type="latent", return_dict=False, **kwargs)[0]
    # assert isinstance(a_o, torch.Tensor)

    pipe.scheduler = b
    b_o = pipe(output_type="latent", return_dict=False, **kwargs)[0]
    assert isinstance(b_o, torch.Tensor)

    a_o = torch.randn_like(b_o)

    pipe.scheduler = original

    compare_latents(a_o, b_o, margin=margin)


def test_sdxl_i2i():
    dt, dv = torch.float32, torch.device("cpu")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dt
    ).to(dv)
    assert isinstance(pipe, StableDiffusionXLImg2ImgPipeline)

    a = SkrampleScheduler()
    b = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    assert isinstance(b, SchedulerMixin)

    compare_schedulers(
        pipe,
        a,
        b,
        image=torch.zeros([1, 4, 32, 32], dtype=dt, device=dv),
        num_inference_steps=5,
        prompt_embeds=torch.zeros([1, 77, 2048], dtype=dt, device=dv),
        negative_prompt_embeds=torch.zeros([1, 77, 2048], dtype=dt, device=dv),
        pooled_prompt_embeds=torch.zeros([1, 1280], dtype=dt, device=dv),
        negative_pooled_prompt_embeds=torch.zeros([1, 1280], dtype=dt, device=dv),
    )
