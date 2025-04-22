import json
from pathlib import Path
from typing import Any

import pytest
import torch
from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import StableDiffusion3Img2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from testing_common import compare_tensors

from skrample.diffusers import SkrampleWrapperScheduler

try:
    from diffusers.pipelines.wan.pipeline_wan import WanPipeline
except ImportError:
    WanPipeline = None

try:
    from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
except ImportError:
    LTXPipeline = None


@torch.inference_mode()
def compare_schedulers(
    pipe: StableDiffusionXLImg2ImgPipeline | FluxImg2ImgPipeline,
    a: SkrampleWrapperScheduler,
    b: EulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler,
    margin: float = 1e-4,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    original = pipe.scheduler

    pipe.scheduler = a
    a_o = pipe(output_type="latent", return_dict=False, generator=torch.Generator("cpu").manual_seed(0), **kwargs)[0]
    assert isinstance(a_o, torch.Tensor)

    pipe.scheduler = b
    b_o = pipe(output_type="latent", return_dict=False, generator=torch.Generator("cpu").manual_seed(0), **kwargs)[0]
    assert isinstance(b_o, torch.Tensor)

    pipe.scheduler = original

    mid = len(a.timesteps) // 2
    compare_tensors(
        a_o,
        b_o,
        "\n"
        + " ".join(
            [
                f"AT0 {a.timesteps[0].item():.3f}",
                f"ATM {a.timesteps[mid].item():.3f}",
                f"ATP {a.timesteps[-1].item():.3f}",
                f"AS0 {a.sigmas[0].item():.3f}",
                f"ASM {a.sigmas[mid].item():.3f}",
                f"ASP {a.sigmas[-2].item():.3f}",
            ]
        )
        + "\n"
        + " ".join(
            [
                f"BT0 {b.timesteps[0].item():.3f}",
                f"BTM {b.timesteps[mid].item():.3f}",
                f"BTP {b.timesteps[-1].item():.3f}",
                f"BS0 {b.sigmas[0].item():.3f}",
                f"BSM {b.sigmas[mid].item():.3f}",
                f"BSP {b.sigmas[-2].item():.3f}",
            ]
        ),
        margin=margin,
    )


def fake_pipe_init[T](
    cls: type[T],
    uri: str,
) -> T:
    torch.manual_seed(0)

    config = json.load(Path(hf_hub_download(uri, "model_index.json")).open())
    components: dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, list) and len(v) == 2 and k != "scheduler":
            mod_cls = getattr(__import__(v[0]), v[1])
            match v[0]:
                case "diffusers":
                    conf = json.load(Path(hf_hub_download(uri, "config.json", subfolder=k)).open())
                    if k == "transformer":
                        conf["num_layers"] = 2
                        conf["num_single_layers"] = 2
                        conf["num_attention_heads"] = 4
                        if "patch_size" in conf and isinstance(conf["patch_size"], list):
                            conf["patch_size"] = conf["patch_size"][0:1] + [16] * (len(conf["patch_size"]) - 1)
                        if "cross_attention_dim" in conf:
                            conf["cross_attention_dim"] = 256
                        if "caption_projection_dim" in conf:
                            conf["caption_projection_dim"] = 1536 // (24 // 4)
                    else:
                        if "block_out_channels" in conf:
                            conf["block_out_channels"] = [32 * (n + 1) for n in range(len(conf["block_out_channels"]))]
                        if "down_block_types" in conf:
                            conf["transformer_layers_per_block"] = [
                                1 * (n + 1) for n in range(len(conf["down_block_types"]))
                            ]
                        if "num_res_blocks" in conf:
                            conf["num_res_blocks"] = 1
                        if "base_dim" in conf:
                            conf["base_dim"] = 16

                    components[k] = mod_cls.from_config(conf)
                case "transformers":
                    if hasattr(mod_cls, "config_class"):
                        components[k] = mod_cls(  # encoders
                            mod_cls.config_class.from_dict(
                                json.load(Path(hf_hub_download(uri, "config.json", subfolder=k)).open())
                                | {"num_hidden_layers": 1}
                            )
                        )
                    else:
                        components[k] = mod_cls.from_pretrained(uri, subfolder=k)  # tokenizers
                case _:
                    components[k] = None

    return cls.from_pretrained(uri, **components).to(dtype=torch.float32, device="cpu")


def test_sdxl_i2i() -> None:
    pipe = fake_pipe_init(StableDiffusionXLImg2ImgPipeline, "stabilityai/stable-diffusion-xl-base-1.0")

    b = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    assert isinstance(b, EulerDiscreteScheduler)

    compare_schedulers(
        pipe,
        SkrampleWrapperScheduler.from_diffusers_config(b),
        b,
        # 0,
        image=torch.zeros([1, 4, 32, 32]),
        num_inference_steps=50,
        strength=1 / 2,
        guidance_scale=1,
        prompt_embeds=torch.zeros([1, 77, 2048]),
        negative_prompt_embeds=torch.zeros([1, 77, 2048]),
        pooled_prompt_embeds=torch.zeros([1, 1280]),
        negative_pooled_prompt_embeds=torch.zeros([1, 1280]),
    )


def test_sd3_i2i() -> None:
    pipe = fake_pipe_init(StableDiffusion3Img2ImgPipeline, "bghira/sd3-reality-mix")  # no gate

    b = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    assert isinstance(b, FlowMatchEulerDiscreteScheduler)

    compare_schedulers(
        pipe,
        SkrampleWrapperScheduler.from_diffusers_config(b),
        b,
        # 0,
        height=256,
        width=256,
        guidance_scale=1,
        image=torch.zeros([1, 16, 256 // 16, 256 // 16]),
        num_inference_steps=50,
        strength=1 / 2,
        prompt_embeds=torch.zeros([1, 256 + 77, 4096]),
        pooled_prompt_embeds=torch.zeros([1, 768 + 1280]),
    )


def test_flux_i2i() -> None:
    pipe = fake_pipe_init(FluxImg2ImgPipeline, "mikeyandfriends/PixelWave_FLUX.1-dev_03")  # no gate

    b = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    assert isinstance(b, FlowMatchEulerDiscreteScheduler)

    fn, pipe._encode_vae_image = (
        pipe._encode_vae_image,
        lambda *_, **__: torch.zeros([1, 16, 32, 32]),
    )

    compare_schedulers(
        pipe,
        SkrampleWrapperScheduler.from_diffusers_config(b),
        b,
        # 0,
        height=256,
        width=256,
        image=torch.zeros([1, 1, 1]),
        num_inference_steps=50,
        strength=1 / 2,
        prompt_embeds=torch.zeros([1, 512, 4096]),
        pooled_prompt_embeds=torch.zeros([1, 768]),
    )

    pipe._encode_vae_image = fn


@pytest.mark.skip("Not implemented")
def test_wan_t2v() -> None:
    if WanPipeline is not None:
        pipe = fake_pipe_init(WanPipeline, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

        b = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        assert isinstance(b, FlowMatchEulerDiscreteScheduler)

        compare_schedulers(
            pipe,
            SkrampleWrapperScheduler.from_diffusers_config(b),
            b,
            # 0,
            height=256,
            width=256,
            guidance_scale=1,
            latents=torch.zeros([1, 16, 77, 32, 32]),
            num_inference_steps=25,
            prompt_embeds=torch.zeros([1, 512, 4096]),
        )


@pytest.mark.skip("Not implemented")
def test_ltx_t2v() -> None:
    if LTXPipeline is not None:
        pipe = fake_pipe_init(LTXPipeline, "Lightricks/LTX-Video-0.9.5")

        b = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        assert isinstance(b, FlowMatchEulerDiscreteScheduler)

        compare_schedulers(
            pipe,
            SkrampleWrapperScheduler.from_diffusers_config(b),
            b,
            # 0,
            height=256,
            width=256,
            guidance_scale=1,
            prompt="",
            num_inference_steps=25,
        )
