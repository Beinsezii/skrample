#! /usr/bin/env python

import json

import huggingface_hub as hf
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import skrample.pytorch.noise as noise
import skrample.scheduling as scheduling
from skrample.common import Predictor, predict_epsilon, predict_sample, predict_velocity
from skrample.sampling import functional, models, structured
from skrample.sampling.interface import StructuredFunctionalAdapter

with torch.inference_mode():
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float16
    steps: int = 15
    cfg: float = 8
    seed = torch.Generator("cpu").manual_seed(0)
    prompts = ["dreamy analog photograph of a kitten in a stained glass church", "blurry, noisy, cropped"]

    schedule = scheduling.Scaled()

    sampler_snr = StructuredFunctionalAdapter(schedule, structured.DPM(order=1))
    sampler_df = functional.RKUltra(schedule, order=1)

    base = "stabilityai/stable-diffusion-xl-base-1.0"

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tokenizer_2: CLIPTokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    text_encoder: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
        base, subfolder="text_encoder", device_map=device, torch_dtype=dtype
    )
    text_encoder_2: CLIPTextModel = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder_2", device_map=device, torch_dtype=dtype
    )
    image_encoder: AutoencoderKL = AutoencoderKL.from_pretrained(  # type: ignore
        base, subfolder="vae", device_map=device, torch_dtype=torch.float32
    )

    text_embeds: torch.Tensor = text_encoder(
        tokenizer(prompts, padding="max_length", return_tensors="pt").input_ids.to(device=device),
        output_hidden_states=True,
    ).hidden_states[-2]
    te2_out = text_encoder_2(
        tokenizer_2(prompts, padding="max_length", return_tensors="pt").input_ids.to(device=device),
        output_hidden_states=True,
    )
    text_embeds = torch.cat([text_embeds, te2_out.hidden_states[-2]], dim=-1)
    pooled_embeds: torch.Tensor = te2_out.pooler_output

    time_embeds = text_embeds.new([[4096, 4096, 0, 0, 4096, 4096]]).repeat(2, 1)

    configs: tuple[tuple[models.ModelTransform, Predictor, str, str], ...] = (
        (models.EpsilonModel, predict_epsilon, base, ""),
        (models.VelocityModel, predict_velocity, "terminusresearch/terminus-xl-velocity-v2", ""),
        (models.XModel, predict_sample, "ByteDance/SDXL-Lightning", "sdxl_lightning_1step_unet_x0.safetensors"),
    )

    for transform, predictor, url, weights in configs:
        model_steps = 1 if transform is models.XModel else steps
        model_cfg = 1 if transform is models.XModel else cfg

        if weights:
            model: UNet2DConditionModel = UNet2DConditionModel.from_config(  # type: ignore
                json.load(open(hf.hf_hub_download(base, "config.json", subfolder="unet"))),
                device_map=device,
                torch_dtype=dtype,
            )
            model.load_state_dict(load_file(hf.hf_hub_download(url, weights)))
            model = model.to(device=device, dtype=dtype)  # pyright: ignore [reportCallIssue]
        else:
            model: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(  # type: ignore
                url, subfolder="unet", device_map=device, torch_dtype=dtype
            )

        def call_model(x: torch.Tensor, t: float, s: float) -> torch.Tensor:
            conditioned, unconditioned = model(
                x.expand([x.shape[0] * 2, *x.shape[1:]]),
                t,
                text_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_embeds},
            ).sample.chunk(2)
            return conditioned + (model_cfg - 1) * (conditioned - unconditioned)

        rng = noise.Random.from_inputs((1, 4, 128, 128), seed.clone_state())
        bar = tqdm(total=model_steps)
        sample = sampler_snr.generate_model(
            model=call_model,
            model_transform=transform,
            steps=model_steps,
            rng=lambda: rng.generate().to(dtype=dtype, device=device),
            callback=lambda x, n, t, s: bar.update(n + 1 - bar.n),
        )

        image: torch.Tensor = image_encoder.decode(
            sample.to(dtype=image_encoder.dtype) / image_encoder.config.scaling_factor  # type: ignore
        ).sample[0]  # type: ignore
        Image.fromarray(
            ((image + 1) * (255 / 2)).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
        ).save(f"{predictor.__name__}.png")

        rng = noise.Random.from_inputs((1, 4, 128, 128), seed.clone_state())
        bar = tqdm(total=sampler_df.adjust_steps(model_steps))
        sample = sampler_df.generate_model(
            model=call_model,
            model_transform=transform,
            steps=sampler_df.adjust_steps(model_steps),
            rng=lambda: rng.generate().to(dtype=dtype, device=device),
            callback=lambda x, n, t, s: bar.update(n + 1 - bar.n),
        )

        image: torch.Tensor = image_encoder.decode(
            sample.to(dtype=image_encoder.dtype) / image_encoder.config.scaling_factor  # type: ignore
        ).sample[0]  # type: ignore
        Image.fromarray(
            ((image + 1) * (255 / 2)).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
        ).save(f"{transform.__name__}.png")

        model = model.to(device="meta")  # pyright: ignore [reportCallIssue]
