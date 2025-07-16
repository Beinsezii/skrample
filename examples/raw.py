#! /usr/bin/env python

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers.models.clip import CLIPTextModel, CLIPTokenizer

import skrample.common
import skrample.sampling as sampling
import skrample.scheduling as scheduling

with torch.inference_mode():
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float16
    url: str = "Lykon/dreamshaper-8"
    seed = torch.Generator("cpu").manual_seed(0)
    steps: int = 25
    cfg: float = 3

    schedule: scheduling.SkrampleSchedule = scheduling.Karras(scheduling.Scaled())
    sampler: sampling.SkrampleSampler = sampling.DPM(order=2, add_noise=True)
    predictor: skrample.common.Predictor = skrample.common.predict_epsilon

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(url, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        url, subfolder="text_encoder", device_map=device, torch_dtype=dtype
    )
    model: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(  # type: ignore
        url, subfolder="unet", device_map=device, torch_dtype=dtype
    )
    image_encoder: AutoencoderKL = AutoencoderKL.from_pretrained(  # type: ignore
        url, subfolder="vae", device_map=device, torch_dtype=dtype
    )

    text_embeds: torch.Tensor = text_encoder(
        tokenizer(
            "bright colorful fantasy art of a kitten in a field of rainbow flowers",
            padding="max_length",
            return_tensors="pt",
        ).input_ids.to(device=device)
    ).last_hidden_state

    sample: torch.Tensor = torch.randn([1, 4, 80, 80], generator=seed).to(dtype=dtype, device=device)
    previous: list[sampling.SKSamples[torch.Tensor]] = []

    for n, (timestep, sigma) in enumerate(tqdm(schedule.schedule(steps))):
        conditioned, unconditioned = model(
            sample.expand([sample.shape[0] * 2, *sample.shape[1:]]),
            timestep,
            torch.cat([text_embeds, torch.zeros_like(text_embeds)]),
        ).sample.chunk(2)
        model_output: torch.Tensor = conditioned + (cfg - 1) * (conditioned - unconditioned)

        prediction = predictor(sample, model_output, sigma, schedule.sigma_transform)

        sampler_output = sampler.sample(
            sample=sample,
            prediction=prediction,
            step=n,
            sigma_schedule=schedule.sigmas(steps),
            sigma_transform=schedule.sigma_transform,
            noise=torch.randn(sample.shape, generator=seed).to(dtype=sample.dtype, device=sample.device),
            previous=tuple(previous),
        )

        previous.append(sampler_output)
        sample = sampler_output.final

    image: torch.Tensor = image_encoder.decode(sample / image_encoder.config.scaling_factor).sample[0]  # type: ignore
    Image.fromarray(
        ((image + 1) * (255 / 2)).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    ).save("raw.png")
