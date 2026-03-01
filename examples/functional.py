#! /usr/bin/env python

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers.models.clip import CLIPTextModel, CLIPTokenizer

import skrample.pytorch.noise as noise
import skrample.scheduling as scheduling
from skrample.sampling import functional, models, structured
from skrample.sampling.interface import StructuredFunctionalAdapter

with torch.inference_mode():
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float16
    url: str = "Lykon/dreamshaper-8"
    seed = torch.Generator("cpu").manual_seed(0)
    steps: int = 25
    cfg: float = 3

    schedule = scheduling.Karras(scheduling.Scaled())

    # Equivalent to structured example
    sampler = StructuredFunctionalAdapter(structured.DPM(order=2, add_noise=True))
    # Native functional example
    sampler = functional.RKUltra(4)
    # Dynamic model calls
    sampler = functional.FastHeun()
    # Dynamic step sizes
    sampler = functional.RKMoire()

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(url, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        url, subfolder="text_encoder", device_map=device, torch_dtype=dtype
    )
    model: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        url, subfolder="unet", device_map=device, torch_dtype=dtype
    )
    image_encoder: AutoencoderKL = AutoencoderKL.from_pretrained(
        url, subfolder="vae", device_map=device, torch_dtype=dtype
    )

    text_embeds: torch.Tensor = text_encoder(
        tokenizer(
            "bright colorful fantasy art of a kitten in a field of rainbow flowers",
            padding="max_length",
            return_tensors="pt",
        ).input_ids.to(device=device)
    ).last_hidden_state

    def call_model(x: torch.Tensor, t: float, s: float) -> torch.Tensor:
        conditioned, unconditioned = model(
            x.expand([x.shape[0] * 2, *x.shape[1:]]),
            t,
            torch.cat([text_embeds, torch.zeros_like(text_embeds)]),
        ).sample.chunk(2)
        return conditioned + (cfg - 1) * (conditioned - unconditioned)

    if isinstance(sampler, functional.FunctionalHigher):
        steps = sampler.adjust_steps(steps)

    rng = noise.Random.from_inputs((1, 4, 80, 80), seed)
    bar = tqdm(total=steps)
    sample = sampler.generate_model(
        model=call_model,
        model_transform=models.NoiseModel(),
        schedule=schedule,
        steps=steps,
        rng=lambda: rng.generate().to(dtype=dtype, device=device),
        callback=lambda x, n, t, s: bar.update(n + 1 - bar.n),
    )

    image: torch.Tensor = image_encoder.decode(sample / image_encoder.config.scaling_factor).sample[0]  # type: ignore
    Image.fromarray(
        ((image + 1) * (255 / 2)).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    ).save("functional.png")
