#! /usr/bin/env python

import dataclasses
import json
import math
import sys

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from skimage.metrics import structural_similarity as ssim
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers.models.clip import CLIPTextModel, CLIPTokenizer

import skrample.scheduling as scheduling
from skrample.sampling import functional, models, tableaux


@dataclasses.dataclass(frozen=True)
class TableauxResult:
    name: str
    stages: int

    steps: int

    ssim: float
    clip: float
    geo: float


with torch.inference_mode():
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float16
    url: str = "Lykon/dreamshaper-8"
    cfg: float = 3

    schedule = scheduling.Scaled()

    noise: torch.Tensor = torch.randn(
        [1, 4, 224 // 8, 224 // 8 + 8], dtype=dtype, device=device, generator=torch.Generator(device).manual_seed(0)
    )

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(url, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        url, subfolder="text_encoder", device_map=device, torch_dtype=dtype
    )
    model: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        url, subfolder="unet", device_map=device, torch_dtype=dtype
    )
    # model.compile(mode="reduce-overhead", fullgraph=True, dynamic=True)
    image_encoder: AutoencoderKL = AutoencoderKL.from_pretrained(
        url, subfolder="vae", device_map=device, torch_dtype=dtype
    )

    clip_id = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(dtype=dtype, device=device)
    clip_processor = AutoProcessor.from_pretrained(clip_id)

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

    reference_tensor = functional.RKUltra(order=1).sample_model(
        noise, model=call_model, model_transform=models.NoiseModel(), schedule=schedule, steps=200
    )

    reference_image: torch.Tensor = (
        image_encoder.decode(reference_tensor / image_encoder.config.scaling_factor)  # type: ignore
        .sample[0]  # pyright: ignore
        .permute(1, 2, 0)
        .float()
        .add(1)
        .div(2)
        .cpu()
        .numpy()
    )

    reference_features = clip_model(
        **clip_processor(images=[reference_image.clip(0, 1)], return_tensors="pt").to(device=device, dtype=dtype)
    ).image_embeds[0]
    reference_features /= reference_features.norm(p=2)

    results: list[TableauxResult] = []

    for target_stages in (2, 3, 4, 6, 7):
        for provider in [
            tableaux.RK2,
            tableaux.RK3,
            tableaux.RK4,
            tableaux.RKZ,
            tableaux.RKE2,
            tableaux.RKE3,
            tableaux.RKE5,
            tableaux.Shanks1965,
        ]:
            for variant in provider:
                stages = len(variant.tableau()[0])
                if stages != target_stages:
                    continue
                for steps in (max(round(x / target_stages), n + 2) for n, x in enumerate([10, 25, 50])):
                    measured_tensor = functional.RKUltra(order=stages, providers={stages: variant}).sample_model(
                        noise, model=call_model, model_transform=models.NoiseModel(), schedule=schedule, steps=steps
                    )

                    measured_image: torch.Tensor = (
                        image_encoder.decode(measured_tensor / image_encoder.config.scaling_factor)  # type: ignore
                        .sample[0]  # pyright: ignore
                        .permute(1, 2, 0)
                        .float()
                        .add(1)
                        .div(2)
                        .cpu()
                        .numpy()
                    )

                    measured_features = clip_model(
                        **clip_processor(
                            images=[measured_image.clip(0, 1)],
                            return_tensors="pt",
                        ).to(device=device, dtype=dtype)
                    ).image_embeds[0]
                    measured_features /= measured_features.norm(p=2)

                    ssim_score: float = ssim(
                        reference_image,
                        measured_image,
                        channel_axis=-1,
                        data_range=(reference_image.max() - reference_image.min()).item(),
                    ).item()  # pyright: ignore
                    clip_score: float = torch.dot(reference_features, measured_features).item()

                    results.append(
                        TableauxResult(
                            name=f"{provider.__name__}.{variant.name}",
                            stages=stages,
                            steps=steps,
                            ssim=ssim_score,
                            clip=clip_score,
                            geo=math.sqrt(ssim_score * clip_score),
                        )
                    )

    json.dump([dataclasses.asdict(r) for r in sorted(results, key=lambda r: (r.stages, r.steps, -r.geo))], sys.stdout)
