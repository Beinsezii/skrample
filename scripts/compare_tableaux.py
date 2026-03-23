#! /usr/bin/env python

import dataclasses
import json
import math
import sys
import time

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers.models.clip import CLIPTextModel, CLIPTokenizer

import skrample.scheduling as scheduling
from skrample.sampling import functional, models, tableaux


@dataclasses.dataclass(frozen=True)
class TableauResult:
    name: str
    stages: int

    steps: int
    nfes: int

    ssim: float
    clip: float
    geo: float


with torch.inference_mode():
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float16
    url: str = "Lykon/dreamshaper-8"
    cfg: float = 7

    schedule = scheduling.Scaled()

    noise: torch.Tensor = torch.randn(
        [1, 4, 512 // 8, 512 // 8], dtype=dtype, device=device, generator=torch.Generator(device).manual_seed(0)
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
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(dtype=dtype, device=device)  # type: ignore
    clip_processor = AutoProcessor.from_pretrained(clip_id)

    text_embeds: torch.Tensor = text_encoder(
        tokenizer(
            ["bright colorful fantasy art of a kitten in a field of rainbow flowers", "blurry, noisy, cropped"],
            padding="max_length",
            return_tensors="pt",
        ).input_ids.to(device=device)
    ).last_hidden_state

    def call_model(x: torch.Tensor, t: float, s: float) -> torch.Tensor:
        conditioned, unconditioned = model(
            x.expand([x.shape[0] * 2, *x.shape[1:]]).to(model.dtype),
            t,
            text_embeds,
        ).sample.chunk(2)
        return (conditioned + (cfg - 1) * (conditioned - unconditioned)).to(torch.float64)

    reference_steps: int = 200
    reference_provider: tableaux.TableauProvider = tableaux.RK1.Euler
    reference_tensor = functional.RKUltra(order=42, providers={42: reference_provider}).sample_model(
        noise, model=call_model, model_transform=models.NoiseModel(), schedule=schedule, steps=reference_steps
    )

    reference_image: np.ndarray = (
        image_encoder.decode(reference_tensor.to(image_encoder.dtype) / image_encoder.config.scaling_factor)  # type: ignore
        .sample[0]  # pyright: ignore
        .permute(1, 2, 0)
        .float()
        .add(1)
        .div(2)
        .cpu()
        .numpy()
    )
    Image.fromarray((reference_image * 255).clip(0, 255).astype(np.uint8)).save("reference.png")

    reference_features = clip_model(
        **clip_processor(images=[reference_image.clip(0, 1)], return_tensors="pt").to(device=device, dtype=dtype)
    ).image_embeds[0]
    reference_features /= reference_features.norm(p=2)

    def measure_provider(provider: tableaux.TableauProvider, steps: int) -> TableauResult:
        stages = len(provider.tableau().stages)
        clip_score: float = 0
        ssim_score: float = 0
        geo_score: float = 0
        nfes: int = 0

        def call_model_nfes(x: torch.Tensor, t: float, s: float) -> torch.Tensor:
            nonlocal nfes
            nfes += 1
            return call_model(x, t, s)

        clock = time.perf_counter()

        measured_tensor = functional.RKUltra(order=stages, providers={stages: provider}).sample_model(
            noise,
            model=call_model_nfes,
            model_transform=models.NoiseModel(),
            schedule=schedule,
            steps=steps,
        )

        measured_image: np.ndarray = (
            image_encoder.decode(measured_tensor.to(image_encoder.dtype) / image_encoder.config.scaling_factor)  # type: ignore
            .sample[0]  # pyright: ignore
            .permute(1, 2, 0)
            .float()
            .add(1)
            .div(2)
            .cpu()
            .numpy()
        )

        clip_clock = time.perf_counter()
        print(
            f"Computed {steps} steps {provider!s} in {round((clip_clock - clock) * 1000)}ms",
            file=sys.stderr,
        )

        if np.isfinite(measured_image).all():
            measured_features = clip_model(
                **clip_processor(
                    images=[measured_image.clip(0, 1)],
                    return_tensors="pt",
                ).to(device=device, dtype=dtype)
            ).image_embeds[0]
            measured_features /= measured_features.norm(p=2)

            ssim_score = ssim(
                reference_image,
                measured_image,
                channel_axis=-1,
                data_range=(reference_image.max() - reference_image.min()).item(),
            ).item()  # pyright: ignore
            clip_score: float = torch.dot(reference_features, measured_features).item()

            try:
                geo_score = math.sqrt(ssim_score * clip_score)
            except ValueError:
                print(f"Cannot comptue geomean from {ssim_score} * {clip_score}", file=sys.stderr)
                geo_score = 0

            # ssim_clock = time.perf_counter()
            # print(
            #     f"Scored {clip_score:.2f} {ssim_score:.2f} in {round((ssim_clock - clip_clock) * 1000)}ms",
            #     file=sys.stderr,
            # )

        return TableauResult(
            name=str(provider),
            stages=stages,
            steps=steps,
            nfes=nfes,
            ssim=ssim_score,
            clip=clip_score,
            geo=geo_score,
        )

    target_score = measure_provider(tableaux.RK1.Euler, 50)
    results: list[TableauResult] = []

    for target_stages in range(1, 99):
        for provider in [
            tableaux.RK1,
            tableaux.RK2,
            tableaux.RK3,
            tableaux.RK4,
            tableaux.RKZ,
            tableaux.RKE2,
            tableaux.RKE3,
            tableaux.RKE5,
            tableaux.SSP,
            # These never win, not even close
            # tableaux.WSO,
            # tableaux.Shanks1965,
        ]:
            for variant in provider:
                stages = len(variant.tableau()[0])
                if stages != target_stages:
                    continue

                steps: int = 0
                result: TableauResult | None = None

                while not result or result.ssim < target_score.ssim:
                    steps += 1
                    result = measure_provider(variant, steps)
                    if steps * stages >= 200:
                        break

                results.append(result)

    json.dump([dataclasses.asdict(r) for r in sorted(results, key=lambda r: (r.stages, r.nfes, -r.geo))], sys.stdout)
