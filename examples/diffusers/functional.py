#! /usr/bin/env python

from typing import ClassVar

import torch
from diffusers.modular_pipelines.components_manager import ComponentsManager
from diffusers.modular_pipelines.flux.denoise import FluxDenoiseStep, FluxLoopDenoiser
from diffusers.modular_pipelines.flux.modular_blocks import TEXT2IMAGE_BLOCKS
from diffusers.modular_pipelines.flux.modular_pipeline import FluxModularPipeline
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from tqdm import tqdm

import skrample.scheduling as scheduling
from skrample.common import predict_flow
from skrample.diffusers import SkrampleWrapperScheduler
from skrample.sampling import functional, models, structured
from skrample.sampling.interface import StructuredFunctionalAdapter

model_id = "black-forest-labs/FLUX.1-dev"

blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

schedule = scheduling.FlowShift(scheduling.Linear(), shift=2)
wrapper = SkrampleWrapperScheduler(
    sampler=structured.Euler(), schedule=schedule, predictor=predict_flow, allow_dynamic=False
)

# Equivalent to structured example
sampler = StructuredFunctionalAdapter(schedule, structured.DPM(order=2, add_noise=True))
# Native functional example
sampler = functional.RKUltra(schedule, 4)
# Dynamic model calls
sampler = functional.FastHeun(schedule)
# Dynamic step sizes
sampler = functional.RKMoire(schedule)


class FunctionalDenoise(FluxDenoiseStep):
    # Exclude the after_denoise block
    block_classes: ClassVar[list[type[ModularPipelineBlocks]]] = [FluxLoopDenoiser]
    block_names: ClassVar[list[str]] = ["denoiser"]

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if isinstance(sampler, functional.FunctionalHigher):
            block_state["num_inference_steps"] = sampler.adjust_steps(block_state["num_inference_steps"])
        progress = tqdm(total=block_state["num_inference_steps"])

        i = 0

        def call_model(sample: torch.Tensor, timestep: float, sigma: float) -> torch.Tensor:
            nonlocal i, components, block_state, progress
            block_state["latents"] = sample
            components, block_state = self.loop_step(
                components,
                block_state,  # type: ignore
                i=i,
                t=sample.new_tensor([timestep] * len(sample)),
            )
            return block_state["noise_pred"]  # type: ignore

        def sample_callback(x: torch.Tensor, n: int, t: float, s: float) -> None:
            nonlocal i
            progress.update(n + 1 - progress.n)
            i = n + 1

        block_state["latents"] = sampler.sample_model(
            sample=block_state["latents"],
            model=call_model,
            model_transform=models.FlowModel,
            steps=block_state["num_inference_steps"],
            callback=sample_callback,
        )

        self.set_block_state(state, block_state)  # type: ignore
        return components, state  # type: ignore


blocks.sub_blocks["denoise"] = FunctionalDenoise()

cm = ComponentsManager()
cm.enable_auto_cpu_offload()
pipe = blocks.init_pipeline(components_manager=cm)
pipe.load_components(["text_encoder"], repo=model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
pipe.load_components(["tokenizer"], repo=model_id, subfolder="tokenizer")
pipe.load_components(["text_encoder_2"], repo=model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
pipe.load_components(["tokenizer_2"], repo=model_id, subfolder="tokenizer_2")
pipe.load_components(["transformer"], repo=model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe.load_components(["vae"], repo=model_id, subfolder="vae", torch_dtype=torch.bfloat16)

pipe.register_components(scheduler=wrapper)


pipe(  # type: ignore
    prompt="sharp, high dynamic range photograph of a kitten on a beach of rainbow pebbles",
    generator=torch.Generator("cpu").manual_seed(42),
    width=1024,
    height=1024,
    num_inference_steps=25,
    guidance_scale=2.5,
).get("images")[0].save("diffusers_functional.png")
