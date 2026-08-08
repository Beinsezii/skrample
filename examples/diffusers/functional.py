#! /usr/bin/env python


import torch
import tqdm
from diffusers.modular_pipelines.flux2.denoise import Flux2KleinBaseDenoiseStep
from diffusers.modular_pipelines.flux2.modular_pipeline import Flux2KleinBaseModularPipeline, Flux2ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import BlockState, ModularPipeline, PipelineState

from skrample import scheduling
from skrample.common import DeltaPoint
from skrample.sampling import functional, interface, models, structured


class SkrampleKleinBaseDenoiseLoop(Flux2KleinBaseDenoiseStep):
    # Remove after denoise
    block_classes = Flux2KleinBaseDenoiseStep.block_classes[:-1]
    block_names = Flux2KleinBaseDenoiseStep.block_names[:-1]

    def __init__(
        self,
        sampler: functional.FunctionalSampler | structured.StructuredSampler,
        schedule: scheduling.SkrampleSchedule,
        model: models.DiffusionModel = models.FlowModel(),
    ) -> None:
        super().__init__()
        self.skrample_sampler: functional.FunctionalSampler | structured.StructuredSampler = sampler
        self.skrample_schedule: scheduling.SkrampleSchedule = schedule
        self.skrample_model: models.DiffusionModel = model

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state: BlockState = self.get_block_state(state)  # type: ignore # diffusers dumb

        if isinstance(self.skrample_sampler, functional.FunctionalHigher):
            block_state["num_inference_steps"] = self.skrample_sampler.adjust_steps(block_state["num_inference_steps"])  # pyright: ignore [reportArgumentType]

        with tqdm.tqdm(total=block_state["num_inference_steps"]) as progress_bar:
            i: int = 0

            def callback(_x: torch.Tensor, n: int, _p: DeltaPoint) -> None:
                nonlocal i
                progress_bar.update(n - i)
                i = n

            def model_eval(x: torch.Tensor, t: float, s: float, a: float) -> torch.Tensor:
                nonlocal i, components, block_state
                block_state["latents"] = x
                components, block_state = self.loop_step(
                    components,
                    block_state,  # type: ignore
                    i=i,
                    t=x.new_tensor([t] * len(x)),
                )
                return block_state["noise_pred"]  # pyright: ignore [reportReturnType]

            block_state["latents"] = (
                self.skrample_sampler
                if isinstance(self.skrample_sampler, functional.FunctionalSampler)
                else interface.StructuredFunctionalAdapter(self.skrample_sampler)
            ).sample_model(
                sample=block_state["latents"],  # pyright: ignore[reportArgumentType]
                model=model_eval,
                model_transform=self.skrample_model,
                schedule=self.skrample_schedule,
                steps=block_state["num_inference_steps"],  # pyright: ignore[reportArgumentType]
                callback=callback,
            )

        self.set_block_state(state, block_state)
        return components, state  # type: ignore # diffusers is so so dumb


with torch.inference_mode():
    # Equivalent to structured example
    sampler = structured.DPM(order=2, stochasticity=True)
    # Native functional example
    sampler = functional.RKUltra(4)
    # # Dynamic step sizes
    sampler = functional.RKMoire()

    schedule = scheduling.Sinner(scheduling.Linear())

    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda")
    model = "black-forest-labs/FLUX.2-klein-base-4B"

    pipe: Flux2KleinBaseModularPipeline = ModularPipeline.from_pretrained(model)  # pyright: ignore
    pipe.load_components(torch_dtype=dtype)
    pipe.to(device)

    skrample_denoise = SkrampleKleinBaseDenoiseLoop(sampler, schedule)
    pipe._blocks.sub_blocks["denoise"].sub_blocks["text2image"].sub_blocks["denoise"] = skrample_denoise  # type: ignore

    # Doesn't take it from the call...? wtf?
    pipe.guider.enable()
    pipe.guider.guidance_scale = 3

    pipe(
        num_inference_steps=30,
        prompt="""
Analogue portrait photograph of a woman in a stained glass church
She is wearing gothic plate armor and has short, curly blonde hair.
The photo is softly lit, with the light in the image being provided by multicolored rays coming from the church windows.
High resolution technicolor photograph.
""",
        width=1280,
        height=832,
        generator=torch.Generator(device=device).manual_seed(42),  # pyright: ignore[reportPrivateImportUsage] # something's changed in pyright and this is private now ig
    ).images[0].save("klein_functional.png")  # pyright: ignore[reportAttributeAccessIssue]
