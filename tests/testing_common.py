import json

import torch
from huggingface_hub import hf_hub_download

FLOW_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "flow_shift": 3.0,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "prediction_type": "flow_prediction",
    "shift": 3.0,
    "use_dynamic_shifting": True,
}
SCALED_CONFIG = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "use_karras_sigmas": False,
}


def hf_scheduler_config(
    hf_repo: str, filename: str = "scheduler_config.json", subfolder: str | None = "scheduler"
) -> dict:
    with open(hf_hub_download(hf_repo, filename, subfolder=subfolder), mode="r") as jfile:
        return json.load(jfile)


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    message: str | None = "",
    margin: float = 1e-4,
) -> None:
    assert a.isfinite().all(), message
    assert b.isfinite().all(), message
    delta = (a - b).abs().square().mean().item()
    assert delta <= margin, f"{delta} <= {margin}" + (" | " + message if message is not None else "")
