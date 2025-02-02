import json

import torch
from huggingface_hub import hf_hub_download


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
):
    assert a.isfinite().all(), message
    assert b.isfinite().all(), message
    delta = (a - b).abs().square().mean().item()
    assert delta <= margin, (message + " : " if message is not None else "") + f"{delta} <= {margin}"
