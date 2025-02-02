import torch


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
