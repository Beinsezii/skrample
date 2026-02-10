import numpy as np
import pytest
import torch

from skrample.common import MergeStrategy, Step, bashforth, sigmoid, softmax, spowf


def test_bashforth() -> None:
    for n, coeffs in enumerate(
        np.array(c) for c in ((1,), (3 / 2, -1 / 2), (23 / 12, -4 / 3, 5 / 12), (55 / 24, -59 / 24, 37 / 24, -3 / 8))
    ):
        assert np.allclose(coeffs, np.array(bashforth(n + 1)), atol=1e-12, rtol=1e-12)


def test_sigmoid() -> None:
    items = spowf(torch.linspace(-2, 2, 9, dtype=torch.float64), 2)
    a = torch.sigmoid(items)
    b = sigmoid(items)
    assert torch.allclose(a, b, rtol=0, atol=1e-12), (a.tolist(), b.tolist())


def test_softmax() -> None:
    items = spowf(torch.linspace(-2, 2, 9, dtype=torch.float64), 2)
    a = torch.softmax(items, 0)
    b = torch.tensor(softmax(tuple(items)), dtype=torch.float64)
    assert torch.allclose(a, b, rtol=0, atol=1e-12), (a.tolist(), b.tolist())


def test_merge() -> None:
    array_deltas: list[tuple[list[int], list[int], list[int], list[int]]] = [
        (list(range(0, 11)), list(range(0, 15, 2)), list(range(1, 10, 2)), list(range(12, 15, 2))),
        (list(range(4, 15)), list(range(0, 11, 2)), list(range(5, 11, 2)) + list(range(11, 15)), list(range(0, 4, 2))),
    ]
    for a, b, aX, bX in array_deltas:
        tests: list[tuple[list[int], list[int], MergeStrategy, list[int]]] = [
            (a, b, MergeStrategy.Ours, a),
            (b, a, MergeStrategy.Ours, b),
            (a, b, MergeStrategy.Theirs, b),
            (b, a, MergeStrategy.Theirs, a),
            (a, b, MergeStrategy.After, a + b),
            (b, a, MergeStrategy.After, b + a),
            (a, b, MergeStrategy.Before, b + a),
            (b, a, MergeStrategy.Before, a + b),
            (a, b, MergeStrategy.UniqueBefore, b + aX),
            (b, a, MergeStrategy.UniqueBefore, a + bX),
            (a, b, MergeStrategy.UniqueAfter, a + bX),
            (b, a, MergeStrategy.UniqueAfter, b + aX),
        ]
        for ours, theirs, ms, merged in tests:
            assert ms.merge(ours, theirs) == merged, f"{ours} {ms} {theirs} : {merged}"


STEP_STEPS: int = 31


@pytest.mark.parametrize("n", range(STEP_STEPS + 1))
def test_step_range(n: int) -> None:
    step = Step.from_int(n, STEP_STEPS)

    assert abs(step.amount() - STEP_STEPS) < 1e-8
    assert abs(step.position() - n) < 1e-8
    assert Step(*reversed(step)).normal() == step

    assert abs(step.offset(-4).position() - (n - 4)) < 1e-8
    assert abs(step.offset(+4).position() - (n + 4)) < 1e-8

    assert step.offset(STEP_STEPS / 2).clamp().position() + 1 <= STEP_STEPS + 1e-8
    assert step.offset(STEP_STEPS / -2).clamp().position() >= 0
