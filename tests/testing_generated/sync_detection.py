"""Detect and count host syncs (DtoH), with or without an accelerator.

    set_sync_debug_mode("error")   torch builtin, needs a device, raises inside the blocking op
    no_host_sync()                 no device, fake tensors, fails on the first sync
    SyncCounter                    no device, real cpu tensors, counts
    count_blocking_calls()         device, counts blocking runtime calls through the profiler

Measured on rocm7.14 / torch2.14:
- set_sync_debug_mode catches item, float, int, bool, tolist, repr, f"{t}", cpu, bool mask indexing,
  nonzero, unique, masked_select, allclose and blocking HtoD copies, but not synchronize(). "warn"
  is TORCH_WARN_ONCE (5 syncs at one site give 1 warning) so it cannot count, and it calls
  _lazy_init so it raises rather than degrading when there is no device.
- no_host_sync needs no list of syncing methods: meta kernels raise DataDependentOutputException on
  reads of contents and DynamicOutputShapeException on data dependent shapes. Device crossings are
  pure metadata and repr/f"{t}" looks inert, both need _Crossings or the other tools.
- torch.compile(fullgraph=True) detects nothing: .item() breaks no graph, concretized on cpu and
  lowered to aten::_local_scalar_dense on device. torch.export fails only when a read guards shape
  or control flow.
- counting needs the read to return a value, so SyncCounter runs on real cpu tensors, not fake ones.
"""

import itertools
import traceback
from collections import Counter
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, ClassVar

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

# .to()/.cpu() report the destination as a kwarg, copy_ as a second positional tensor
COPY_OPS: frozenset[str] = frozenset({"aten::copy_", "aten::_to_copy", "aten::to", "aten::cpu"})
# an index on another device costs an implicit host->device copy that never appears as a copy op
INDEX_OPS: frozenset[str] = frozenset({"aten::index", "aten::_index_put_impl_"})
# The two layers are disjoint except float/int/bool, which are a method and a nested
# aten::_local_scalar_dense (SyncCounter._inside suppresses the inner one). item() is op only,
# tolist() and repr() are method only.
READ_METHODS: frozenset[str] = frozenset({"__float__", "__int__", "__bool__", "__repr__", "tolist"})
# .to() counts only when it names a device, .to(torch.float32) is free
TRANSFER_METHODS: frozenset[str] = frozenset({"cpu", "cuda", "numpy", "to"})
_INTERNAL: frozenset[str] = frozenset({"__torch_function__", "__torch_dispatch__", "hit"})
SYNC_OPS: frozenset[str] = frozenset(
    {"aten::_local_scalar_dense", "aten::nonzero", "aten::masked_select", "aten::_unique2", "aten::multinomial"}
)
# kineto names for the calls that really block, same spelling on cuda and rocm past the prefix
SYNC_RUNTIME: tuple[str, ...] = ("StreamSynchronize", "DeviceSynchronize", "Memcpy")


def _names_device(args: tuple, kwargs: dict) -> bool:
    if kwargs.get("device") is not None:
        return True
    return bool(args) and isinstance(args[0], str | torch.device)


def _crosses(op: str, args: tuple, kwargs: dict) -> str | None:
    src = args[0] if args and isinstance(args[0], torch.Tensor) else None
    if src is None:
        return None
    if op in COPY_OPS:
        dst: Any = args[1] if op == "aten::copy_" and len(args) > 1 else kwargs.get("device")
        dst = dst.device if isinstance(dst, torch.Tensor) else dst
        if isinstance(dst, torch.device) and dst != src.device:
            return f"{op} {src.device} -> {dst}"
    if op in INDEX_OPS and len(args) > 1:
        for index in args[1] if isinstance(args[1], (list | tuple)) else (args[1],):
            if isinstance(index, torch.Tensor) and index.device != src.device:
                return f"{op} indexed by {index.device} tensor, {src.device} base"
    return None


class _Crossings(TorchDispatchMode):
    """strict raises on the first device crossing, otherwise counts into owner."""

    def __init__(self, strict: bool = False, owner: "SyncCounter | None" = None) -> None:
        self.strict = strict
        self.owner = owner
        super().__init__()

    def __torch_dispatch__(
        self,
        func: Any,  # noqa: ANN401
        types: Any,  # noqa: ANN401
        args: Any = (),  # noqa: ANN401
        kwargs: Any = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        kwargs = kwargs or {}
        op: str = getattr(func, "_name", str(func)).split(".")[0]
        nested = self.owner is not None and self.owner._inside > 0
        detail = _crosses(op, args, kwargs)
        if detail is None and not self.strict and not nested and op in SYNC_OPS:
            detail = op
        if detail is not None:
            if self.strict:
                msg = f"host sync: {detail}"
                raise AssertionError(msg)
            assert self.owner is not None
            self.owner.hit(str(func))
        return func(*args, **kwargs)


@contextmanager
def no_host_sync() -> Generator[None]:
    """Run a region on fake accelerator tensors, raising on anything that would block the host.
    Real cpu tensors pass through untouched.

        with no_host_sync():
            sample = torch.randn([1, 4, 8], device="cuda")
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    with _Crossings(strict=True), FakeTensorMode(allow_non_fake_inputs=True):
        yield


class SyncCounter(TorchFunctionMode):
    """Count syncs instead of failing, so a test can assert an exact number and where it happens.

        counter = SyncCounter()
        with counter:
            for _ in range(28):
                counter.mark()
                wrapper.step(model_output=sample, timestep=step, sample=sample)
        assert counter.total == 1
        assert counter.per_step == [0] * 27 + [1]

    Runs on real cpu tensors, where .cpu()/.numpy()/.to(device=) are free: they go to `transfers`
    instead of `total` unless count_transfers=True, and are not comparable to a device count, where
    a .to(device=) already on device is free and one pageable copy costs two runtime calls.
    4 steps of SkrampleWrapperScheduler: 4 total, 8 transfers, 13 blocking calls on the GPU.
    """

    enabled_globally: ClassVar[bool] = False

    def __init__(self, count_transfers: bool = False) -> None:
        super().__init__()
        self.count_transfers = count_transfers
        self.total = 0
        self.transfers = 0
        self.per_site: Counter[str] = Counter()
        self.marks: list[int] = []
        self._inside = 0
        self._ops = _Crossings(owner=self)

    def __enter__(self) -> "SyncCounter":
        self._ops.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        self._ops.__exit__(exc_type, exc_val, exc_tb)
        super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(
        self,
        func: Callable,
        types: Any,  # noqa: ANN401
        args: Any = (),  # noqa: ANN401
        kwargs: Any = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        name = getattr(func, "__name__", "")
        read = name in READ_METHODS
        transfer = name in TRANSFER_METHODS and (name != "to" or _names_device(args, kwargs or {}))
        if transfer:
            self.transfers += 1
        if not (read or transfer):
            return func(*args, **(kwargs or {}))
        self._inside += 1
        try:
            if read or (self.count_transfers and self._inside == 1):
                self.hit(func)
            return func(*args, **(kwargs or {}))
        finally:
            self._inside -= 1

    def hit(self, func: object) -> None:
        self.total += 1
        outside = [
            f for f in traceback.extract_stack() if "site-packages" not in f.filename and f.name not in _INTERNAL
        ]
        frame = outside[-1]
        self.per_site[
            f"{frame.filename.split('/')[-1]}:{frame.lineno} {frame.name} {getattr(func, '__name__', func)}"
        ] += 1

    def mark(self) -> None:
        """Step boundary for per_step."""
        self.marks.append(self.total)

    @property
    def per_step(self) -> list[int]:
        edges = [*self.marks, self.total]
        return [b - a for a, b in itertools.pairwise(edges)]

    def report(self) -> str:
        sites = "\n".join(f"{count}x {site}" for site, count in self.per_site.most_common())
        return f"{sites}\n{self.transfers} transfer calls (.cpu/.numpy/.to(device=))" if self.transfers else sites


def count_blocking_calls(fn: Callable[[], Any]) -> int:
    """Exact count of blocking runtime calls, no method list involved. Needs a device. Number only,
    with_stack=True yields no python frames here. Measured: 3 .item() -> 3, .cpu() + .tolist() -> 2,
    (x * 2 + 1).sum() -> 0.
    """
    from torch.profiler import ProfilerActivity, profile

    def scan(run: Callable[[], Any]) -> int:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
        runtime = torch.autograd.DeviceType.CPU  # pyright: ignore[reportPrivateImportUsage]
        events = prof.events() or []
        return sum(1 for e in events if e.device_type == runtime and any(s in e.name for s in SYNC_RUNTIME))

    return max(scan(fn) - scan(lambda: None), 0)  # profiler teardown sync


def test_sync_counter_counts_exactly() -> None:
    """Counts the deferred read and nothing else, per step and in total."""
    counter = SyncCounter()
    with counter:
        tensor, doubled = torch.randn(4), torch.randn(4)
        for step in range(3):
            counter.mark()
            doubled = tensor * 2 + 1
            if step == 2:
                assert tensor.sum().item() < 1e30
    assert doubled.shape == torch.Size([4])
    assert counter.total == 1, counter.report()
    assert counter.per_step == [0, 0, 1], counter.report()

    other = SyncCounter()
    with other:
        float(other_tensor := torch.randn(4)[0])
        other_tensor.tolist()
    assert other.total == 2, other.report()


def test_no_host_sync_detects_the_usual_ones() -> None:
    """Fails if detection has silently become a no op. Runs without a device."""
    cases: dict[str, Callable[[Any], Any]] = {
        "timestep.item()": lambda t: t.sum().item(),
        "assert t == 0.5": lambda t: bool(t[0] == 0.5),
        "t[t > 0]": lambda t: t[t > 0],
        "t.tolist()": lambda t: t.tolist(),
        "t.cpu()": lambda t: t.cpu().sum(),
        "t.nonzero()": lambda t: t.nonzero(),
    }
    for detail, sync in cases.items():
        detected = False
        with no_host_sync():
            try:
                sync(torch.randn(4, device="cuda"))
            except (AssertionError, RuntimeError):
                detected = True
        assert detected, f"sync not detected: {detail}"

    with no_host_sync():
        async_tensor = torch.randn(4, device="cuda")
        assert (async_tensor * 2 + 1).sum().shape == torch.Size([])


def test_count_blocking_calls_agrees_with_the_hardware() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs an accelerator")
    tensor = torch.randn(8, device="cuda")

    def three_syncs() -> None:
        for _ in range(3):
            tensor.sum().item()

    def async_work() -> Any:  # noqa: ANN401
        return (tensor * 2 + 1).sum()

    assert count_blocking_calls(three_syncs) == 3
    assert count_blocking_calls(async_work) == 0
