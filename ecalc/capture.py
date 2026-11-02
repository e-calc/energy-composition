"""Capture the kernels of a fused DetectionModel with their real inputs, at two granularities.

``capture_layers`` (level "module") reproduces ``BaseModel._predict_once`` routing
(``m.f == -1`` -> previous output, int -> saved output, list -> gathered outputs) so
each top-level layer can be re-executed in isolation with exactly the tensors it saw.

``capture_leaves`` (level "leaf") re-runs every top-level layer with forward hooks on
its leaf submodules (``LEAF_CLASSES``) and a ``TorchFunctionMode`` that records the
functional glue between them (``torch.cat``, ``chunk``, ``split``, residual ``add``).
Nested leaves (Convs inside Attention/Detect) are suppressed by a depth counter, so
every executed op is recorded exactly once, in execution order.

``capture_sequential`` (level "module") handles models whose forward runs a top-level
``layers`` ModuleList in order (``ecalc.rfml.SequentialRFModel``): each entry gets the
previous entry's output.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from typing import Any, Callable

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from .signature import input_signature, kernel_key, kernel_signature, op_signature, output_specs, short_desc


def _leaf_classes() -> tuple[type, ...]:
    from ultralytics.nn.modules.block import Attention
    from ultralytics.nn.modules.conv import Concat, Conv
    from ultralytics.nn.modules.head import Detect

    # DWConv subclasses Conv. Detect is kept whole (module-level row inside the leaf composition).
    return (Conv, Concat, Attention, Detect, nn.MaxPool2d, nn.Upsample)


LEAF_CLASSES: tuple[type, ...] = _leaf_classes()

# Functional ops that appear between leaves in YOLO26 blocks. Anything else seen outside a leaf
# is an error: the composition would be incomplete.
RECORDED_OPS = frozenset({"cat", "chunk", "split", "add"})


@dataclasses.dataclass
class LayerRecord:
    layer_idx: int
    layer_type: str
    from_idx: int | list[int]
    module: nn.Module | None
    inputs: torch.Tensor | list[torch.Tensor]
    input_sig: dict
    output_specs: list[dict]
    signature: dict
    kernel_key: str
    short_desc: str
    level: str = "module"
    parent_layer_idx: int | None = None
    path: str = ""
    seq: int | None = None
    op: dict | None = None  # {"name", "args", "kwargs", "seq_input"} for functional ops

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.module.parameters()) if self.module is not None else 0

    def run(self, inputs=None):
        """Execute this kernel once on ``inputs`` (default: the captured ones)."""
        inputs = self.inputs if inputs is None else inputs
        if self.op is None:
            return self.module(inputs)
        tensors = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
        fn = getattr(torch, self.op["name"])
        lead = (tensors,) if self.op["seq_input"] else tuple(tensors)
        return fn(*lead, *self.op["args"], **self.op["kwargs"])

    def target(self, inputs=None) -> Callable[[], None]:
        """Measurement closure (inference mode, no return value)."""
        inputs = self.inputs if inputs is None else inputs
        inference = torch.inference_mode()
        run = self.run

        def fn() -> None:
            with inference:
                run(inputs)

        return fn


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return [t.detach().clone() for t in x]


def _route(m: nn.Module, x, y: list):
    if m.f == -1:
        return x
    if isinstance(m.f, int):
        return y[m.f]
    return [x if j == -1 else y[j] for j in m.f]


@torch.inference_mode()
def capture_layers(det: nn.Module, x: torch.Tensor, record: bool = True) -> list[LayerRecord]:
    """Run ``det`` layer by layer. First pass (un-recorded) warms cudnn plans and the
    Detect anchor cache; the second pass records inputs per layer."""
    if record:
        capture_layers(det, x, record=False)
    records: list[LayerRecord] = []
    y: list = []
    cur = x
    for m in det.model:
        inp = _route(m, cur, y)
        out = m(inp)
        y.append(out if m.i in det.save else None)
        if record:
            sig = kernel_signature(m, inp)
            records.append(LayerRecord(
                layer_idx=int(m.i),
                layer_type=type(m).__name__,
                from_idx=m.f,
                module=m,
                inputs=_clone(inp),
                input_sig=input_signature(inp),
                output_specs=output_specs(out),
                signature=sig,
                kernel_key=kernel_key(sig),
                short_desc=short_desc(sig),
                level="module",
                parent_layer_idx=int(m.i),
                path="",
                seq=int(m.i),
            ))
        cur = out
    return records


# Tensor property access (``x.shape`` -> ``Tensor.shape.__get__``) also goes through
# __torch_function__; it launches nothing and is ignored.
_IGNORED_OPS = frozenset({"__get__"})


class _LeafState:
    def __init__(self) -> None:
        self.depth = 0
        self.muted = False  # True while our own hooks clone inputs / read output specs
        self.events: list[dict] = []
        self.current: dict | None = None


class _OpRecorder(TorchFunctionMode):
    """Records functional ops executed outside any leaf module."""

    def __init__(self, state: _LeafState) -> None:
        super().__init__()
        self.state = state
        self.unexpected: set[str] = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.state.depth == 0 and not self.state.muted:
            name = getattr(func, "__name__", str(func))
            if name in _IGNORED_OPS:
                pass
            elif name in RECORDED_OPS:
                seq_input = bool(args) and isinstance(args[0], (list, tuple))
                if seq_input:
                    tensors, rest = list(args[0]), list(args[1:])
                else:
                    n = 0
                    while n < len(args) and isinstance(args[n], torch.Tensor):
                        n += 1
                    tensors, rest = list(args[:n]), list(args[n:])
                if any(isinstance(a, torch.Tensor) for a in rest) or any(isinstance(v, torch.Tensor) for v in kwargs.values()):
                    raise RuntimeError(f"op {name}: tensor arguments in an unexpected position")
                self.state.events.append(dict(
                    kind="op", name=name, tensors=[t.detach().clone() for t in tensors], args=rest,
                    kwargs=dict(kwargs), seq_input=seq_input, out=output_specs(out),
                ))
            else:
                self.unexpected.add(name)
        return out


def _leaf_hooks(top: nn.Module, state: _LeafState) -> list:
    handles = []
    for name, sub in top.named_modules():
        if not isinstance(sub, LEAF_CLASSES):
            continue

        def pre(mod, args, _name=name):
            if state.depth == 0:
                state.muted = True
                try:
                    state.current = dict(kind="leaf", module=mod, path=_name, inputs=_clone(args[0]), out=None)
                finally:
                    state.muted = False
                state.events.append(state.current)
            state.depth += 1

        def post(mod, args, out):
            state.depth -= 1
            if state.depth == 0 and state.current is not None:
                state.muted = True
                try:
                    state.current["out"] = output_specs(out)
                finally:
                    state.muted = False
                state.current = None

        handles.append(sub.register_forward_pre_hook(pre))
        handles.append(sub.register_forward_hook(post))
    return handles


def capture_leaves(det: nn.Module, x: torch.Tensor, layers: list[LayerRecord] | None = None) -> list[LayerRecord]:
    """Leaf-level records of every top-level layer, in execution order."""
    layers = capture_layers(det, x) if layers is None else layers
    records: list[LayerRecord] = []
    seq = 0
    for rec in layers:
        state = _LeafState()
        handles = _leaf_hooks(rec.module, state)
        try:
            with torch.inference_mode(), _OpRecorder(state) as recorder:
                rec.module(rec.inputs)
        finally:
            for h in handles:
                h.remove()
        if recorder.unexpected:
            raise RuntimeError(f"layer {rec.layer_idx} ({rec.layer_type}): unrecorded functional ops outside leaves: "
                               f"{sorted(recorder.unexpected)}; extend RECORDED_OPS")
        if state.depth != 0:
            raise RuntimeError(f"layer {rec.layer_idx}: unbalanced leaf hooks (depth {state.depth})")
        for ev in state.events:
            if ev["kind"] == "leaf":
                mod, inp = ev["module"], ev["inputs"]
                sig = kernel_signature(mod, inp)
                records.append(LayerRecord(
                    layer_idx=seq, layer_type=type(mod).__name__, from_idx=rec.layer_idx, module=mod, inputs=inp,
                    input_sig=input_signature(inp), output_specs=ev["out"] or [], signature=sig,
                    kernel_key=kernel_key(sig), short_desc=short_desc(sig), level="leaf",
                    parent_layer_idx=rec.layer_idx, path=ev["path"], seq=seq,
                ))
            else:
                sig = op_signature(ev["name"], ev["tensors"], ev["args"], ev["kwargs"])
                op = dict(name=ev["name"], args=sig["op"]["args"], kwargs=sig["op"]["kwargs"], seq_input=ev["seq_input"])
                records.append(LayerRecord(
                    layer_idx=seq, layer_type=f"op:{ev['name']}", from_idx=rec.layer_idx, module=None,
                    inputs=ev["tensors"], input_sig=sig["input"], output_specs=ev["out"], signature=sig,
                    kernel_key=kernel_key(sig), short_desc=short_desc(sig), level="leaf",
                    parent_layer_idx=rec.layer_idx, path=f"op:{ev['name']}", seq=seq, op=op,
                ))
            seq += 1
    return records


@torch.inference_mode()
def capture_sequential(model: nn.Module, x: torch.Tensor, record: bool = True) -> list[LayerRecord]:
    """Run ``model.layers`` in order, recording each entry's input. First pass is un-recorded warm-up."""
    if record:
        capture_sequential(model, x, record=False)
    records: list[LayerRecord] = []
    cur = x
    for i, m in enumerate(model.layers):
        out = m(cur)
        if record:
            sig = kernel_signature(m, cur)
            records.append(LayerRecord(
                layer_idx=i, layer_type=type(m).__name__, from_idx=-1, module=m, inputs=_clone(cur),
                input_sig=input_signature(cur), output_specs=output_specs(out), signature=sig,
                kernel_key=kernel_key(sig), short_desc=short_desc(sig), level="module",
                parent_layer_idx=i, path="", seq=i,
            ))
        cur = out
    return records


def capture(det: nn.Module, x: torch.Tensor, level: str = "module") -> list[LayerRecord]:
    from .rfml import SequentialRFModel

    if isinstance(det, SequentialRFModel):
        if level != "module":
            raise ValueError(f"level {level!r} is not implemented for RFML models (module level only)")
        return capture_sequential(det, x)
    if level == "module":
        return capture_layers(det, x)
    if level == "leaf":
        return capture_leaves(det, x)
    raise ValueError(f"unknown level {level!r}")


def final_output(det: nn.Module, x: torch.Tensor) -> Any:
    with torch.inference_mode():
        return det(x)


def model_composition(records: list[LayerRecord], model_name: str, dtype: str, batch: int, imgsz: int,
                      ultralytics_version: str) -> list[dict]:
    """Rows for the ``model_kernels`` table (``layer_idx`` = graph index at module level, execution
    index at leaf level)."""
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return [
        dict(
            model_name=model_name, dtype=dtype, batch=batch, imgsz=imgsz, level=r.level, layer_idx=r.layer_idx,
            parent_layer_idx=r.parent_layer_idx, path=r.path,
            layer_type=r.layer_type, from_json=json.dumps(r.from_idx), kernel_key=r.kernel_key,
            short_desc=r.short_desc, input_spec_json=json.dumps(r.input_sig),
            output_spec_json=json.dumps(r.output_specs), n_params=r.n_params,
            ultralytics_version=ultralytics_version, captured_at=now,
        )
        for r in records
    ]


def unique_records(records: list[LayerRecord]) -> list[LayerRecord]:
    seen: set[str] = set()
    out = []
    for r in records:
        if r.kernel_key not in seen:
            seen.add(r.kernel_key)
            out.append(r)
    return out
