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

``backward_records`` turns a module-level forward capture into the kernels of the backward
pass: every layer in the backward direction, plus one ``add`` per extra consumer of a layer's
output (autograd sums the gradients arriving at a tensor that feeds several layers).
``BackwardGraph`` holds one retained autograd graph and replays its backward pass.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import functools
import json
from typing import Any, Callable

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from .calculus.kernel import BACKWARD, FORWARD
from .signature import (backward_signature, flatten_tensors, input_signature, kernel_key, kernel_signature,
                        op_signature, output_specs, short_desc)


@functools.cache
def leaf_classes() -> tuple[type, ...]:
    """Leaf module classes of the YOLO26 graph. Imported on first use, so RFML-only work does not
    need Ultralytics."""
    from ultralytics.nn.modules.block import Attention
    from ultralytics.nn.modules.conv import Concat, Conv
    from ultralytics.nn.modules.head import Detect

    # DWConv subclasses Conv. Detect is kept whole (module-level row inside the leaf composition).
    return (Conv, Concat, Attention, Detect, nn.MaxPool2d, nn.Upsample)


def __getattr__(name: str):
    if name == "LEAF_CLASSES":  # kept for callers of the former module constant
        return leaf_classes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    direction: str = FORWARD
    input_gradient: bool = True  # backward only: is the gradient with respect to the inputs computed

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
        """Forward measurement closure (inference mode, no return value)."""
        if self.direction != FORWARD:
            raise ValueError(f"{self.short_desc}: target() is the forward closure; use backward_graph()")
        inputs = self.inputs if inputs is None else inputs
        inference = torch.inference_mode()
        run = self.run

        def fn() -> None:
            with inference:
                run(inputs)

        return fn

    def backward_graph(self, inputs=None) -> "BackwardGraph":
        """Retained autograd graph of this kernel; its ``step`` is the backward measurement closure."""
        inputs = self.inputs if inputs is None else inputs
        modules = list(self.module.modules()) if self.module is not None else []
        parameters = list(self.module.parameters()) if self.module is not None else []
        return BackwardGraph(self.run, inputs, parameters, modules, input_gradient=self.input_gradient)


def differentiable_outputs(output: Any) -> list[torch.Tensor]:
    """The tensors a backward pass starts from. For an output that carries raw head tensors in a
    dict next to a decoded result (Ultralytics ``Detect`` in eval mode), only the raw tensors."""
    def in_dicts(obj: Any) -> list[torch.Tensor]:
        if isinstance(obj, dict):
            return flatten_tensors(obj)
        if isinstance(obj, (list, tuple)):
            return [t for o in obj for t in in_dicts(o)]
        return []

    tensors = in_dicts(output) or flatten_tensors(output)
    return [t for t in tensors if t.is_floating_point() and t.requires_grad]


def _replace_inference_tensors(modules: list) -> None:
    """Swap tensors a module cached during an inference-mode run (``Detect`` anchors and strides)
    for ordinary copies with the same values, so autograd can track computations that use them."""
    for m in modules:
        for name, value in list(vars(m).items()):
            if isinstance(value, torch.Tensor) and value.is_inference():
                setattr(m, name, value.clone())
        for name, value in list(m._buffers.items()):
            if value is not None and value.is_inference():
                m._buffers[name] = value.clone()


class BackwardGraph:
    """One retained autograd graph; ``step()`` replays its backward pass.

    The forward runs once, with gradients enabled, on fresh copies of the inputs. ``step`` is
    ``torch.autograd.grad(outputs, inputs + parameters, grad_outputs, retain_graph=True)``: the
    gradient kernels only, with no loss, no ``.grad`` accumulation and no optimizer. ``close``
    restores the parameter flags and module modes it changed.

    Recurrent layers are switched to train mode while the graph is alive, because cuDNN refuses
    a recurrent backward in eval mode. Their dropout is left as configured.
    """

    def __init__(self, run: Callable, inputs, parameters: list, modules: list, input_gradient: bool = True,
                 seed: int = 0) -> None:
        _replace_inference_tensors(modules)
        self._restore_eval = [m for m in modules if isinstance(m, nn.RNNBase) and not m.training]
        for m in self._restore_eval:
            m.train()
        self._restore_frozen = [p for p in parameters if not p.requires_grad]
        for p in self._restore_frozen:
            p.requires_grad_(True)
        single = isinstance(inputs, torch.Tensor)
        # Captured inputs are inference tensors; clone them outside inference mode so autograd can use them.
        leaves = [t.detach().clone().requires_grad_(input_gradient) for t in ([inputs] if single else inputs)]
        try:
            with torch.enable_grad():
                output = run(leaves[0] if single else leaves)
            self.outputs = differentiable_outputs(output)
            if not self.outputs:
                raise RuntimeError("the kernel has no differentiable output")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.grad_outputs = [torch.randn(tuple(o.shape), generator=generator, dtype=o.dtype).to(o.device)
                                 for o in self.outputs]
            self.wrt = [t for t in leaves if t.requires_grad] + list(parameters)
            if not self.wrt:
                raise RuntimeError("nothing to differentiate with respect to: no input gradient and no parameters")
        except Exception:
            self.close()
            raise

    def step(self) -> None:
        torch.autograd.grad(self.outputs, self.wrt, self.grad_outputs, retain_graph=True, allow_unused=True)

    def close(self) -> None:
        self.outputs = self.grad_outputs = self.wrt = None
        for p in self._restore_frozen:
            p.requires_grad_(False)
        for m in self._restore_eval:
            m.eval()
        self._restore_frozen, self._restore_eval = [], []

    def __enter__(self) -> "BackwardGraph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def model_backward_graph(model: nn.Module, x: torch.Tensor) -> BackwardGraph:
    """Backward pass of the whole model: gradients of its raw outputs with respect to every
    parameter (the model input needs no gradient)."""
    return BackwardGraph(model, x, list(model.parameters()), list(model.modules()), input_gradient=False)


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
            kept = _clone(inp)  # the tensors the kernel is measured on; their layout is what gets recorded
            sig = kernel_signature(m, kept)
            records.append(LayerRecord(
                layer_idx=int(m.i),
                layer_type=type(m).__name__,
                from_idx=m.f,
                module=m,
                inputs=kept,
                input_sig=input_signature(kept),
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
        if not isinstance(sub, leaf_classes()):
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
            kept = _clone(cur)
            sig = kernel_signature(m, kept)
            records.append(LayerRecord(
                layer_idx=i, layer_type=type(m).__name__, from_idx=-1, module=m, inputs=kept,
                input_sig=input_signature(kept), output_specs=output_specs(out), signature=sig,
                kernel_key=kernel_key(sig), short_desc=short_desc(sig), level="module",
                parent_layer_idx=i, path="", seq=i,
            ))
        cur = out
    return records


def capture(det: nn.Module, x: torch.Tensor, level: str = "module", direction: str = FORWARD) -> list[LayerRecord]:
    """Kernels of ``det`` on input ``x``: the forward pass, or (module level only) the backward pass."""
    from .rfml import SequentialRFModel

    if direction not in (FORWARD, BACKWARD):
        raise ValueError(f"unknown direction {direction!r}")
    if direction == BACKWARD and level != "module":
        raise ValueError("the backward pass is captured at module level only")
    if isinstance(det, SequentialRFModel):
        if level != "module":
            raise ValueError(f"level {level!r} is not implemented for RFML models (module level only)")
        records = capture_sequential(det, x)
    elif level == "module":
        records = capture_layers(det, x)
    elif level == "leaf":
        records = capture_leaves(det, x)
    else:
        raise ValueError(f"unknown level {level!r}")
    return backward_records(records) if direction == BACKWARD else records


def consumer_counts(records: list[LayerRecord]) -> dict[int, int]:
    """How many layers consume each layer's output, from the ``from`` indices of a module-level
    capture. The model input (source index -1 of the first layer) is not counted."""
    position = {r.layer_idx: n for n, r in enumerate(records)}
    counts: dict[int, int] = {}
    for n, r in enumerate(records):
        sources = r.from_idx if isinstance(r.from_idx, (list, tuple)) else [r.from_idx]
        for source in sources:
            if source < 0:
                if n + source < 0:
                    continue  # the model input
                source = records[n + source].layer_idx
            if source not in position:
                raise ValueError(f"layer {r.layer_idx} reads from unknown layer {source}")
            counts[source] = counts.get(source, 0) + 1
    return counts


def backward_records(records: list[LayerRecord]) -> list[LayerRecord]:
    """Kernels of the backward pass of a module-level forward capture.

    Each layer becomes the same kernel in the backward direction (its own key). The first layer
    reads the model input, which needs no gradient, and that is part of its key. A layer whose
    output feeds ``n`` layers adds ``n - 1`` gradient accumulations, each priced as an ``add``
    of two tensors of that output's shape. Order: last layer first.
    """
    if any(r.level != "module" for r in records):
        raise ValueError("the backward pass is captured at module level only")
    out: list[LayerRecord] = []
    for n, r in reversed(list(enumerate(records))):
        input_gradient = n > 0
        sig = backward_signature(r.signature, input_gradient)
        out.append(dataclasses.replace(r, signature=sig, kernel_key=kernel_key(sig), short_desc=short_desc(sig),
                                       direction=BACKWARD, input_gradient=input_gradient))
    by_index = {r.layer_idx: r for r in records}
    next_index = max(by_index) + 1
    for source, count in sorted(consumer_counts(records).items()):
        if count < 2:
            continue
        producer = by_index[source]
        if len(producer.output_specs) != 1:
            raise ValueError(f"layer {source} feeds {count} layers but has {len(producer.output_specs)} outputs")
        spec = producer.output_specs[0]
        device = flatten_tensors(producer.inputs)[0].device
        generator = torch.Generator(device="cpu").manual_seed(source)
        dtype = getattr(torch, spec["dtype"])
        tensors = [torch.randn(tuple(spec["shape"]), generator=generator, dtype=dtype).to(device) for _ in range(2)]
        sig = op_signature("add", tensors, [], {})
        for _ in range(count - 1):
            out.append(LayerRecord(
                layer_idx=next_index, layer_type="op:add", from_idx=source, module=None, inputs=tensors,
                input_sig=sig["input"], output_specs=[dict(spec)], signature=sig, kernel_key=kernel_key(sig),
                short_desc=short_desc(sig), level="module", parent_layer_idx=source, path="gradient_accumulation",
                seq=next_index, op=dict(name="add", args=[], kwargs={}, seq_input=False),
            ))
            next_index += 1
    return out


def final_output(det: nn.Module, x: torch.Tensor) -> Any:
    with torch.inference_mode():
        return det(x)


def model_composition(records: list[LayerRecord], model_name: str, dtype: str, batch: int, imgsz: int,
                      ultralytics_version: str) -> list[dict]:
    """Rows for the ``model_kernels`` table (``layer_idx`` = graph index at module level, execution
    index at leaf level)."""
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    direction = BACKWARD if any(r.direction == BACKWARD for r in records) else FORWARD
    return [
        dict(
            model_name=model_name, dtype=dtype, batch=batch, imgsz=imgsz, level=r.level, direction=direction,
            layer_idx=r.layer_idx, parent_layer_idx=r.parent_layer_idx, path=r.path,
            signature_json=json.dumps(r.signature, sort_keys=True),
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
