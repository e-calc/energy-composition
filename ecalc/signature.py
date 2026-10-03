"""Structural kernel signatures.

A *kernel* is one top-level Ultralytics module (a layer of the YOLO yaml graph)
applied to inputs of a given shape/dtype. Its identity is a canonical JSON
signature made of (a) the module structure, walking ``named_modules()`` with a
per-class attribute whitelist plus parameter shapes, and (b) the input tensor
specs. Graph-position attributes (``i``, ``f``, ``type``, ``np``) and per-shape
caches (Detect anchors/strides/shape) are excluded so the same block at a
different position or in a different model size shares the key.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import torch
from torch import nn

from .config import SIG_VERSION

# Attributes that describe where a module sits in the graph or cache state, never its compute.
EXCLUDE_ATTRS = frozenset({
    "i", "f", "type", "np", "training", "shape", "anchors", "strides", "export", "format",
    "dynamic", "inplace", "name", "pt_path", "task", "save", "legacy", "return_indices",
    "transposed", "output_padding", "padding_mode", "ceil_mode", "scale", "no",
})

# Per-class whitelist. Unknown classes fall back to all scalar public attributes minus EXCLUDE_ATTRS.
ATTR_WHITELIST: dict[str, tuple[str, ...]] = {
    "Conv2d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups"),
    "ConvTranspose2d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups"),
    "Upsample": ("scale_factor", "mode"),
    "MaxPool2d": ("kernel_size", "stride", "padding", "dilation"),
    "Concat": ("d",),
    "Detect": ("nc", "nl", "reg_max", "max_det", "end2end", "agnostic_nms", "xyxy", "stride"),
    "Attention": ("num_heads", "head_dim", "key_dim"),
    "Bottleneck": ("add",),
    "PSABlock": ("add",),
    "SPPF": ("n", "add"),
    "C3k2": ("c",),
    "C2PSA": ("c",),
    "SiLU": (),
    # RFML (ecalc.rfml) building blocks; no YOLO26 module contains these classes.
    "Conv1d": ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups"),
    "MaxPool1d": ("kernel_size", "stride", "padding", "dilation"),
    "Linear": ("in_features", "out_features"),
    "LSTM": ("input_size", "hidden_size", "num_layers", "bias", "batch_first", "dropout", "bidirectional",
             "proj_size"),
    "ReLU": (),
    "SELU": (),
    "Softmax": ("dim",),
    "Flatten": ("start_dim", "end_dim"),
    "AlphaDropout": ("p",),
    "Identity": (),
    "Sequential": (),
    "ModuleList": (),
}

_SCALAR = (int, float, bool, str)


def _jsonable(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, torch.Size):
        return list(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(i) for i in v]
    if isinstance(v, _SCALAR) or v is None:
        return v
    return str(v)


def _attrs_of(m: nn.Module) -> dict[str, Any]:
    cls = type(m).__name__
    out: dict[str, Any] = {}
    if cls in ATTR_WHITELIST:
        for k in ATTR_WHITELIST[cls]:
            if hasattr(m, k):
                out[k] = _jsonable(getattr(m, k))
        return out
    for k, v in vars(m).items():
        if k.startswith("_") or k in EXCLUDE_ATTRS:
            continue
        if isinstance(v, _SCALAR) or (isinstance(v, (list, tuple)) and all(isinstance(i, _SCALAR) for i in v)):
            out[k] = _jsonable(v)
    return dict(sorted(out.items()))


def module_signature(m: nn.Module) -> dict:
    """Structure of ``m``: ordered submodule classes + whitelisted attrs + parameter/buffer shapes."""
    mods = []
    for name, sub in m.named_modules():
        mods.append({"n": name, "cls": type(sub).__name__, "a": _attrs_of(sub)})
    params = [{"n": n, "s": list(p.shape), "dt": str(p.dtype).replace("torch.", "")} for n, p in m.named_parameters()]
    bufs = [
        {"n": n, "s": list(b.shape), "dt": str(b.dtype).replace("torch.", "")}
        for n, b in m.named_buffers()
        if n.split(".")[-1] not in ("anchors", "strides")
    ]
    return {"cls": type(m).__name__, "modules": mods, "params": params, "buffers": bufs}


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str  # e.g. "float32"

    @classmethod
    def of(cls, t: torch.Tensor) -> "TensorSpec":
        return cls(tuple(t.shape), str(t.dtype).replace("torch.", ""))

    def as_dict(self) -> dict:
        return {"shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, d: dict) -> "TensorSpec":
        return cls(tuple(d["shape"]), d["dtype"])


def input_signature(x: torch.Tensor | list | tuple) -> dict:
    if isinstance(x, torch.Tensor):
        return {"kind": "tensor", **TensorSpec.of(x).as_dict()}
    if isinstance(x, (list, tuple)):
        return {"kind": "list", "items": [TensorSpec.of(t).as_dict() for t in x]}
    raise TypeError(f"unsupported input type {type(x)}")


def input_specs(input_sig: dict) -> list[TensorSpec]:
    if input_sig["kind"] == "tensor":
        return [TensorSpec.from_dict(input_sig)]
    return [TensorSpec.from_dict(d) for d in input_sig["items"]]


def kernel_signature(m: nn.Module, x: torch.Tensor | list | tuple) -> dict:
    return {"sig_version": SIG_VERSION, "module": module_signature(m), "input": input_signature(x)}


def op_signature(name: str, tensors: list[torch.Tensor], args: list, kwargs: dict) -> dict:
    """Signature of a functional op (``torch.cat``, ``chunk``, ``split``, ``add``) applied to
    ``tensors`` with the non-tensor positional ``args`` and ``kwargs`` (e.g. ``{"dim": 1}``)."""
    return {
        "sig_version": SIG_VERSION,
        "op": {"name": name, "args": _jsonable(list(args)), "kwargs": {k: _jsonable(v) for k, v in sorted(kwargs.items())}},
        "input": input_signature(list(tensors)),
    }


def canonical_json(sig: dict) -> str:
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def kernel_key(sig: dict) -> str:
    return hashlib.sha256(canonical_json(sig).encode()).hexdigest()[:16]


def _shape_str(spec: dict) -> str:
    return "x".join(str(i) for i in spec["shape"])


def short_desc(sig: dict) -> str:
    """Human-readable label, e.g. ``Conv[3->16,k3,s2]@1x3x640x640`` or ``op:cat[dim=1]@1x64x80x80+1x64x80x80``."""
    inp = sig["input"]
    shape = _shape_str(inp) if inp["kind"] == "tensor" else "+".join(_shape_str(s) for s in inp["items"])
    if "op" in sig:
        op = sig["op"]
        parts = [str(a) for a in op["args"]] + [f"{k}={v}" for k, v in op["kwargs"].items()]
        return f"op:{op['name']}[{','.join(parts)}]@{shape}"
    mod = sig["module"]
    cls = mod["cls"]
    mods = mod["modules"]
    convs = [m for m in mods if m["cls"] in ("Conv2d", "Conv1d")]
    linears = [m for m in mods if m["cls"] == "Linear"]
    lstms = [m for m in mods if m["cls"] == "LSTM"]
    nparam = sum(int(torch.Size(p["s"]).numel()) for p in mod["params"])
    top = mods[0]["a"] if mods else {}
    if cls == "Conv" and convs:
        a = convs[0]["a"]
        inner = f"{a['in_channels']}->{a['out_channels']},k{a['kernel_size'][0]},s{a['stride'][0]}"
        if a.get("groups", 1) != 1:
            inner += f",g{a['groups']}"
    elif cls == "Upsample":
        inner = f"x{top.get('scale_factor')},{top.get('mode')}"
    elif cls == "Concat":
        inner = f"d={top.get('d')}"
    elif cls == "Detect":
        inner = f"nc={top.get('nc')},{'e2e' if top.get('end2end') else 'o2m'}"
    elif convs:
        inner = f"{convs[0]['a']['in_channels']}->{convs[-1]['a']['out_channels']},p={nparam}"
    elif lstms:
        inner = f"{lstms[0]['a']['input_size']}->{lstms[-1]['a']['hidden_size']}"
    elif linears:
        inner = f"{linears[0]['a']['in_features']}->{linears[-1]['a']['out_features']}"
    else:
        inner = f"p={nparam}"
    return f"{cls}[{inner}]@{shape}"


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def synthesize_inputs(input_sig: dict, device: str | torch.device = "cuda", seed: int = 0):
    """Random inputs matching ``input_sig`` so a DB row can be re-measured from its signature alone."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def mk(spec: dict) -> torch.Tensor:
        return torch.randn(tuple(spec["shape"]), generator=g, dtype=_DTYPES[spec["dtype"]]).to(device)

    if input_sig["kind"] == "tensor":
        return mk(input_sig)
    return [mk(s) for s in input_sig["items"]]


def flatten_tensors(obj: Any) -> list[torch.Tensor]:
    """All tensors inside nested tuples/lists/dicts (Detect returns ``(y, {"one2one": {...}})``)."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in flatten_tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in flatten_tensors(o)]
    return []


def output_specs(obj: Any) -> list[dict]:
    return [TensorSpec.of(t).as_dict() for t in flatten_tensors(obj)]
