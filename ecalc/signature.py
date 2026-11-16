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

from typing import Any

import torch
from torch import nn

from .calculus.kernel import (BACKWARD, FORWARD, KernelSpecification, TensorSpecification, canonical_json,
                              input_specifications, kernel_key, short_description)
from .config import SIG_VERSION

# Names used before the framework-free parts moved to ``ecalc.calculus.kernel``.
TensorSpec = TensorSpecification
short_desc = short_description

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


def input_signature(x: torch.Tensor | list | tuple) -> dict:
    if isinstance(x, torch.Tensor):
        return {"kind": "tensor", **TensorSpec.of(x).as_dict()}
    if isinstance(x, (list, tuple)):
        return {"kind": "list", "items": [TensorSpec.of(t).as_dict() for t in x]}
    raise TypeError(f"unsupported input type {type(x)}")


def input_specs(input_sig: dict) -> list[TensorSpecification]:
    return list(input_specifications(input_sig))


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


def backward_signature(signature: dict, input_gradient: bool = True) -> dict:
    """Signature of the backward pass of the kernel ``signature`` describes: gradients with respect
    to its parameters and, unless ``input_gradient`` is False, its inputs."""
    return KernelSpecification.from_signature(signature).with_direction(BACKWARD, input_gradient).signature


def signature_direction(signature: dict) -> str:
    return signature.get("direction", FORWARD)


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def synthesize_inputs(input_sig: dict, device: str | torch.device = "cuda", seed: int = 0):
    """Random inputs matching ``input_sig`` so a DB row can be re-measured from its signature alone."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def mk(spec: dict) -> torch.Tensor:
        t = torch.randn(tuple(spec["shape"]), generator=g, dtype=_DTYPES[spec["dtype"]]).to(device)
        if spec.get("layout") == "channels_last":
            t = t.contiguous(memory_format=torch.channels_last)
        return t

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
