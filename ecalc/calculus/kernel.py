"""Kernel identity: the parameters ``theta`` of an energy signature.

A kernel is one callable (a module or a functional operation) applied to inputs of a given
shape, dtype and layout, in a given direction. Its identity is a canonical JSON document, the
*structural signature*, and its key is a hash of that document. This module holds the parts of
the identity that need no deep-learning framework; ``ecalc.signature`` builds the structural
signature from live ``torch`` modules.

Two fields are written only when they differ from their default, so keys computed before the
fields existed stay valid:

- ``direction`` (top level): ``"backward"``; absent means forward.
- ``layout`` (per input tensor): ``"channels_last"``; absent means the default memory format.

``layout`` is the tensor's memory format. Other stride patterns (a transposed or sliced view) are
not told apart from the default format: a kernel is keyed by what it computes on which shapes,
and the default-format key is what every earlier measurement was stored under.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from functools import cached_property
from typing import Any, Mapping

SIGNATURE_VERSION = 1

FORWARD = "forward"
BACKWARD = "backward"
DIRECTIONS = (FORWARD, BACKWARD)

DEFAULT_PRECISION = "fp32"
"""32-bit floats, with TF32 allowed for convolutions and matrix products."""
PRECISIONS = (DEFAULT_PRECISION,)
"""The supported precisions. ``fp32`` is the only one so far; another precision would be written
into the kernel's signature and so get its own keys."""

CONTIGUOUS = "contiguous"
LAYOUTS = (CONTIGUOUS, "channels_last")

DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
}


@dataclasses.dataclass(frozen=True)
class TensorSpecification:
    """Shape, dtype and memory layout of one input tensor."""

    shape: tuple[int, ...]
    dtype: str = "float32"
    layout: str = CONTIGUOUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(i) for i in self.shape))
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r}; known: {LAYOUTS}")

    @classmethod
    def of(cls, tensor: Any) -> "TensorSpecification":
        """Specification of a live tensor (anything with ``shape``, ``dtype`` and ``is_contiguous``)."""
        layout = CONTIGUOUS
        if tensor.dim() == 4 and not tensor.is_contiguous():
            import torch  # only reached with a torch tensor in hand

            if tensor.is_contiguous(memory_format=torch.channels_last):
                layout = "channels_last"
        return cls(tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""), layout)

    def as_dict(self) -> dict:
        out: dict = {"shape": list(self.shape), "dtype": self.dtype}
        if self.layout != CONTIGUOUS:
            out["layout"] = self.layout
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> "TensorSpecification":
        return cls(tuple(data["shape"]), data["dtype"], data.get("layout", CONTIGUOUS))

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_count(self) -> int:
        return self.element_count * DTYPE_BYTES.get(self.dtype, 4)


def canonical_json(signature: Mapping) -> str:
    """The one serialized form of a structural signature that keys are computed from."""
    return json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)


def kernel_key(signature: Mapping) -> str:
    return hashlib.sha256(canonical_json(signature).encode()).hexdigest()[:16]


def input_specifications(input_signature: Mapping) -> tuple[TensorSpecification, ...]:
    """Tensor specifications of the ``input`` part of a structural signature."""
    if input_signature["kind"] == "tensor":
        return (TensorSpecification.from_dict(input_signature),)
    return tuple(TensorSpecification.from_dict(item) for item in input_signature["items"])


def _shape_text(specification: Mapping) -> str:
    return "x".join(str(i) for i in specification["shape"])


def short_description(signature: Mapping) -> str:
    """Human-readable label, e.g. ``Conv[3->16,k3,s2]@1x3x640x640`` or
    ``op:cat[dim=1]@1x64x80x80+1x64x80x80``; backward kernels end in ``<backward>``, kernels at a
    non-default precision in ``<that precision>``."""
    suffix = ""
    if signature.get("direction", FORWARD) != FORWARD:
        suffix = f" <{signature['direction']}" + ("" if signature.get("input_gradient", True) else ", no input gradient") + ">"
    if signature.get("precision", DEFAULT_PRECISION) != DEFAULT_PRECISION:
        suffix += f" <{signature['precision']}>"
    if "unresolved" in signature:
        return (signature["unresolved"].get("description") or signature["unresolved"]["key"]) + suffix
    inputs = signature["input"]
    shape = _shape_text(inputs) if inputs["kind"] == "tensor" else "+".join(_shape_text(s) for s in inputs["items"])
    if "op" in signature:
        op = signature["op"]
        parts = [str(a) for a in op["args"]] + [f"{k}={v}" for k, v in op["kwargs"].items()]
        return f"op:{op['name']}[{','.join(parts)}]@{shape}{suffix}"
    module = signature["module"]
    try:
        inner = _module_summary(module)
    except (KeyError, IndexError, TypeError):  # a label must never fail on an unusual module
        inner = f"p={sum(math.prod(p['s']) for p in module.get('params', []))}"
    return f"{module['cls']}[{inner}]@{shape}{suffix}"


def _module_summary(module: Mapping) -> str:
    cls = module["cls"]
    modules = module["modules"]
    convolutions = [m for m in modules if m["cls"] in ("Conv2d", "Conv1d")]
    linears = [m for m in modules if m["cls"] == "Linear"]
    recurrent = [m for m in modules if m["cls"] == "LSTM"]
    parameter_count = sum(math.prod(p["s"]) for p in module["params"])
    top = modules[0]["a"] if modules else {}
    if cls == "Conv" and convolutions:
        a = convolutions[0]["a"]
        inner = f"{a['in_channels']}->{a['out_channels']},k{a['kernel_size'][0]},s{a['stride'][0]}"
        if a.get("groups", 1) != 1:
            inner += f",g{a['groups']}"
        return inner
    if cls == "Upsample":
        return f"x{top.get('scale_factor')},{top.get('mode')}"
    if cls == "Concat":
        return f"d={top.get('d')}"
    if cls == "Detect":
        return f"nc={top.get('nc')},{'e2e' if top.get('end2end') else 'o2m'}"
    if convolutions:
        return f"{convolutions[0]['a']['in_channels']}->{convolutions[-1]['a']['out_channels']},p={parameter_count}"
    if recurrent:
        return f"{recurrent[0]['a']['input_size']}->{recurrent[-1]['a']['hidden_size']}"
    if linears:
        return f"{linears[0]['a']['in_features']}->{linears[-1]['a']['out_features']}"
    return f"p={parameter_count}"


@dataclasses.dataclass(frozen=True)
class KernelSpecification:
    """``theta``: what a kernel is, independent of how much it costs.

    Wraps the canonical structural signature. Two specifications are equal exactly when their
    canonical JSON is equal, and ``key`` is the hash the databases are indexed by.
    """

    canonical: str

    @classmethod
    def from_signature(cls, signature: Mapping) -> "KernelSpecification":
        return cls(canonical_json(signature))

    @classmethod
    def from_key(cls, key: str, description: str = "") -> "KernelSpecification":
        """A kernel known only by its key (its structure was not recorded). It can be looked up
        in a database but cannot be estimated by a formula."""
        return cls(canonical_json({"unresolved": {"key": key, "description": description}}))

    @cached_property
    def signature(self) -> dict:
        return json.loads(self.canonical)

    @property
    def is_resolved(self) -> bool:
        return "unresolved" not in self.signature

    @cached_property
    def key(self) -> str:
        if set(self.signature) == {"unresolved"}:
            return self.signature["unresolved"]["key"]
        return hashlib.sha256(self.canonical.encode()).hexdigest()[:16]

    @property
    def kind(self) -> str:
        """``"operation"`` for a functional op (``torch.cat``, ...), ``"module"`` otherwise."""
        return "operation" if "op" in self.signature else "module"

    @property
    def name(self) -> str:
        """Module class name, or ``op:<name>`` for a functional operation."""
        if not self.is_resolved:
            return "unresolved"
        if "op" in self.signature:
            return f"op:{self.signature['op']['name']}"
        return self.signature["module"]["cls"]

    @property
    def direction(self) -> str:
        return self.signature.get("direction", FORWARD)

    @property
    def precision(self) -> str:
        """Numeric precision the kernel runs at; part of what the kernel is, like its direction."""
        return self.signature.get("precision", DEFAULT_PRECISION)

    @property
    def input_gradient(self) -> bool:
        """Backward only: whether the gradient with respect to the inputs is computed."""
        return bool(self.signature.get("input_gradient", True))

    @cached_property
    def inputs(self) -> tuple[TensorSpecification, ...]:
        return input_specifications(self.signature["input"]) if self.is_resolved else ()

    @property
    def parameter_shapes(self) -> tuple[tuple[int, ...], ...]:
        if self.kind != "module" or not self.is_resolved:
            return ()
        return tuple(tuple(p["s"]) for p in self.signature["module"]["params"])

    @property
    def has_parameters(self) -> bool:
        return bool(self.parameter_shapes)

    @property
    def description(self) -> str:
        return short_description(self.signature)

    def with_direction(self, direction: str, input_gradient: bool = True) -> "KernelSpecification":
        """The same kernel in another direction."""
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}; known: {DIRECTIONS}")
        signature = {k: v for k, v in self.signature.items() if k not in ("direction", "input_gradient")}
        if direction != FORWARD:
            signature["direction"] = direction
            if not input_gradient:
                signature["input_gradient"] = False
        return KernelSpecification.from_signature(signature)

    def with_precision(self, precision: str) -> "KernelSpecification":
        """The same kernel at another numeric precision. The default precision is not written, so
        the key of a default-precision kernel is the key of its structure alone."""
        if precision not in PRECISIONS:
            raise ValueError(f"unknown precision {precision!r}; known: {PRECISIONS}")
        signature = {k: v for k, v in self.signature.items() if k != "precision"}
        if precision != DEFAULT_PRECISION:
            signature["precision"] = precision
        return KernelSpecification.from_signature(signature)

    def __str__(self) -> str:
        return f"{self.key} {self.description}"
