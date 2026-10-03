"""YOLO26 registry, weight download, fused model loading and input creation.

We never call ``YOLO(...).predict``: it adds letterbox preprocessing, dtype
selection and CPU post-processing. The measured object is the fused
``DetectionModel`` forward, exactly what ``AutoBackend`` runs inside predict
when ``nms=False`` (NMS-free one-to-one head, the YOLO26 default design).

RFML models (``ecalc.rfml``) are dispatched by name through ``load_model`` / ``model_input``;
for them the ``imgsz`` column and ``--imgsz`` option hold the IQ frame length.
"""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import torch

from .config import IMGSZ, INPUT_SEED, INPUT_SHAPE, WEIGHTS_DIR
from .rfml import ALL_RFML, FRAME_LEN, is_rfml, load_rfml_model, make_rfml_input

YOLO26 = {
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
}
ALL_MODELS = list(YOLO26)
MODEL_ALIASES = {"all": ALL_MODELS, "rfml": ALL_RFML}


def parse_model_list(s: str) -> list[str]:
    """``"all"`` (YOLO26 sizes), ``"rfml"`` (RFML models) or a comma list."""
    return list(MODEL_ALIASES[s]) if s in MODEL_ALIASES else [m.strip() for m in s.split(",") if m.strip()]


def base_name(model_name: str) -> str:
    """``"yolo26n-nms" -> "yolo26n"``."""
    return model_name.removesuffix("-nms")


def variant_name(name: str, end2end: bool) -> str:
    """DB model name: plain for the NMS-free head, ``-nms`` suffix for the one-to-many head."""
    return name if (end2end or is_rfml(name)) else f"{name}-nms"


def resolve_weights(name: str) -> Path:
    name = base_name(name)
    if name not in YOLO26:
        raise ValueError(f"unknown model {name!r}; known: {ALL_MODELS}")
    path = WEIGHTS_DIR / YOLO26[name]
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(path))
    return path


def load_fused_model(name: str, device: str | torch.device = "cuda", end2end: bool = True):
    """Fused, eval-mode ``DetectionModel`` with all grads disabled."""
    from ultralytics.nn.tasks import load_checkpoint

    path = resolve_weights(name)
    model, _ckpt = load_checkpoint(str(path), device=device, fuse=False)
    if hasattr(model, "end2end"):
        model.end2end = end2end
    model = model.fuse(verbose=False).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    assert model.is_fused(), "model did not fuse"
    assert not model.training
    return model


def load_model(name: str, device: str | torch.device = "cuda", end2end: bool = True):
    """Fused YOLO26 ``DetectionModel`` or eval-mode RFML model, by name."""
    if is_rfml(name):
        return load_rfml_model(name, device)
    return load_fused_model(name, device, end2end=end2end)


def default_size(name: str) -> int:
    """Default ``imgsz``: image side for YOLO26, IQ frame length for RFML."""
    return FRAME_LEN if is_rfml(name) else IMGSZ


def model_input(name: str, batch: int, size: int | None = None, device: str | torch.device = "cuda",
                seed: int = INPUT_SEED) -> torch.Tensor:
    size = default_size(name) if size is None else size
    if is_rfml(name):
        return make_rfml_input(name, batch, size, device, seed)
    return make_input(device, seed, shape=(batch, 3, size, size))


def make_input(device: str | torch.device = "cuda", seed: int = INPUT_SEED, shape=INPUT_SHAPE) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


def set_precision(strict_fp32: bool = False) -> str:
    """Return the dtype label. Ampere runs fp32 conv/matmul in TF32 by default."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = not strict_fp32
    torch.backends.cuda.matmul.allow_tf32 = not strict_fp32
    return "fp32_strict" if strict_fp32 else "fp32"


def precision_flags() -> dict:
    return {
        "tf32_conv": int(torch.backends.cudnn.allow_tf32),
        "tf32_matmul": int(torch.backends.cuda.matmul.allow_tf32),
    }


def versions() -> dict:
    out = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ultralytics_version": md.version("ultralytics"),
        "zeus_version": md.version("zeus"),
    }
    try:
        from .gpu import driver_version

        out["driver_version"] = driver_version()
    except Exception:
        out["driver_version"] = None
    return out
