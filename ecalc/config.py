"""Measurement configuration, project paths and constants."""

from __future__ import annotations

import dataclasses
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = DATA_DIR / "weights"
EXPORT_DIR = DATA_DIR / "export"
DB_PATH = DATA_DIR / "ecalc.sqlite"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
STATIC_DIR = RESULTS_DIR / "static_power"
LOCK_PATH = RESULTS_DIR / ".lock"

# Study scope: inference only, batch 1, fp32, 640x640.
BATCH = 1
IMGSZ = 640
INPUT_SHAPE = (BATCH, 3, IMGSZ, IMGSZ)
INPUT_SEED = 0

SIG_VERSION = 1

# Kernel granularities. "module" = one top-level Ultralytics layer; "leaf" = leaf module
# (Conv, MaxPool2d, Upsample, Concat, Attention; Detect kept whole) or a functional glue op.
LEVELS = ("module", "leaf")


@dataclasses.dataclass
class MeasureConfig:
    """Parameters of one zeus.profile.measure call plus our repeat policy.

    ``measurement_duration_s`` / ``cooldown_s`` map 1:1 to the kareus call
    ``zeus_measure(..., measurement_duration=5.0, cooldown_duration=5.0)``.
    ``repeats`` defaults to 1: across 192 three-repeat rows on the A40 the repeat-to-repeat
    energy CV had median 0.4% / max 2.1% (time 0.04% / 0.4%), far below the 1-5% composition
    errors studied, because one window already averages 10^4-10^5 iterations.
    """

    measurement_duration_s: float = 5.0
    cooldown_s: float = 3.0
    num_warmup: int = 50
    num_calibration: int = 1000
    repeats: int = 1
    clock_sample_period_s: float = 0.05
    dtype: str = "fp32"          # "fp32" (TF32 allowed, Ampere default) or "fp32_strict"
    freq_label: str = "default"  # "default" = unlocked clocks; str(mhz) when a lock succeeded
    profile_kernel_time: bool = False
    lock_mhz: int | None = None

    def zeus_kwargs(self) -> dict:
        return dict(
            measurement_duration=self.measurement_duration_s,
            cooldown_duration=self.cooldown_s,
            num_warmup_iterations=self.num_warmup,
            num_calibration_iterations=self.num_calibration,
        )


def ensure_dirs() -> None:
    for d in (DATA_DIR, WEIGHTS_DIR, EXPORT_DIR, REPORTS_DIR, PLOTS_DIR, LOGS_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
