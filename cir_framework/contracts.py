"""Versioned contracts between the framework and component predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Context:
    gpu_name: str
    freq_label: str = "default"
    dtype: str = "fp32"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    batch: int = 1
    size: int = 640
    level: str = "module"


@dataclass(frozen=True)
class Observation:
    energy_j: float
    time_s: float
    # Original metadata includes runtime versions, clocks and protocol fields.
    metadata_json: str


@dataclass(frozen=True)
class ComponentRequest:
    kernel_key: str
    signature_json: str
    source: Context
    target: Context
    observation: Observation
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class ComponentPrediction:
    kernel_key: str
    status: str  # "supported" or "unsupported"
    energy_j: float | None = None
    time_s: float | None = None
    reason: str = ""
    provenance_json: str = "{}"


class Predictor(Protocol):
    """A deterministic per-signature predictor, scoped to one calibrated backend.

    The workflow calls predict once per unique key and reuses its result for
    every occurrence. Graph/context-dependent algorithms need a future contract.
    """

    name: str
    version: str
    uses_target_component_measurements: bool

    def predict(self, request: ComponentRequest) -> ComponentPrediction: ...
