"""Adapters for upstream E-CALC CSV exports; no upstream code is modified."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from .contracts import Context, Observation

def _number(value, *, positive=False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"Expected finite {'positive' if positive else 'nonnegative'} value, got {value!r}")
    return number


def _canonical(raw: str) -> str:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read(path: Path, required: set[str]) -> tuple[list[dict], dict]:
    # Hash and parse the same bytes, even if the export changes during a run.
    import io

    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if not required <= set(headers) or len(headers) != len(set(headers)):
        raise ValueError(f"{path}: missing required or duplicate column names")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


class KernelCatalog:
    """One explicit device/frequency/dtype view of a kernel_energy export."""

    def __init__(self, directory: Path, context: Context):
        self.context = context
        rows, self.provenance = _read(Path(directory) / "kernel_energy.csv", {
            "kernel_key", "gpu_name", "freq_label", "dtype", "signature_json", "energy_j", "time_s",
        })
        self.rows = {}
        for row in rows:
            if not all(row[key] == value for key, value in asdict(context).items()):
                continue
            key = row["kernel_key"]
            if key in self.rows:
                raise ValueError(f"Duplicate measurement for {key} in {directory}")
            _number(row["energy_j"])
            _number(row["time_s"], positive=True)
            _canonical(row["signature_json"])
            self.rows[key] = row
        if not self.rows:
            raise ValueError(f"No kernel measurements for {context} in {directory}")

    def observation(self, key: str) -> Observation:
        row = self.rows[key]
        return Observation(float(row["energy_j"]), float(row["time_s"]), json.dumps(row, sort_keys=True))
