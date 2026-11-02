"""Built-in predictors. Actual C-IR translation is supplied by an external plugin."""

from __future__ import annotations

import json

from .contracts import ComponentRequest, ComponentPrediction
from .ecalc_io import KernelCatalog, _canonical

class LookupPredictor:
    """Measured-target lookup baseline, not cross-device inference."""

    name = "target-lookup-baseline"
    version = "1"
    uses_target_component_measurements = True

    def __init__(self, catalog: KernelCatalog):
        self.catalog = catalog

    def predict(self, request: ComponentRequest) -> ComponentPrediction:
        if request.target != self.catalog.context:
            raise ValueError("Lookup context differs from requested target")
        row = self.catalog.rows.get(request.kernel_key)
        if row is None:
            return ComponentPrediction(request.kernel_key, "unsupported", reason="Target component measurement missing")
        if _canonical(row["signature_json"]) != _canonical(request.signature_json):
            return ComponentPrediction(request.kernel_key, "unsupported", reason="Source/target signature mismatch")
        return ComponentPrediction(
            request.kernel_key, "supported", float(row["energy_j"]), float(row["time_s"]),
            provenance_json=json.dumps({"kind": "target_measurement", "input": self.catalog.provenance,
                                        "measurement": row}, sort_keys=True),
        )
