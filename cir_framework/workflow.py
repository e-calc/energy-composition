"""Compose predictor outputs and evaluate them after prediction.

This integration consumes E-CALC exports. The collaborator's C-IR translation
algorithm connects through the predictor contract and is not implemented here.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from .contracts import SCHEMA_VERSION, Context, ModelSpec, ComponentRequest, ComponentPrediction, Predictor
from .ecalc_io import KernelCatalog, _canonical, _number, _read

def _validate(prediction: ComponentPrediction, key: str) -> None:
    if not isinstance(prediction, ComponentPrediction) or prediction.kernel_key != key:
        raise ValueError(f"Predictor returned the wrong result type or key for {key}")
    _canonical(prediction.provenance_json)
    if prediction.status == "supported":
        _number(prediction.energy_j)
        if prediction.time_s is not None:
            _number(prediction.time_s, positive=True)
    elif prediction.status == "unsupported":
        if not prediction.reason or prediction.energy_j is not None or prediction.time_s is not None:
            raise ValueError("Unsupported results need a reason and must not contain estimates")
    else:
        raise ValueError(f"Unknown prediction status: {prediction.status!r}")


def run_workflow(source_data: Path, source: KernelCatalog, target: Context,
                 spec: ModelSpec, predictor: Predictor) -> dict:
    if spec.batch <= 0 or spec.size <= 0 or spec.level not in {"module", "leaf"}:
        raise ValueError("Invalid model batch, size or level")
    if not predictor.name or not predictor.version or not isinstance(predictor.uses_target_component_measurements, bool):
        raise ValueError("Predictor must declare name, version and target-measurement usage")
    rows, composition_provenance = _read(Path(source_data) / "model_kernels.csv", {
        "model_name", "dtype", "batch", "imgsz", "level", "layer_idx", "kernel_key", "input_spec_json",
    })
    rows = [r for r in rows if r["model_name"] == spec.name and r["dtype"] == source.context.dtype
            and int(r["batch"]) == spec.batch and int(r["imgsz"]) == spec.size and r["level"] == spec.level]
    if not rows:
        raise ValueError(f"No captured composition for {spec}")
    rows.sort(key=lambda r: int(r["layer_idx"]))
    indices = [int(r["layer_idx"]) for r in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Composition must have consecutive, unique execution indices starting at zero")
    predictions = {}
    layers = []
    for row in rows:
        key = row["kernel_key"]
        measurement = source.rows.get(key)
        if measurement is not None:
            signature = json.loads(measurement["signature_json"])
            if signature.get("input") != json.loads(row["input_spec_json"]):
                raise ValueError(f"Composition input differs from source signature for {key}")
        if key not in predictions:
            if measurement is None:
                prediction = ComponentPrediction(key, "unsupported", reason="Source component measurement missing")
            else:
                request = ComponentRequest(key, _canonical(measurement["signature_json"]), source.context,
                                           target, source.observation(key))
                prediction = predictor.predict(request)
            _validate(prediction, key)
            predictions[key] = prediction
        layers.append({"layer_idx": int(row["layer_idx"]), "layer_type": row.get("layer_type", ""),
                       "path": row.get("path", ""), "parent_layer_idx": row.get("parent_layer_idx", ""),
                       **asdict(predictions[key])})
    supported = [r for r in layers if r["status"] == "supported"]
    complete = len(supported) == len(layers)
    energy = math.fsum(r["energy_j"] for r in supported)
    all_times = complete and all(r["time_s"] is not None for r in layers)
    return {
        "schema_version": SCHEMA_VERSION, "model": asdict(spec), "source": asdict(source.context),
        "target": asdict(target),
        "predictor": {"name": predictor.name, "version": predictor.version,
                      "uses_target_component_measurements": predictor.uses_target_component_measurements},
        "inputs": [source.provenance, composition_provenance],
        "composition_method": "sum_of_isolated_component_energies",
        "coverage": {"instances": len(layers), "supported_instances": len(supported),
                     "unique_keys": len(predictions),
                     "supported_unique_keys": sum(p.status == "supported" for p in predictions.values()),
                     "complete": complete},
        "energy_j": energy if complete else None,
        "supported_component_energy_j": energy,
        "component_time_sum_s": math.fsum(r["time_s"] for r in layers) if all_times else None,
        "layers": layers,
        "limitations": ["Isolated component sums do not model fusion, overlap or cache interactions.",
                        "Component time sum is not a separate whole-model runtime predictor.",
                        "Energy boundaries and runtime conditions may differ between devices; inspect metadata."],
    }


def evaluate(result: dict, reference_data: Path) -> dict:
    """Attach ground truth after prediction; never provide it to the predictor."""
    rows, provenance = _read(Path(reference_data) / "model_runs.csv", {
        "model_name", "batch", "imgsz", "gpu_name", "freq_label", "dtype", "energy_j", "time_s",
    })
    spec = result["model"]
    matches = [r for r in rows if all(r[k] == v for k, v in result["target"].items())
               and r["model_name"] == spec["name"] and int(r["batch"]) == spec["batch"]
               and int(r["imgsz"]) == spec["size"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one matching whole-model reference, found {len(matches)}")
    row = matches[0]
    energy = _number(row["energy_j"], positive=True)
    elapsed = _number(row["time_s"], positive=True)
    def error(value, reference):
        return 100 * (value / reference - 1) if value is not None else None
    return {**result, "evaluation": {
        "reference": row, "input": provenance,
        "energy_error_pct": error(result["energy_j"], energy),
        "component_time_sum_error_pct": error(result["component_time_sum_s"], elapsed),
    }}


def write_reports(result: dict, output: Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "components.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["layers"][0]))
        writer.writeheader()
        writer.writerows(result["layers"])
    def value(number, suffix=""):
        return "unavailable" if number is None else f"{number:.6g}{suffix}"
    coverage = result["coverage"]
    lines = [f"# {result['model']['name']} component energy report", "",
             f"Predictor: {result['predictor']['name']} ({result['predictor']['version']}).",
             f"Source: {result['source']['gpu_name']} → target: {result['target']['gpu_name']}.",
             f"Batch: {result['model']['batch']}; size: {result['model']['size']}; level: {result['model']['level']}.", ""]
    if result["predictor"]["uses_target_component_measurements"]:
        lines += ["**Measured-target lookup baseline: this is not cross-device prediction accuracy.**", ""]
    lines += [f"Coverage: {coverage['supported_instances']}/{coverage['instances']} component occurrences; "
              f"{coverage['supported_unique_keys']}/{coverage['unique_keys']} unique keys.",
              f"Composed energy: {value(result['energy_j'], ' J')}.",
              f"Sum of component times: {value(result['component_time_sum_s'], ' s')}."]
    if not coverage["complete"]:
        lines += ["Incomplete coverage: no whole-model energy or accuracy is reported.",
                  f"Supported-component subtotal: {value(result['supported_component_energy_j'], ' J')}."]
    if "evaluation" in result:
        evaluation = result["evaluation"]
        lines += [f"Measured whole-model energy: {value(float(evaluation['reference']['energy_j']), ' J')}.",
                  f"Signed energy error: {value(evaluation['energy_error_pct'], '%')}.",
                  f"Signed component-time-sum error: {value(evaluation['component_time_sum_error_pct'], '%')}."]
    lines += ["", "| Index | Component | Status | Energy (J) | Time (s) | Reason |",
              "|---|---|---|---|---|---|"]
    for row in result["layers"]:
        cells = [row["layer_idx"], row["layer_type"], row["status"], value(row["energy_j"]),
                 value(row["time_s"]), row["reason"]]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines += ["", *[f"- {s}" for s in result["limitations"]], "",
              "Input hashes and prediction provenance are recorded in report.json."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
