"""Run a selectable component predictor: python -m cir_framework.predict --help."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from .contracts import Context, ModelSpec
from .ecalc_io import KernelCatalog
from .predictors import LookupPredictor
from .workflow import evaluate, run_workflow, write_reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True, help="Directory of source ecalc CSV exports")
    parser.add_argument("--source-gpu", required=True)
    parser.add_argument("--source-freq", default="default")
    parser.add_argument("--target-gpu", required=True)
    parser.add_argument("--target-freq", default="default")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--level", choices=["module", "leaf"], default="module")
    parser.add_argument("--predictor", default="lookup", help="lookup or importable module:factory(config)")
    parser.add_argument("--predictor-config", type=Path, help="JSON object passed only to the external predictor factory")
    parser.add_argument("--target-data", type=Path, help="Target kernel export; only for the measured lookup baseline")
    parser.add_argument("--reference-data", type=Path, help="Optional target model_runs export, read after prediction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source_context = Context(args.source_gpu, args.source_freq, args.dtype)
        target_context = Context(args.target_gpu, args.target_freq, args.dtype)
        source = KernelCatalog(args.source_data, source_context)
        config_provenance = None
        if args.predictor == "lookup":
            if args.target_data is None or args.predictor_config is not None:
                raise ValueError("lookup requires --target-data and does not accept --predictor-config")
            predictor = LookupPredictor(KernelCatalog(args.target_data, target_context))
        else:
            if args.target_data is not None:
                raise ValueError("External predictors do not accept --target-data; pass calibration via --predictor-config")
            module, separator, factory = args.predictor.partition(":")
            if not separator or not module or not factory:
                raise ValueError("External predictor must have the form module:factory")
            config_bytes = args.predictor_config.read_bytes() if args.predictor_config else b"{}"
            config = json.loads(config_bytes)
            if args.predictor_config:
                config_provenance = {"path": str(args.predictor_config.resolve()),
                                     "sha256": hashlib.sha256(config_bytes).hexdigest()}
            if not isinstance(config, dict):
                raise ValueError("Predictor config must be a JSON object")
            predictor = getattr(importlib.import_module(module), factory)(config)
        spec = ModelSpec(args.model, args.batch, args.size, args.level)
        result = run_workflow(args.source_data, source, target_context, spec, predictor)
        result["predictor"]["entrypoint"] = args.predictor
        if config_provenance:
            result["inputs"].append(config_provenance)
        if args.reference_data:
            result = evaluate(result, args.reference_data)
        write_reports(result, args.output)
    except (ValueError, TypeError, OSError, ImportError, AttributeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Report: {args.output / 'report.md'}")
    if not result["coverage"]["complete"]:
        print("Incomplete component coverage; no whole-model estimate. See report for missing components.")
        return 1
    print(f"Composed energy: {result['energy_j']:.6g} J")
    if result["predictor"]["uses_target_component_measurements"]:
        print("Measured-target lookup baseline; not cross-device prediction accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
