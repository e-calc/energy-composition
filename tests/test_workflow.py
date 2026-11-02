"""CPU-only workflow contract tests, runnable with stdlib unittest."""
import csv
import json
import math
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cir_framework.predict as predict_module
from cir_framework.predict import main
from cir_framework.contracts import ComponentPrediction, Context, ModelSpec
from cir_framework.ecalc_io import KernelCatalog
from cir_framework.predictors import LookupPredictor
from cir_framework.workflow import evaluate, run_workflow


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_dir = self.root / "source"
        self.target_dir = self.root / "target"
        self.source_dir.mkdir()
        self.target_dir.mkdir()
        self.source_context = Context("source")
        self.target_context = Context("target")
        self.spec = ModelSpec("tiny")
        self.input = {"kind": "tensor", "dtype": "float32", "shape": [1, 2]}
        self.signature = json.dumps({"input": self.input, "module": {"cls": "Linear"}, "sig_version": 1})
        self.measurements = [dict(kernel_key=k, gpu_name="source", freq_label="default", dtype="fp32",
                                  signature_json=self.signature, energy_j=e, time_s=t,
                                  torch_version="fixture", notes="fixture boundary")
                             for k, e, t in [("a", 1, .1), ("b", 2, .2)]]
        write_csv(self.source_dir / "kernel_energy.csv", self.measurements)
        self.target_rows = [{**r, "gpu_name": "target", "energy_j": r["energy_j"] * 2}
                            for r in self.measurements]
        write_csv(self.target_dir / "kernel_energy.csv", self.target_rows)
        self.composition = [dict(model_name="tiny", dtype="fp32", batch=1, imgsz=640, level="module",
                                 layer_idx=i, kernel_key=k, input_spec_json=json.dumps(self.input),
                                 layer_type="Linear") for i, k in enumerate(["a", "b", "a"])]
        write_csv(self.source_dir / "model_kernels.csv", self.composition)
        write_csv(self.target_dir / "model_runs.csv", [dict(
            model_name="tiny", dtype="fp32", batch=1, imgsz=640, gpu_name="target",
            freq_label="default", energy_j=10, time_s=.5)])

    def run_lookup(self):
        return run_workflow(self.source_dir, KernelCatalog(self.source_dir, self.source_context),
                            self.target_context, self.spec,
                            LookupPredictor(KernelCatalog(self.target_dir, self.target_context)))

    def test_repeated_occurrences_and_signed_errors(self):
        result = self.run_lookup()
        self.assertEqual(result["energy_j"], 8)
        self.assertEqual(result["coverage"]["instances"], 3)
        self.assertEqual(result["coverage"]["unique_keys"], 2)
        self.assertNotIn("evaluation", result)
        scored = evaluate(result, self.target_dir)
        self.assertAlmostEqual(scored["evaluation"]["energy_error_pct"], -20)
        self.assertAlmostEqual(scored["evaluation"]["component_time_sum_error_pct"], -20)
        self.assertNotIn("evaluation", result)

    def test_missing_target_suppresses_total_and_accuracy(self):
        write_csv(self.target_dir / "kernel_energy.csv", self.target_rows[:1])
        result = evaluate(self.run_lookup(), self.target_dir)
        self.assertEqual(result["supported_component_energy_j"], 4)
        self.assertIsNone(result["energy_j"])
        self.assertIsNone(result["component_time_sum_s"])
        self.assertIsNone(result["evaluation"]["energy_error_pct"])
        self.assertIn("missing", result["layers"][1]["reason"])

    def test_missing_source_is_explicit(self):
        write_csv(self.source_dir / "kernel_energy.csv", self.measurements[:1])
        result = self.run_lookup()
        self.assertIsNone(result["energy_j"])
        self.assertIn("Source", result["layers"][1]["reason"])

    def test_same_hash_different_signature_is_unsupported(self):
        self.target_rows[0]["signature_json"] = json.dumps({"input": self.input, "module": {"cls": "Other"}})
        write_csv(self.target_dir / "kernel_energy.csv", self.target_rows)
        result = self.run_lookup()
        self.assertEqual(result["coverage"]["supported_instances"], 1)
        self.assertIn("mismatch", result["layers"][0]["reason"])

    def test_duplicate_measurement_and_ambiguous_reference_rejected(self):
        write_csv(self.target_dir / "kernel_energy.csv", self.target_rows + self.target_rows[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_lookup()
        write_csv(self.target_dir / "kernel_energy.csv", self.target_rows)
        path = self.target_dir / "model_runs.csv"
        with path.open() as stream:
            row = next(csv.DictReader(stream))
        write_csv(path, [row, row])
        with self.assertRaisesRegex(ValueError, "found 2"):
            evaluate(self.run_lookup(), self.target_dir)

    def test_execution_indices_and_input_contract(self):
        self.composition[1]["layer_idx"] = 0
        write_csv(self.source_dir / "model_kernels.csv", self.composition)
        with self.assertRaisesRegex(ValueError, "execution indices"):
            self.run_lookup()
        self.composition[1]["layer_idx"] = 1
        self.composition[2]["input_spec_json"] = json.dumps({"shape": [99]})
        write_csv(self.source_dir / "model_kernels.csv", self.composition)
        with self.assertRaisesRegex(ValueError, "input differs"):
            self.run_lookup()

    def test_wrong_context_is_not_silently_substituted(self):
        with self.assertRaisesRegex(ValueError, "No kernel measurements"):
            KernelCatalog(self.target_dir, Context("target", "different-clock"))

    def plugin(self, response=None):
        class Plugin:
            name = "fixture-plugin"
            version = "test"
            uses_target_component_measurements = False
            def __init__(self):
                self.requests = []
            def predict(self, request):
                self.requests.append(request)
                if response is not None:
                    return response(request)
                return ComponentPrediction(request.kernel_key, "supported", request.observation.energy_j,
                                           provenance_json='{"calibration_id":"fixture"}')
        return Plugin()

    def test_plugin_receives_only_source_and_context_and_runs_once_per_key(self):
        predictor = self.plugin()
        result = run_workflow(self.source_dir, KernelCatalog(self.source_dir, self.source_context),
                              self.target_context, self.spec, predictor)
        self.assertEqual([r.kernel_key for r in predictor.requests], ["a", "b"])
        self.assertEqual(result["energy_j"], 4)
        self.assertIsNone(result["component_time_sum_s"])
        request = predictor.requests[0]
        self.assertEqual(request.schema_version, 1)
        self.assertEqual(request.observation.energy_j, 1)
        self.assertFalse(hasattr(request, "reference"))
        self.assertEqual(json.loads(request.observation.metadata_json)["torch_version"], "fixture")

    def test_invalid_plugin_outputs_rejected(self):
        responses = [lambda r: ComponentPrediction(r.kernel_key, "supported", math.nan),
                     lambda r: ComponentPrediction(r.kernel_key, "supported", -1),
                     lambda r: ComponentPrediction("wrong-key", "supported", 1),
                     lambda r: ComponentPrediction(r.kernel_key, "unsupported", 1, reason="missing"),
                     lambda r: ComponentPrediction(r.kernel_key, "unsupported"),
                     lambda r: ComponentPrediction(r.kernel_key, "supported", 1, 0)]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                run_workflow(self.source_dir, KernelCatalog(self.source_dir, self.source_context),
                             self.target_context, self.spec, self.plugin(response))

    def arguments(self):
        return ["--source-data", str(self.source_dir), "--source-gpu", "source", "--target-gpu", "target",
                "--model", "tiny", "--output", str(self.root / "output")]

    def test_cli_outputs_and_incomplete_exit(self):
        args = self.arguments() + ["--target-data", str(self.target_dir), "--reference-data", str(self.target_dir)]
        self.assertEqual(main(args), 0)
        report = self.root / "output/report.md"
        self.assertIn("not cross-device prediction accuracy", report.read_text())
        self.assertIn("sha256", (self.root / "output/report.json").read_text())
        write_csv(self.target_dir / "kernel_energy.csv", self.target_rows[:1])
        self.assertEqual(main(args), 1)
        self.assertIn("Incomplete coverage", report.read_text())

    def test_external_factory_and_evaluation_happen_in_order(self):
        predictor = self.plugin()
        class Module:
            @staticmethod
            def build(config):
                self.assertEqual(config, {"calibration_id": "test"})
                return predictor
        config = self.root / "config.json"
        config.write_text('{"calibration_id":"test"}')
        original = evaluate
        def check_evaluation(result, directory):
            self.assertEqual(len(predictor.requests), 2)
            return original(result, directory)
        with patch.object(predict_module, "evaluate", side_effect=check_evaluation), \
                patch.object(predict_module.importlib, "import_module", return_value=Module) as load:
            code = main(self.arguments() + ["--predictor", "translator:build", "--predictor-config", str(config),
                                            "--reference-data", str(self.target_dir)])
        self.assertEqual(code, 0)
        load.assert_called_once_with("translator")
        result = json.loads((self.root / "output/report.json").read_text())
        self.assertFalse(result["predictor"]["uses_target_component_measurements"])

    def test_plugin_cannot_accidentally_receive_lookup_directory(self):
        with self.assertRaises(SystemExit) as exit_info:
            main(self.arguments() + ["--predictor", "translator:build", "--target-data", str(self.target_dir)])
        self.assertEqual(exit_info.exception.code, 2)

    def test_real_external_module_import(self):
        module_path = self.root / "fixture_energy_plugin.py"
        module_path.write_text('''from cir_framework.contracts import ComponentPrediction
class Predictor:
    name = "import-test"
    version = "fixture-v1"
    uses_target_component_measurements = False
    def predict(self, request):
        return ComponentPrediction(request.kernel_key, "unsupported", reason="No calibration")
def build(config):
    return Predictor()
''')
        self.addCleanup(sys.modules.pop, "fixture_energy_plugin", None)
        with patch.object(sys, "path", [str(self.root), *sys.path]):
            self.assertEqual(main(self.arguments() + ["--predictor", "fixture_energy_plugin:build"]), 1)
        report = json.loads((self.root / "output/report.json").read_text())
        self.assertEqual(report["predictor"]["name"], "import-test")
        self.assertTrue(all(r["reason"] == "No calibration" for r in report["layers"]))


if __name__ == "__main__":
    unittest.main()
