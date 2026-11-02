"""Exercise the E-CALC export API consumed by the component workflow."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cir_framework.contracts import Context, ModelSpec
from cir_framework.ecalc_io import KernelCatalog
from cir_framework.predictors import LookupPredictor
from cir_framework.workflow import evaluate, run_workflow

REPOSITORY = Path(__file__).resolve().parents[1]


class EcalcCompatibilityTests(unittest.TestCase):
    def test_upstream_export_can_feed_lookup_and_evaluation(self):
        # A separate interpreter keeps upstream imports out of the root package.
        exporter = '''
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from ecalc import db
directory = Path(sys.argv[2])
conn = db.connect(directory / "fixture.sqlite")
inputs = {"kind": "tensor", "shape": [1, 2], "dtype": "float32"}
signature = json.dumps({"sig_version": 1, "input": inputs, "module": {"cls": "Linear"}})
for gpu, energy in [("source", 1.0), ("target", 2.0)]:
    db.upsert_kernel(conn, dict(kernel_key="shared", gpu_name=gpu, freq_label="default",
        dtype="fp32", kernel_type="Linear", signature_json=signature,
        input_spec_json=json.dumps(inputs), energy_j=energy, time_s=.1))
rows = [dict(model_name="fixture", dtype="fp32", batch=1, imgsz=640,
             layer_idx=i, layer_type="Linear", kernel_key="shared",
             input_spec_json=json.dumps(inputs)) for i in range(2)]
db.replace_model_kernels(conn, "fixture", "fp32", 1, 640, rows, level="module")
db.upsert_model_run(conn, dict(model_name="fixture", gpu_name="target", freq_label="default",
    dtype="fp32", batch=1, imgsz=640, energy_j=5.0, time_s=.25))
db.export_csv(conn, directory / "export")
conn.close()
'''
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run([sys.executable, "-c", exporter, str(REPOSITORY), temporary],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            exported = Path(temporary) / "export"
            target = Context("target")
            result = run_workflow(exported, KernelCatalog(exported, Context("source")), target,
                                  ModelSpec("fixture"), LookupPredictor(KernelCatalog(exported, target)))
            result = evaluate(result, exported)
            self.assertEqual(result["coverage"]["instances"], 2)
            self.assertEqual(result["coverage"]["unique_keys"], 1)
            self.assertEqual(result["energy_j"], 4.0)
            self.assertAlmostEqual(result["evaluation"]["energy_error_pct"], -20)


if __name__ == "__main__":
    unittest.main()
