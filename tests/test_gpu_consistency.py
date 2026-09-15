"""GPU checks (skipped without CUDA). Run: uv run pytest tests -m gpu -s"""

import pytest
import torch

from ecalc.capture import capture_layers
from ecalc.config import MeasureConfig
from ecalc.gpu import ClockSampler, max_sm_clock_mhz, nvml_handle_for_torch_device, snapshot
from ecalc.measure import make_monitor, measure_kernel
from ecalc.models import load_fused_model, make_input, set_precision
from ecalc.static_power import measure_active_idle_power

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def env():
    set_precision(False)
    handle = nvml_handle_for_torch_device(0)
    det = load_fused_model("yolo26n", "cuda")
    x = make_input("cuda")
    return dict(handle=handle, det=det, x=x, recs=capture_layers(det, x), monitor=make_monitor(0))


def test_captured_vs_synth_inputs_agree(env):
    cfg = MeasureConfig(measurement_duration_s=1.5, cooldown_s=0.5, repeats=1, num_calibration=200)
    for idx in (0, 8, 23):  # Conv, C3k2, Detect
        r = env["recs"][idx]
        a = measure_kernel(r, env["monitor"], env["handle"], cfg, "test", input_source="captured")
        b = measure_kernel(r, env["monitor"], env["handle"], cfg, "test", input_source="synth")
        assert abs(a["time_s"] - b["time_s"]) / a["time_s"] < 0.10, (r.short_desc, a["time_s"], b["time_s"])
        assert abs(a["energy_j"] - b["energy_j"]) / a["energy_j"] < 0.10, (r.short_desc, a["energy_j"], b["energy_j"])


def test_clock_sampler_sees_boost_under_load(env):
    with ClockSampler(env["handle"], 0.02) as s:
        with torch.inference_mode():
            for _ in range(200):
                env["det"](env["x"])
        torch.cuda.synchronize()
    summ = s.summary()
    assert summ["n"] > 5
    assert summ["sm_clock_mhz_mean"] > 0.8 * max_sm_clock_mhz(env["handle"])


def test_active_idle_is_p0_and_above_p8(env):
    row = measure_active_idle_power(env["det"], env["x"], env["monitor"], env["handle"], burst_s=1.0,
                                    idle_window_s=0.5, n=3, settle_s=0.5, require_persistence=False)
    assert row["n_samples"] >= 1 and row["pstate"] == 0
    assert 40.0 < row["power_w"] < 150.0, row["power_w"]
    assert snapshot(env["handle"]).sm_clock_mhz > 1000


def test_leaf_sum_vs_module_row(env):
    """Sum of isolated leaf/op energies of layer 6 (C3k2) vs the module-level row. First-time
    sanity: the ratio is printed and only loosely bounded (time composition is expected to differ)."""
    from ecalc.capture import capture_leaves

    cfg = MeasureConfig(measurement_duration_s=1.5, cooldown_s=0.5, repeats=1, num_calibration=200)
    leaves = capture_leaves(env["det"], env["x"], env["recs"])
    mod = measure_kernel(env["recs"][6], env["monitor"], env["handle"], cfg, "test")
    rows = {}
    for r in leaves:
        if r.parent_layer_idx == 6 and r.kernel_key not in rows:
            rows[r.kernel_key] = measure_kernel(r, env["monitor"], env["handle"], cfg, "test")
    e_leaf = sum(rows[r.kernel_key]["energy_j"] for r in leaves if r.parent_layer_idx == 6)
    t_leaf = sum(rows[r.kernel_key]["time_s"] for r in leaves if r.parent_layer_idx == 6)
    print(f"\nlayer 6 C3k2: E_leaf/E_mod = {e_leaf / mod['energy_j']:.3f}  t_leaf/t_mod = {t_leaf / mod['time_s']:.3f} "
          f"({len(rows)} unique leaf keys)")
    assert mod["energy_std_j"] is None  # single repeat -> NULL std
    assert 0.5 < e_leaf / mod["energy_j"] < 2.0
