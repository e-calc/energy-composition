#!/usr/bin/env python
"""Experiment: run two models at the same time on one GPU and compare with the overlap rule.

Each model runs in its own thread on its own CUDA stream. For every pair the script measures

  alone      each model by itself for a window: time and energy per call
  together   both at once for a window: calls of each, window time, window energy

and compares the measured window energy with two predictions built from the *alone* numbers and
the calls that actually happened in the window:

  no overlap     sum of n_i * e_i                      (each call costs what it costs alone)
  overlap rule   s * T_window + sum of n_i * d_i       (static power once, dynamic energies add)

where d_i = e_i - s * t_i. It also reports how much each model slowed down while sharing the
GPU, which the calculus does not model: it assumes a kernel's time and dynamic energy are the
same alone and overlapped.

    python scripts/measure_overlap.py --pairs yolo26n:1+rf_vgg:32,yolo26m:8+yolo26n:8

This is an experiment, not part of the measurement protocol: results are printed and written to
results/overlap/, and nothing is stored in the signature store. Needs a CUDA GPU.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from ecalc import config as C, db  # noqa: E402
from ecalc.gpu import gpu_name, nvml_handle_for_torch_device, persistence_mode_on  # noqa: E402
from ecalc.measure import make_monitor  # noqa: E402
from ecalc.models import default_size, load_model, model_input, set_precision  # noqa: E402


class Runner:
    """One model called in a loop on its own stream; ``calls`` counts finished calls."""

    def __init__(self, name: str, batch_size: int, device: str) -> None:
        self.label = f"{name}:{batch_size}"
        self.model = load_model(name, device)
        self.x = model_input(name, batch_size, default_size(name), device)
        self.stream = torch.cuda.Stream(device=device)
        self.calls = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            while not self._stop.is_set():
                self.model(self.x)
                self.stream.synchronize()  # one call at a time: the count is finished work
                self.calls += 1

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def window(runners: list[Runner], monitor, device_index: int, duration: float, warmup: float, cooldown: float) -> dict:
    """Run ``runners`` together; measure a ``duration`` window after ``warmup`` seconds."""
    time.sleep(cooldown)
    for r in runners:
        r.start()
    time.sleep(warmup)
    before = [r.calls for r in runners]
    monitor.begin_window("overlap", sync_execution=False)
    time.sleep(duration)
    measurement = monitor.end_window("overlap", sync_execution=False)
    after = [r.calls for r in runners]
    for r in runners:
        r.stop()
    torch.cuda.synchronize()
    return dict(time_s=measurement.time, energy_j=measurement.gpu_energy[device_index],
                calls=[b - a for a, b in zip(before, after)])


def median_window(runners, monitor, device_index, args) -> dict:
    trials = [window(runners, monitor, device_index, args.duration, args.warmup, args.cooldown)
              for _ in range(args.repeats)]
    return sorted(trials, key=lambda t: t["energy_j"] / t["time_s"])[len(trials) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default="yolo26n:1+rf_vgg:32,yolo26m:8+yolo26n:8",
                        help="comma list of model:batch+model:batch")
    parser.add_argument("--store", default=str(C.DB_PATH), help="measurement store holding the static power")
    parser.add_argument("--static-power", default="active_idle", help="stored method or a number of watts")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0, help="seconds of each measurement window")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    set_precision(False)
    handle = nvml_handle_for_torch_device(args.device)
    gpu = gpu_name(handle)
    try:
        static_watts = float(args.static_power)
    except ValueError:
        row = db.get_static(db.connect(args.store), gpu, "default", args.static_power)
        if row is None:
            sys.exit(f"no static power {args.static_power!r} for {gpu} in {args.store}; pass --static-power <watts>")
        static_watts = float(row["power_w"])
    monitor = make_monitor(args.device)
    print(f"{gpu}, static power {static_watts:.1f} W ({args.static_power}), persistence mode "
          f"{'on' if persistence_mode_on(handle) else 'OFF'}; windows of {args.duration:.0f} s, median of {args.repeats}")

    results = []
    for pair in args.pairs.split(","):
        runners = [Runner(spec.split(":")[0], int(spec.split(":")[1]), device) for spec in pair.split("+")]
        alone = []
        for r in runners:
            w = median_window([r], monitor, 0, args)
            alone.append(dict(label=r.label, time_s=w["time_s"] / w["calls"][0], energy_j=w["energy_j"] / w["calls"][0]))
        together = median_window(runners, monitor, 0, args)
        calls = together["calls"]
        dynamic = [a["energy_j"] - static_watts * a["time_s"] for a in alone]
        no_overlap = sum(n * a["energy_j"] for n, a in zip(calls, alone))
        overlap_rule = static_watts * together["time_s"] + sum(n * d for n, d in zip(calls, dynamic))
        busy = sum(n * a["time_s"] for n, a in zip(calls, alone))  # time the same calls would take back to back
        slowdown = [together["time_s"] / n / a["time_s"] for n, a in zip(calls, alone)]
        result = dict(pair=pair, alone=alone, together=together, static_power_watts=static_watts,
                      predicted_no_overlap_j=no_overlap, predicted_overlap_rule_j=overlap_rule,
                      back_to_back_time_s=busy, slowdown=slowdown)
        results.append(result)
        print(f"\n{pair}")
        for a, n, d, k in zip(alone, calls, dynamic, slowdown):
            print(f"  {a['label']:14} alone {a['time_s'] * 1e3:8.3f} ms  {a['energy_j'] * 1e3:9.2f} mJ "
                  f"({a['energy_j'] / a['time_s']:5.1f} W, d = {d * 1e3:8.2f} mJ)   together: {n} calls, "
                  f"{k:.2f}x slower per call")
        measured = together["energy_j"]
        print(f"  window {together['time_s']:.2f} s; the same calls back to back would take {busy:.2f} s")
        print(f"  measured            {measured:9.1f} J")
        print(f"  no overlap          {no_overlap:9.1f} J  ({100 * (no_overlap / measured - 1):+.1f}%)")
        print(f"  overlap rule        {overlap_rule:9.1f} J  ({100 * (overlap_rule / measured - 1):+.1f}%)")
        del runners
        torch.cuda.empty_cache()

    out_dir = C.RESULTS_DIR / "overlap"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"overlap_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(dict(gpu=gpu, torch=torch.__version__, arguments=vars(args), results=results), indent=2))
    errors = [abs(r["predicted_overlap_rule_j"] / r["together"]["energy_j"] - 1) for r in results]
    print(f"\noverlap rule: mean absolute error {100 * statistics.mean(errors):.1f}% over {len(results)} pairs -> {path}")


if __name__ == "__main__":
    main()
