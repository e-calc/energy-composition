#!/usr/bin/env python
"""Measure static power definitions and store them in the DB.

active_idle (primary): Zeus window --settle seconds after a burst of full-model forwards (GPU at P0).
post_burst (reference): 0.2 s window immediately after the burst (includes the decaying transient).
p8_idle (optional): sleeping-GPU floor.
Also records a post-burst power decay trace to results/static_power/decay_<gpu>.csv.
"""

from __future__ import annotations

import argparse
import csv

from _common import Session, add_measure_args, add_model_args

from ecalc import config as C, db
from ecalc.gpu import max_sm_clock_mhz
from ecalc.measure import make_monitor
from ecalc.models import load_fused_model, make_input
from ecalc.static_power import measure_active_idle_power, measure_p8_idle_power, p0_plateau_s, record_power_decay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_measure_args(ap)
    add_model_args(ap, "yolo26n")
    ap.add_argument("--burst", type=float, default=2.0, help="seconds of model forwards before each idle window")
    ap.add_argument("--idle-window", type=float, default=1.0, help="seconds of the Zeus window")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds between burst end and window start")
    ap.add_argument("--n", type=int, default=20, help="number of burst/idle cycles")
    ap.add_argument("--no-post-burst", action="store_true", help="skip the immediate post-burst reference row")
    ap.add_argument("--auto-window", action="store_true", help="set idle window from the measured P0 plateau")
    ap.add_argument("--with-p8", action="store_true", help="also measure the P8 sleeping floor (~1 min)")
    ap.add_argument("--no-decay", action="store_true")
    ap.add_argument("--p8-only", action="store_true", help="only (re)measure the P8 sleeping floor")
    args = ap.parse_args()
    if args.p8_only:
        args.with_p8 = True
    p8_row = None
    if args.with_p8:
        # before Session(): torch.cuda.set_device() would create a CUDA context and keep the GPU at P0
        from ecalc.gpu import nvml_handle_for_visible_index

        h = nvml_handle_for_visible_index(args.device)
        print("p8_idle: waiting for the GPU to drop to P8 (no CUDA context yet) ...", flush=True)
        p8_row = measure_p8_idle_power(h, log=print)
    s = Session(args, "measure_static")
    conn = db.connect(args.db)
    try:
        if p8_row is not None:
            p8_row.update(gpu_name=s.gpu_name, freq_label=s.cfg.freq_label, model_name=None,
                          driver_version=s.meta["versions"]["driver_version"])
            db.upsert_static(conn, p8_row)
            s.log(f"p8_idle = {p8_row['power_w']:.2f} W (P{p8_row['pstate']}) {p8_row['notes'] or ''}")
            if args.p8_only:
                return
        model_name = args.models.split(",")[0]
        det = load_fused_model(model_name, s.device, end2end=not args.one2many)
        x = make_input(s.device, shape=s.input_shape)
        monitor = make_monitor(args.device)
        max_sm = max_sm_clock_mhz(s.handle)
        window = args.idle_window
        if not args.no_decay:
            s.log(f"recording post-burst power decay ({args.burst}s burst, 5s trace)")
            trace = record_power_decay(det, x, s.handle, burst_s=args.burst)
            p = C.STATIC_DIR / f"decay_{s.gpu_name}.csv"
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
                w.writeheader()
                w.writerows(trace)
            plateau = p0_plateau_s(trace, max_sm)
            first = [r for r in trace if r["t_s"] < 0.05]
            s.log(f"  decay trace -> {p}; P0 plateau after burst = {plateau:.3f}s; "
                  f"power in first 50 ms = {sum(r['power_w'] for r in first) / max(1, len(first)):.1f} W; "
                  f"power at {trace[-1]['t_s']:.1f}s = {trace[-1]['power_w']:.1f} W (P{trace[-1]['pstate']}, {trace[-1]['sm_clock_mhz']} MHz)")
            if args.auto_window:
                window = max(0.05, min(args.idle_window, 0.8 * (plateau - args.settle)))
                s.log(f"  auto idle window = {window:.3f}s (settle {args.settle}s)")
        common = dict(gpu_name=s.gpu_name, freq_label=s.cfg.freq_label, model_name=model_name,
                      driver_version=s.meta["versions"]["driver_version"])
        s.log(f"active_idle: {args.n} cycles of {args.burst}s burst, {args.settle}s settle, {window:.3f}s window on {model_name}")
        row = measure_active_idle_power(det, x, monitor, s.handle, burst_s=args.burst, idle_window_s=window,
                                        n=args.n, settle_s=args.settle, log=s.log,
                                        require_persistence=not args.allow_no_persistence)
        row.update(common)
        db.upsert_static(conn, row)
        s.log(f"active_idle = {row['power_w']:.2f} W ({row['n_samples']} accepted, {row['n_rejected']} rejected; {row['notes']})")
        if not args.no_post_burst:
            s.log(f"post_burst: {args.n} cycles of {args.burst}s burst + 0.2s window immediately after")
            row = measure_active_idle_power(det, x, monitor, s.handle, burst_s=args.burst, idle_window_s=0.2,
                                            n=args.n, settle_s=0.0, log=s.log, method="post_burst",
                                            require_persistence=not args.allow_no_persistence)
            row.update(common)
            db.upsert_static(conn, row)
            s.log(f"post_burst = {row['power_w']:.2f} W ({row['notes']})")
    finally:
        s.close()


if __name__ == "__main__":
    main()
