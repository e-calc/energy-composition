#!/usr/bin/env python
"""Measure the full fused forward of each model with the same Zeus protocol (ground truth)."""

from __future__ import annotations

import argparse

import torch
from _common import Session, add_measure_args, add_model_args, parse_models

from ecalc import config as C, db
from ecalc.measure import gpu_kernel_time, make_monitor, measure_model_e2e, model_target
from ecalc.models import load_fused_model, make_input, variant_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_measure_args(ap)
    add_model_args(ap, "all")
    args = ap.parse_args()
    s = Session(args, "measure_model_e2e")
    conn = db.connect(args.db)
    monitor = make_monitor(args.device)
    x = make_input(s.device, shape=s.input_shape)
    loaded = []
    try:
        for name in parse_models(args.models):
            vname = variant_name(name, not args.one2many)
            det = load_fused_model(name, s.device, end2end=not args.one2many)
            s.log(f"== {vname} end-to-end (batch {s.batch}, {s.imgsz}px)")
            row = measure_model_e2e(det, x, monitor, s.handle, s.cfg, s.gpu_name, vname, s.batch, s.imgsz, log=s.log)
            if not s.persistence:
                row["notes"] = ";".join(filter(None, [row.get("notes"), "no_persistence_mode"]))
            db.upsert_model_run(conn, row)
            s.log(f"    -> {row['energy_j'] * 1e3:.3f} mJ  {row['time_s'] * 1e6:.1f} us  {row['power_w']:.1f} W  {row['notes'] or ''}")
            if args.profile_kernel_time:
                loaded.append((vname, det))
            else:
                del det
                torch.cuda.empty_cache()
        for vname, det in loaded:  # profiler last: it slows down all later CUDA launches in this process
            t = gpu_kernel_time(model_target(det, x))
            db.set_model_run_gpu_time(conn, vname, s.gpu_name, s.cfg.freq_label, s.cfg.dtype, s.batch, s.imgsz, t)
            s.log(f"    {vname} gpu_kernel_time {t * 1e6 if t else float('nan'):.1f} us")
    finally:
        s.close()


if __name__ == "__main__":
    main()
