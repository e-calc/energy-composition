#!/usr/bin/env python
"""Capture every kernel of the given models (at --level module or leaf) and measure each unique
kernel in isolation.

Resumable: keys already present for (gpu, freq, dtype) are skipped unless --force. Keys are
level-agnostic, so leaf rows that coincide with a top-level layer reuse the module-level row.
Each kernel is committed as soon as it is measured, so Ctrl-C keeps the rows done so far.
"""

from __future__ import annotations

import argparse
import time

import torch
from _common import Session, add_level_arg, add_measure_args, add_model_args, parse_models

from ecalc import config as C, db
from ecalc.capture import capture, model_composition
from ecalc.measure import gpu_kernel_time, kernel_target, make_monitor, measure_kernel
from ecalc.models import load_model, model_input, variant_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_measure_args(ap)
    add_model_args(ap, "all")
    add_level_arg(ap)
    ap.add_argument("--input-source", choices=["captured", "synth"], default="captured")
    ap.add_argument("--force", action="store_true", help="re-measure keys already in the DB")
    ap.add_argument("--layers", default=None,
                    help="comma list of top-level layer indices to measure (leaf level: their leaves) (debug)")
    args = ap.parse_args()
    s = Session(args, "build_kernel_db")
    conn = db.connect(args.db)
    monitor = make_monitor(args.device)
    done_this_run: set[str] = set()
    only = {int(i) for i in args.layers.split(",")} if args.layers else None
    t_start = time.time()
    n_meas = 0
    to_profile: list = []  # (LayerRecord) measured or already present; profiled after ALL Zeus measurements
    try:
        for name in parse_models(args.models):
            vname = variant_name(name, not args.one2many)
            size = s.size_for(name)
            det = load_model(name, s.device, end2end=not args.one2many)
            x = model_input(name, s.batch, size, s.device)
            recs = capture(det, x, level=args.level)
            db.replace_model_kernels(conn, vname, s.cfg.dtype, s.batch, size,
                                     model_composition(recs, vname, s.cfg.dtype, s.batch, size,
                                                       s.meta["versions"]["ultralytics_version"]),
                                     level=args.level)
            todo = [r for r in recs if (only is None or r.parent_layer_idx in only)]
            n_uniq = len({r.kernel_key for r in recs})
            s.log(f"== {vname} (batch {s.batch}, size {size}, level {args.level}): {len(recs)} kernels captured, "
                  f"{n_uniq} unique")
            for r in todo:
                if r.kernel_key in done_this_run:
                    continue
                if not args.force and db.has_kernel(conn, r.kernel_key, s.gpu_name, s.cfg.freq_label, s.cfg.dtype):
                    s.log(f"  skip L{r.parent_layer_idx:02d}/{r.layer_idx:03d} {r.kernel_key} {r.short_desc} (in DB)")
                    done_this_run.add(r.kernel_key)
                    to_profile.append(r)
                    continue
                s.log(f"  measure L{r.parent_layer_idx:02d}/{r.layer_idx:03d} {r.kernel_key} {r.short_desc}")
                row = measure_kernel(r, monitor, s.handle, s.cfg, s.gpu_name, input_source=args.input_source,
                                     device=s.device, log=s.log)
                if not s.persistence:
                    row["notes"] = ";".join(filter(None, [row.get("notes"), "no_persistence_mode"]))
                db.upsert_kernel(conn, row)
                done_this_run.add(r.kernel_key)
                to_profile.append(r)
                n_meas += 1
                std = f"  std {row['energy_std_j'] * 1e3:.3f} mJ" if row['energy_std_j'] is not None else ""
                s.log(f"    -> {row['energy_j'] * 1e3:.3f} mJ  {row['time_s'] * 1e6:.1f} us  {row['power_w']:.1f} W"
                      f"{std}  {row['notes'] or ''}")
            if not args.profile_kernel_time:
                del det, recs
                torch.cuda.empty_cache()
        if args.profile_kernel_time:
            # torch.profiler slows down all later CUDA launches in this process, so it runs last.
            s.log(f"== profiling GPU kernel time of {len(to_profile)} kernels (after all Zeus measurements)")
            for r in to_profile:
                t = gpu_kernel_time(kernel_target(r))
                db.set_kernel_gpu_time(conn, r.kernel_key, s.gpu_name, s.cfg.freq_label, s.cfg.dtype, t)
                s.log(f"  L{r.parent_layer_idx:02d}/{r.layer_idx:03d} {r.kernel_key} gpu_kernel_time "
                      f"{t * 1e6 if t else float('nan'):.1f} us")
    except KeyboardInterrupt:
        s.log("interrupted; committed rows are kept")
    finally:
        s.log(f"measured {n_meas} kernels in {(time.time() - t_start) / 60:.1f} min")
        s.close()


if __name__ == "__main__":
    main()
