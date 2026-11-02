#!/usr/bin/env python
"""Estimate model energy from the kernel DB and compare with the end-to-end measurement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecalc import config as C, db  # noqa: E402
from ecalc.estimate import (compare_levels, compare_levels_table, db_size_summary, db_size_table,  # noqa: E402
                            estimate_model, per_layer_table, plot_estimates, plot_level_comparison, summary_lines,
                            write_compare_report, write_report)
from ecalc.models import default_size, parse_model_list, variant_name  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="all", help="comma list, 'all' (YOLO26) or 'rfml'")
    ap.add_argument("--db", default=str(C.DB_PATH))
    ap.add_argument("--gpu", default=None, help="gpu_name in the DB (default: the only one present)")
    ap.add_argument("--freq", default="default")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="YOLO image size (default 640) or RFML IQ frame length (default 1024)")
    ap.add_argument("--one2many", action="store_true")
    ap.add_argument("--level", choices=list(C.LEVELS), default="module")
    ap.add_argument("--compare-levels", action="store_true",
                    help="also estimate at the other level and print the per-block leaf-vs-module and DB-size tables")
    ap.add_argument("--static-method", choices=["active_idle", "post_burst", "p8_idle", "user"],
                    default="active_idle")
    ap.add_argument("--static-w", type=float, default=None)
    ap.add_argument("--allow-miss", action="store_true", help="report even when some kernels are missing")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-layers", action="store_true", help="do not print per-layer tables")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    conn = db.connect(args.db)
    gpu = args.gpu
    if gpu is None:
        gpus = [r[0] for r in conn.execute("SELECT DISTINCT gpu_name FROM kernel_energy")]
        if len(gpus) != 1:
            sys.exit(f"--gpu required; DB has {gpus}")
        gpu = gpus[0]
    names = parse_model_list(args.models)
    if args.imgsz is None:
        sizes = {default_size(n) for n in names}
        if len(sizes) != 1:
            sys.exit(f"--imgsz required: models {names} have different default sizes {sorted(sizes)}")
        args.imgsz = sizes.pop()
    vnames = [variant_name(n, not args.one2many) for n in names]
    ests = []
    for v in vnames:
        e = estimate_model(conn, v, gpu, args.freq, args.dtype, args.batch, args.imgsz, args.static_method, args.static_w,
                           level=args.level)
        if e.misses and not args.allow_miss:
            sys.exit(f"{v}: {e.misses} kernels missing from the DB (use --allow-miss or run build_kernel_db.py "
                     f"--level {args.level})")
        print(f"[batch {args.batch}, size {args.imgsz}, level {args.level}]")
        print("\n".join(summary_lines(e)))
        if not args.no_layers:
            print(per_layer_table(e))
        print()
        ests.append(e)
    md, csvp = write_report(ests, conn, C.REPORTS_DIR, vnames, args.dtype, args.batch, args.imgsz, tag=args.tag)
    print(f"report: {md}\nlayers: {csvp}")
    if args.plot:
        p = plot_estimates(ests, C.PLOTS_DIR / f"{md.stem}.png")
        print(f"plot:   {p}")
    if args.compare_levels:
        other = "leaf" if args.level == "module" else "module"
        level_ests = {args.level: ests, other: []}
        for v in vnames:
            try:
                level_ests[other].append(estimate_model(conn, v, gpu, args.freq, args.dtype, args.batch, args.imgsz,
                                                        args.static_method, args.static_w, level=other))
            except LookupError as ex:
                print(f"WARNING: {ex}")
        level_ests = {lv: level_ests[lv] for lv in C.LEVELS if level_ests.get(lv)}
        cmps = []
        for v in vnames:
            try:
                c = compare_levels(conn, v, gpu, args.freq, args.dtype, args.batch, args.imgsz)
            except LookupError as ex:
                print(f"WARNING: {ex}")
                continue
            cmps.append(c)
            print(f"== per-block comparison {v} (leaf sum / module row)")
            print(compare_levels_table(c))
            print()
        size_rows = db_size_summary(conn, vnames, gpu, args.freq, args.dtype, args.batch, args.imgsz)
        print("== DB size per level (unique keys; 'new' = keys not seen in the previous models of the list)")
        print(db_size_table(size_rows))
        if cmps:
            md2, csv2 = write_compare_report(cmps, size_rows, level_ests, C.REPORTS_DIR,
                                             tag=args.tag or f"b{args.batch}")
            print(f"compare report: {md2}\nblocks: {csv2}")
            if args.plot:
                p2 = plot_level_comparison(cmps, size_rows, level_ests, C.PLOTS_DIR / f"{md2.stem}.png")
                print(f"compare plot:   {p2}")


if __name__ == "__main__":
    main()
