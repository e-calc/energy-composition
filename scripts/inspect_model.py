#!/usr/bin/env python
"""Capture kernels of YOLO26 or RFML models (no energy measurement) at module or leaf level and print
the kernel table, per-class instance/unique counts and key reuse across models."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from ecalc import config as C  # noqa: E402
from ecalc.capture import capture, final_output, unique_records  # noqa: E402
from ecalc.models import load_model, model_input, parse_model_list, set_precision, variant_name  # noqa: E402
from ecalc.signature import flatten_tensors  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="yolo26n", help="comma list, 'all' (YOLO26) or 'rfml'")
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO image size or RFML frame length (family default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--one2many", action="store_true", help="use the one-to-many head (needs external NMS)")
    ap.add_argument("--level", choices=list(C.LEVELS), default="module")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    set_precision(False)
    names = parse_model_list(args.models)
    key_models: dict[str, set[str]] = defaultdict(set)
    key_desc: dict[str, str] = {}
    for name in names:
        det = load_model(name, args.device, end2end=not args.one2many)
        x = model_input(name, args.batch, args.imgsz, args.device)
        vname = variant_name(name, not args.one2many)
        recs = capture(det, x, level=args.level)
        out = final_output(det, x)
        uniq = unique_records(recs)
        n_ops = sum(1 for r in recs if r.op is not None)
        print(f"\n== {vname} [{args.level}]: {len(recs)} kernels ({len(recs) - n_ops} modules + {n_ops} ops), "
              f"{len(uniq)} unique keys, final output shapes {[tuple(t.shape) for t in flatten_tensors(out)][:1]}")
        by_cls: dict[str, list[str]] = defaultdict(list)
        for r in recs:
            by_cls[r.layer_type].append(r.kernel_key)
        print("   per class (instances/unique): " + ", ".join(f"{c} {len(v)}/{len(set(v))}" for c, v in sorted(by_cls.items())))
        if not args.quiet:
            if args.level == "leaf":
                print(f"{'seq':>3} {'par':>3} {'path':14} {'key':16} {'params':>9}  desc")
                for r in recs:
                    print(f"{r.layer_idx:>3} {r.parent_layer_idx:>3} {r.path[:14]:14} {r.kernel_key} {r.n_params:>9}  {r.short_desc}")
            else:
                print(f"{'idx':>3} {'from':>12} {'key':16} {'params':>9}  desc")
                for r in recs:
                    print(f"{r.layer_idx:>3} {str(r.from_idx):>12} {r.kernel_key} {r.n_params:>9}  {r.short_desc}")
        for r in recs:
            key_models[r.kernel_key].add(vname)
            key_desc[r.kernel_key] = r.short_desc
        del det
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if len(names) > 1:
        print(f"\n== reuse [{args.level}]: {len(key_models)} unique kernels across {len(names)} models")
        shared = {k: v for k, v in key_models.items() if len(v) > 1}
        print(f"   shared by >1 model: {len(shared)}")
        seen: set[str] = set()
        for name in names:
            vname = variant_name(name, not args.one2many)
            ks = {k for k, v in key_models.items() if vname in v}
            print(f"   adding {vname}: +{len(ks - seen)} new keys (cumulative {len(ks | seen)})")
            seen |= ks
        if not args.quiet:
            for k, v in sorted(shared.items(), key=lambda kv: key_desc[kv[0]]):
                print(f"   {k} {key_desc[k]:55s} {sorted(v)}")


if __name__ == "__main__":
    main()
