"""GPU measurement commands: static power, the kernel database, and whole-model runs.

These are the bodies of ``ecalc measure static-power | kernels | model``; the files under
``scripts/`` are thin wrappers that call them. Everything here needs a CUDA GPU, torch, Zeus
and NVML, so this module is imported only when a measurement is actually requested.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fcntl
import json
import os
import sys
import time

os.environ.setdefault("KINETO_LOG_LEVEL", "5")  # silence torch.profiler USDT chatter on stderr

import torch  # noqa: E402

from . import config as C, db  # noqa: E402
from .calculus import DIRECTIONS, FORMULA, FORWARD, INTERPOLATED, Platform  # noqa: E402
from .capture import capture, model_composition  # noqa: E402
from .gpu import (ensure_persistence_mode, gpu_name, max_sm_clock_mhz, nvml_handle_for_torch_device,  # noqa: E402
                  nvml_handle_for_visible_index, reset_clocks, snapshot, try_lock_clocks)
from .measure import (gpu_kernel_time, kernel_callable, make_monitor, measure_kernel, measure_model_e2e,  # noqa: E402
                      model_callable)
from .models import (default_size, load_fused_model, load_model, make_input, model_input, parse_model_list,  # noqa: E402
                     precision_flags, set_precision, variant_name, versions)
from .static_power import (measure_active_idle_power, measure_p8_idle_power, p0_plateau_s,  # noqa: E402
                           record_power_decay)


# ---- shared options and session ----------------------------------------------------------------

def add_measure_arguments(parser: argparse.ArgumentParser) -> None:
    """Options shared by every measurement command. The short spellings of the former scripts
    (``--db``, ``--batch``, ``--imgsz``) are still accepted."""
    parser.add_argument("--store", "--db", dest="db", default=str(C.DB_PATH),
                        help="SQLite measurement store (default data/ecalc.sqlite)")
    parser.add_argument("--device", type=int, default=0, help="torch cuda device index")
    parser.add_argument("--batch-size", "--batch", dest="batch", type=int, default=C.BATCH, help="input batch size")
    parser.add_argument("--input-size", "--image-size", "--imgsz", dest="imgsz", type=int, default=None,
                        help="YOLO image side in pixels (default 640) or RFML frame length (default 1024)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Zeus trials per kernel (median kept); 1 is enough, repeat CV is <2%% on the A40")
    parser.add_argument("--measurement-duration", type=float, default=5.0)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--calibration", type=int, default=1000)
    parser.add_argument("--settle", type=float, default=None,
                        help="run the workload this many seconds before each Zeus window (warmup_settle_duration; "
                             "replaces --warmup for the trial) so the GPU reaches its thermal state")
    parser.add_argument("--lock-mhz", type=int, default=None, help="best-effort SM clock lock (needs root)")
    parser.add_argument("--strict-fp32", action="store_true", help="disable TF32 (precision label fp32_strict)")
    parser.add_argument("--profile-kernel-time", action="store_true")
    parser.add_argument("--one2many", action="store_true", help="measure the one-to-many head variant (<name>-nms)")
    parser.add_argument("--allow-no-persistence", action="store_true",
                        help="measure even if GPU persistence mode is off (rows are flagged no_persistence_mode)")


def add_model_argument(parser: argparse.ArgumentParser, default: str = "yolo26n") -> None:
    parser.add_argument("--models", "--model", dest="models", default=default,
                        help="comma list, 'all' (YOLO26 sizes) or 'rfml'")


def add_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level", choices=list(C.LEVELS), default="module",
                        help="kernel granularity: top-level layer (module) or leaf module + glue op (leaf)")


def add_direction_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--direction", choices=list(DIRECTIONS), default=FORWARD,
                        help="forward pass, or backward pass (module level only)")


def make_config(args) -> C.MeasureConfig:
    return C.MeasureConfig(
        measurement_duration_s=args.measurement_duration, cooldown_s=args.cooldown, num_warmup=args.warmup,
        num_calibration=args.calibration, warmup_settle_s=args.settle, repeats=args.repeats,
        dtype=set_precision(args.strict_fp32), profile_kernel_time=args.profile_kernel_time, lock_mhz=args.lock_mhz,
    )


class Session:
    """GPU handle + name + lock file + JSON log for one measurement run."""

    def __init__(self, args, script: str, require_persistence: bool = True):
        C.ensure_dirs()
        self.args = args
        self.device = f"cuda:{args.device}"
        self.batch = int(getattr(args, "batch", C.BATCH))
        self.imgsz_arg = getattr(args, "imgsz", None)
        self.imgsz = int(self.imgsz_arg or C.IMGSZ)
        self.input_shape = (self.batch, 3, self.imgsz, self.imgsz)  # YOLO input (static-power bursts)
        torch.cuda.set_device(args.device)
        self.handle = nvml_handle_for_torch_device(args.device)
        self.gpu_name = gpu_name(self.handle)
        self.cfg = make_config(args)
        self.lock_f = open(C.LOCK_PATH, "w")
        try:
            fcntl.flock(self.lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"another measurement process holds {C.LOCK_PATH}")
        pm = ensure_persistence_mode(self.handle, required=require_persistence and not args.allow_no_persistence)
        if not pm:
            print("WARNING: continuing WITHOUT persistence mode (results flagged)", file=sys.stderr)
        self.persistence = pm
        if args.lock_mhz:
            if try_lock_clocks(self.handle, args.lock_mhz):
                self.cfg.freq_label = str(args.lock_mhz)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = C.LOGS_DIR / f"{script}_{stamp}.log"
        self.meta = dict(script=script, args=vars(args), gpu=self.gpu_name, cfg=vars(self.cfg),
                         snapshot=snapshot(self.handle).as_dict(), versions=versions(), precision=precision_flags())
        self.log(json.dumps(self.meta, default=str))

    @property
    def platform(self) -> Platform:
        return Platform(self.gpu_name, self.cfg.freq_label, self.cfg.dtype)

    def size_for(self, name: str) -> int:
        """``--input-size`` if given, else the model family default (640 px YOLO, 1024-sample RFML frame)."""
        return int(self.imgsz_arg or default_size(name))

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def flag(self, row: dict) -> dict:
        """Mark a row measured without GPU persistence mode."""
        if not self.persistence:
            row["notes"] = ";".join(filter(None, [row.get("notes"), "no_persistence_mode"]))
        return row

    def close(self) -> None:
        if self.args.lock_mhz:
            reset_clocks(self.handle)
        fcntl.flock(self.lock_f, fcntl.LOCK_UN)
        self.lock_f.close()


# ---- ecalc measure kernels ---------------------------------------------------------------------

def kernels_main(argv: list[str] | None = None) -> None:
    """Capture every kernel of the given models and measure each unique kernel in isolation.

    Resumable: keys already present for (gpu, clock, precision) are skipped unless --force. Keys
    do not depend on the level, so leaf rows that coincide with a top-level layer reuse the
    module-level row. Each kernel is committed as soon as it is measured, so Ctrl-C keeps the
    rows done so far. With ``--fill formula`` kernels the formulas can already estimate from the
    store are not measured.
    """
    parser = argparse.ArgumentParser(prog="ecalc measure kernels", description=kernels_main.__doc__)
    add_measure_arguments(parser)
    add_model_argument(parser, "all")
    add_level_argument(parser)
    add_direction_argument(parser)
    parser.add_argument("--input-source", choices=["captured", "synth"], default="captured")
    parser.add_argument("--force", action="store_true", help="re-measure keys already in the store")
    parser.add_argument("--fill", choices=["formula"], default=None,
                        help="skip kernels that formula mode can interpolate or fit from the store")
    parser.add_argument("--layers", default=None,
                        help="comma list of top-level layer indices to measure (leaf level: their leaves) (debug)")
    args = parser.parse_args(argv)
    s = Session(args, "measure_kernels")
    conn = db.connect(args.db)
    monitor = make_monitor(args.device)
    done_this_run: set[str] = set()
    only = {int(i) for i in args.layers.split(",")} if args.layers else None
    t_start = time.time()
    n_meas = n_filled = 0
    to_profile: list = []  # (LayerRecord) measured or already present; profiled after ALL Zeus measurements
    formulas = None
    if args.fill == "formula":
        from .calculus import SourceChain
        from .formula import formula_sources

        formulas = SourceChain(formula_sources(db.signature_database(conn)))
    try:
        for name in parse_model_list(args.models):
            vname = variant_name(name, not args.one2many)
            size = s.size_for(name)
            det = load_model(name, s.device, end2end=not args.one2many)
            x = model_input(name, s.batch, size, s.device)
            recs = capture(det, x, level=args.level, direction=args.direction)
            db.replace_model_kernels(conn, vname, s.cfg.dtype, s.batch, size,
                                     model_composition(recs, vname, s.cfg.dtype, s.batch, size,
                                                       s.meta["versions"]["ultralytics_version"]),
                                     level=args.level, direction=args.direction)
            todo = [r for r in recs if (only is None or r.parent_layer_idx in only)]
            n_uniq = len({r.kernel_key for r in recs})
            s.log(f"== {vname} (batch {s.batch}, size {size}, level {args.level}, {args.direction}): "
                  f"{len(recs)} kernels captured, {n_uniq} unique")
            for r in todo:
                if r.kernel_key in done_this_run:
                    continue
                tag = f"L{r.parent_layer_idx:02d}/{r.layer_idx:03d} {r.kernel_key} {r.short_desc}"
                if not args.force and db.has_kernel(conn, r.kernel_key, s.gpu_name, s.cfg.freq_label, s.cfg.dtype):
                    s.log(f"  skip {tag} (in store)")
                    done_this_run.add(r.kernel_key)
                    to_profile.append(r)
                    continue
                if formulas is not None:
                    filled = formulas.lookup(db.kernel_specification(r.kernel_key, json.dumps(r.signature)), s.platform)
                    if filled is not None and filled.provenance in (INTERPOLATED, FORMULA):
                        s.log(f"  skip {tag} ({filled.provenance}: {filled.detail})")
                        done_this_run.add(r.kernel_key)
                        n_filled += 1
                        continue
                s.log(f"  measure {tag}")
                row = s.flag(measure_kernel(r, monitor, s.handle, s.cfg, s.gpu_name, input_source=args.input_source,
                                            device=s.device, log=s.log))
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
                with kernel_callable(r) as fn:
                    t = gpu_kernel_time(fn)
                db.set_kernel_gpu_time(conn, r.kernel_key, s.gpu_name, s.cfg.freq_label, s.cfg.dtype, t)
                s.log(f"  L{r.parent_layer_idx:02d}/{r.layer_idx:03d} {r.kernel_key} gpu_kernel_time "
                      f"{t * 1e6 if t else float('nan'):.1f} us")
    except KeyboardInterrupt:
        s.log("interrupted; committed rows are kept")
    finally:
        filled = f", left {n_filled} to the formulas" if formulas is not None else ""
        s.log(f"measured {n_meas} kernels in {(time.time() - t_start) / 60:.1f} min{filled}")
        s.close()


# ---- ecalc measure model -----------------------------------------------------------------------

def model_main(argv: list[str] | None = None) -> None:
    """Measure each whole model with the same Zeus protocol: the reference a composition is compared with."""
    parser = argparse.ArgumentParser(prog="ecalc measure model", description=model_main.__doc__)
    add_measure_arguments(parser)
    add_model_argument(parser, "all")
    add_direction_argument(parser)
    args = parser.parse_args(argv)
    s = Session(args, "measure_model")
    conn = db.connect(args.db)
    monitor = make_monitor(args.device)
    loaded = []
    try:
        for name in parse_model_list(args.models):
            vname = variant_name(name, not args.one2many)
            size = s.size_for(name)
            det = load_model(name, s.device, end2end=not args.one2many)
            x = model_input(name, s.batch, size, s.device)
            s.log(f"== {vname} end-to-end {args.direction} (batch {s.batch}, size {size})")
            row = s.flag(measure_model_e2e(det, x, monitor, s.handle, s.cfg, s.gpu_name, vname, s.batch, size,
                                           log=s.log, direction=args.direction))
            db.upsert_model_run(conn, row)
            s.log(f"    -> {row['energy_j'] * 1e3:.3f} mJ  {row['time_s'] * 1e6:.1f} us  {row['power_w']:.1f} W  "
                  f"{row['notes'] or ''}")
            if args.profile_kernel_time:
                loaded.append((vname, det, x, size))
            else:
                del det
                torch.cuda.empty_cache()
        for vname, det, x, size in loaded:  # profiler last: it slows down all later CUDA launches in this process
            with model_callable(det, x, args.direction) as fn:
                t = gpu_kernel_time(fn)
            db.set_model_run_gpu_time(conn, vname, s.gpu_name, s.cfg.freq_label, s.cfg.dtype, s.batch, size, t,
                                      direction=args.direction)
            s.log(f"    {vname} gpu_kernel_time {t * 1e6 if t else float('nan'):.1f} us")
    finally:
        s.close()


# ---- ecalc measure static-power ----------------------------------------------------------------

def static_power_main(argv: list[str] | None = None) -> None:
    """Measure static power definitions and store them.

    active_idle (primary): Zeus window --settle seconds after a burst of full-model forwards (GPU at P0).
    post_burst (reference): 0.2 s window immediately after the burst (includes the decaying transient).
    p8_idle (optional): sleeping-GPU floor.
    Also records a post-burst power decay trace to results/static_power/decay_<gpu>.csv.
    """
    parser = argparse.ArgumentParser(prog="ecalc measure static-power", description=static_power_main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter, conflict_handler="resolve")
    add_measure_arguments(parser)
    add_model_argument(parser, "yolo26n")
    parser.add_argument("--burst", type=float, default=2.0, help="seconds of model forwards before each idle window")
    parser.add_argument("--idle-window", type=float, default=1.0, help="seconds of the Zeus window")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds between burst end and window start")
    parser.add_argument("--n", "--cycles", dest="n", type=int, default=20, help="number of burst/idle cycles")
    parser.add_argument("--no-post-burst", action="store_true", help="skip the immediate post-burst reference row")
    parser.add_argument("--auto-window", action="store_true", help="set idle window from the measured P0 plateau")
    parser.add_argument("--with-p8", action="store_true", help="also measure the P8 sleeping floor (~1 min)")
    parser.add_argument("--no-decay", action="store_true")
    parser.add_argument("--p8-only", action="store_true", help="only (re)measure the P8 sleeping floor")
    args = parser.parse_args(argv)
    if args.p8_only:
        args.with_p8 = True
    p8_row = None
    if args.with_p8:
        # before Session(): torch.cuda.set_device() would create a CUDA context and keep the GPU at P0
        h = nvml_handle_for_visible_index(args.device)
        print("p8_idle: waiting for the GPU to drop to P8 (no CUDA context yet) ...", flush=True)
        p8_row = measure_p8_idle_power(h, log=print)
    s = Session(args, "measure_static_power")
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
                  f"power at {trace[-1]['t_s']:.1f}s = {trace[-1]['power_w']:.1f} W "
                  f"(P{trace[-1]['pstate']}, {trace[-1]['sm_clock_mhz']} MHz)")
            if args.auto_window:
                window = max(0.05, min(args.idle_window, 0.8 * (plateau - args.settle)))
                s.log(f"  auto idle window = {window:.3f}s (settle {args.settle}s)")
        common = dict(gpu_name=s.gpu_name, freq_label=s.cfg.freq_label, model_name=model_name,
                      driver_version=s.meta["versions"]["driver_version"])
        s.log(f"active_idle: {args.n} cycles of {args.burst}s burst, {args.settle}s settle, {window:.3f}s window "
              f"on {model_name}")
        row = measure_active_idle_power(det, x, monitor, s.handle, burst_s=args.burst, idle_window_s=window,
                                        n=args.n, settle_s=args.settle, log=s.log,
                                        require_persistence=not args.allow_no_persistence)
        row.update(common)
        db.upsert_static(conn, row)
        s.log(f"active_idle = {row['power_w']:.2f} W ({row['n_samples']} accepted, {row['n_rejected']} rejected; "
              f"{row['notes']})")
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
