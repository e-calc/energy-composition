"""Shared CLI plumbing for the measurement scripts."""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KINETO_LOG_LEVEL", "5")  # silence torch.profiler USDT chatter on stderr

import torch  # noqa: E402

from ecalc import config as C  # noqa: E402
from ecalc.gpu import (ensure_persistence_mode, gpu_name, nvml_handle_for_torch_device, reset_clocks,  # noqa: E402
                       snapshot, try_lock_clocks)
from ecalc.models import default_size, parse_model_list, precision_flags, set_precision, versions  # noqa: E402


def add_measure_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--db", default=str(C.DB_PATH))
    ap.add_argument("--device", type=int, default=0, help="torch cuda device index")
    ap.add_argument("--batch", type=int, default=C.BATCH, help="input batch size")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="YOLO input image size (default 640) or RFML IQ frame length (default 1024)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Zeus trials per kernel (median kept); 1 is enough, repeat CV is <2% on the A40")
    ap.add_argument("--measurement-duration", type=float, default=5.0)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--calibration", type=int, default=1000)
    ap.add_argument("--settle", type=float, default=None,
                    help="run the workload this many seconds before each Zeus window (warmup_settle_duration; "
                         "replaces --warmup for the trial) so the GPU reaches its thermal state")
    ap.add_argument("--lock-mhz", type=int, default=None, help="best-effort SM clock lock (needs root)")
    ap.add_argument("--strict-fp32", action="store_true", help="disable TF32 (dtype label fp32_strict)")
    ap.add_argument("--profile-kernel-time", action="store_true")
    ap.add_argument("--one2many", action="store_true", help="measure the one-to-many head variant (<name>-nms)")
    ap.add_argument("--allow-no-persistence", action="store_true",
                    help="measure even if GPU persistence mode is off (smoke tests only)")


def add_model_args(ap: argparse.ArgumentParser, default: str = "yolo26n") -> None:
    ap.add_argument("--models", default=default, help="comma list or 'all'")


def add_level_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--level", choices=list(C.LEVELS), default="module",
                    help="kernel granularity: top-level Ultralytics layer (module) or leaf module + glue op (leaf)")


def parse_models(s: str) -> list[str]:
    """``all`` (YOLO26 sizes), ``rfml`` (RFML models) or a comma list."""
    return parse_model_list(s)


def make_cfg(args) -> C.MeasureConfig:
    return C.MeasureConfig(
        measurement_duration_s=args.measurement_duration, cooldown_s=args.cooldown, num_warmup=args.warmup,
        num_calibration=args.calibration, warmup_settle_s=args.settle, repeats=args.repeats, dtype=set_precision(args.strict_fp32),
        profile_kernel_time=args.profile_kernel_time, lock_mhz=args.lock_mhz,
    )


class Session:
    """GPU handle + name + lock file + JSON log for one script run."""

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
        self.cfg = make_cfg(args)
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

    def size_for(self, name: str) -> int:
        """``--imgsz`` if given, else the model family default (640 px YOLO, 1024-sample RFML frame)."""
        return int(self.imgsz_arg or default_size(name))

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def close(self) -> None:
        if self.args.lock_mhz:
            reset_clocks(self.handle)
        fcntl.flock(self.lock_f, fcntl.LOCK_UN)
        self.lock_f.close()
