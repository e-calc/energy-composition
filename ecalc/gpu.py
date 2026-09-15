"""NVML helpers: device handle, snapshots, persistence mode, best-effort clock
locking and a background clock/power sampler.

Clock lock/reset follow kareus ``tests/bayesian/common/hardware.py`` but are
best-effort because the non-root user gets NVMLError_NoPermission.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
import threading
import time
from typing import Any

import pynvml

_NVML_INIT = False


def nvml_init() -> None:
    global _NVML_INIT
    if not _NVML_INIT:
        pynvml.nvmlInit()
        _NVML_INIT = True


def _decode(s: Any) -> str:
    return s.decode() if isinstance(s, bytes) else str(s)


def nvml_handle_for_torch_device(idx: int = 0):
    """NVML handle for ``cuda:idx`` matched by UUID (robust to CUDA_VISIBLE_DEVICES)."""
    import torch

    nvml_init()
    uuid = str(torch.cuda.get_device_properties(idx).uuid)
    return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode() if not uuid.startswith("GPU-") else uuid.encode())


def nvml_handle_for_visible_index(idx: int = 0):
    """NVML handle for CUDA-visible device ``idx`` WITHOUT touching torch.cuda (no CUDA context)."""
    import os

    nvml_init()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        ids = [v.strip() for v in vis.split(",") if v.strip()]
        sel = ids[idx]
        if sel.startswith("GPU-") or sel.startswith("MIG-"):
            return pynvml.nvmlDeviceGetHandleByUUID(sel.encode())
        return pynvml.nvmlDeviceGetHandleByIndex(int(sel))
    return pynvml.nvmlDeviceGetHandleByIndex(int(idx))


def nvml_index(handle) -> int:
    return int(pynvml.nvmlDeviceGetIndex(handle))


def gpu_name(handle) -> str:
    """Normalised GPU name, e.g. ``"NVIDIA A40" -> "A40"``."""
    name = _decode(pynvml.nvmlDeviceGetName(handle))
    for prefix in ("NVIDIA ", "Tesla "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace(" ", "_")


def driver_version() -> str:
    nvml_init()
    return _decode(pynvml.nvmlSystemGetDriverVersion())


def max_sm_clock_mhz(handle) -> int:
    return int(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM))


@dataclasses.dataclass
class GpuSnapshot:
    pstate: int
    sm_clock_mhz: int
    mem_clock_mhz: int
    temp_c: int
    power_w: float
    persistence_mode: bool
    t: float

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def snapshot(handle) -> GpuSnapshot:
    return GpuSnapshot(
        pstate=int(pynvml.nvmlDeviceGetPerformanceState(handle)),
        sm_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
        mem_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
        temp_c=int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
        power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        persistence_mode=bool(pynvml.nvmlDeviceGetPersistenceMode(handle)),
        t=time.perf_counter(),
    )


def persistence_mode_on(handle) -> bool:
    return bool(pynvml.nvmlDeviceGetPersistenceMode(handle))


def ensure_persistence_mode(handle, required: bool = True) -> bool:
    """Return True if persistence mode is on. Try to enable it; if that needs
    root, print the command the user must run and raise when ``required``."""
    if persistence_mode_on(handle):
        return True
    try:
        pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
    except pynvml.NVMLError as e:  # NoPermission on a non-root user
        idx = nvml_index(handle)
        msg = (
            f"GPU persistence mode is DISABLED on GPU {idx} and enabling it needs root "
            f"({e}). Run once:\n    sudo nvidia-smi -i {idx} -pm 1\n"
            f"and verify with: nvidia-smi --query-gpu=persistence_mode --format=csv"
        )
        if required:
            raise RuntimeError(msg) from None
        print("WARNING: " + msg, file=sys.stderr)
        return False
    return persistence_mode_on(handle)


def try_lock_clocks(handle, mhz: int) -> bool:
    """kareus ``_set_gpu_frequency`` pattern; returns False when not permitted."""
    try:
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, int(mhz), int(mhz))
    except pynvml.NVMLError as e:
        print(f"WARNING: cannot lock GPU clocks to {mhz} MHz ({e}); using default clocks", file=sys.stderr)
        return False
    time.sleep(2)
    return True


def reset_clocks(handle) -> bool:
    try:
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    except pynvml.NVMLError:
        return False
    time.sleep(1)
    return True


class ClockSampler:
    """Background thread sampling SM/mem clock, power and pstate at a fixed period."""

    def __init__(self, handle, period_s: float = 0.05):
        self.handle = handle
        self.period_s = period_s
        self.samples: list[tuple[float, int, int, float, int]] = []  # (t, sm, mem, power_w, pstate)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ClockSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        h = self.handle
        while not self._stop.is_set():
            try:
                self.samples.append((
                    time.perf_counter(),
                    int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
                    int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)),
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    int(pynvml.nvmlDeviceGetPerformanceState(h)),
                ))
            except pynvml.NVMLError:
                pass
            self._stop.wait(self.period_s)

    def summary(self, since: float | None = None) -> dict:
        rows = [s for s in self.samples if since is None or s[0] >= since]
        if not rows:
            return dict(n=0, sm_clock_mhz_mean=None, sm_clock_mhz_min=None, sm_clock_mhz_max=None,
                        mem_clock_mhz_mean=None, power_w_mean=None, pstate_max=None)
        sm = [r[1] for r in rows]
        mem = [r[2] for r in rows]
        pw = [r[3] for r in rows]
        ps = [r[4] for r in rows]
        return dict(
            n=len(rows),
            sm_clock_mhz_mean=statistics.fmean(sm),
            sm_clock_mhz_min=min(sm),
            sm_clock_mhz_max=max(sm),
            mem_clock_mhz_mean=statistics.fmean(mem),
            power_w_mean=statistics.fmean(pw),
            pstate_max=max(ps),
        )
