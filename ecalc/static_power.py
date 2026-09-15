"""Static-power definitions.

``active_idle`` (primary): power in a Zeus window opened ``settle_s`` after a burst
of full-model forwards, while the GPU is still at P0 / boost clocks but the
post-burst transient has decayed (A40: ~1 s). Cycles where the GPU already
ramped down are rejected.
``post_burst`` (reference): the same with ``settle_s = 0`` and a short window,
i.e. immediately after the burst; includes the decaying transient.
``p8_idle`` (reference): the sleeping-GPU floor.
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics
import time

import pynvml
import torch
from zeus.monitor import ZeusMonitor

from .gpu import ensure_persistence_mode, max_sm_clock_mhz, persistence_mode_on, snapshot

METHODS = ("active_idle", "post_burst", "p8_idle")
P0_CLOCK_FRACTION = 0.95


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@torch.inference_mode()
def _burst(det, x, burst_s: float) -> None:
    t_end = time.perf_counter() + burst_s
    while time.perf_counter() < t_end:
        det(x)
    torch.cuda.synchronize()


def measure_active_idle_power(det, x, monitor: ZeusMonitor, handle, burst_s: float = 2.0,
                              idle_window_s: float = 1.0, n: int = 20, settle_s: float = 1.0, log=None,
                              require_persistence: bool = True, method: str = "active_idle") -> dict:
    """Median power over accepted post-burst windows (GPU still at P0 and >= 95% boost clock)."""
    ensure_persistence_mode(handle, required=require_persistence)
    max_sm = max_sm_clock_mhz(handle)
    samples, rejected = [], []
    for i in range(n):
        _burst(det, x, burst_s)
        if settle_s > 0:
            time.sleep(settle_s)
        monitor.begin_window(method)
        time.sleep(idle_window_s)
        snap = snapshot(handle)  # state at the end of the window, before end_window sync
        res = monitor.end_window(method)
        power = res.total_energy / res.time
        ok = snap.pstate == 0 and snap.sm_clock_mhz >= P0_CLOCK_FRACTION * max_sm
        entry = dict(cycle=i, power_w=power, energy_j=res.total_energy, time_s=res.time, pstate=snap.pstate,
                     sm_clock_mhz=snap.sm_clock_mhz, mem_clock_mhz=snap.mem_clock_mhz, temp_c=snap.temp_c,
                     accepted=ok)
        (samples if ok else rejected).append(entry)
        if log:
            log(f"  cycle {i + 1}/{n}: {power:.2f} W  P{snap.pstate} {snap.sm_clock_mhz} MHz  {'ok' if ok else 'REJECT'}")
    if not samples:
        raise RuntimeError("no active_idle cycle stayed at P0; shorten --idle-window or check the decay trace")
    powers = [s["power_w"] for s in samples]
    last = samples[-1]
    return dict(
        method=method, power_w=statistics.median(powers), n_samples=len(samples), n_rejected=len(rejected),
        samples_json=json.dumps(samples + rejected), pstate=last["pstate"], sm_clock_mhz=last["sm_clock_mhz"],
        mem_clock_mhz=last["mem_clock_mhz"], temp_c=last["temp_c"], burst_s=burst_s, window_s=idle_window_s,
        persistence_mode=int(persistence_mode_on(handle)), measured_at=_now(),
        notes=f"settle={settle_s}s;std={statistics.pstdev(powers):.3f}W;min={min(powers):.2f};max={max(powers):.2f}",
    )


def record_power_decay(det, x, handle, burst_s: float = 2.0, trace_s: float = 5.0, period_s: float = 0.01) -> list[dict]:
    """Poll NVML power/clock/pstate after a burst; rows relative to the burst end."""
    _burst(det, x, burst_s)
    t0 = time.perf_counter()
    rows = []
    while True:
        t = time.perf_counter() - t0
        if t > trace_s:
            break
        rows.append(dict(
            t_s=t,
            power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            sm_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            mem_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
            pstate=int(pynvml.nvmlDeviceGetPerformanceState(handle)),
        ))
        time.sleep(period_s)
    return rows


def p0_plateau_s(trace: list[dict], max_sm: int) -> float:
    """Seconds after the burst until the first sample that left P0 / boost clocks."""
    for r in trace:
        if r["pstate"] != 0 or r["sm_clock_mhz"] < P0_CLOCK_FRACTION * max_sm:
            return r["t_s"]
    return trace[-1]["t_s"] if trace else 0.0


def measure_p8_idle_power(handle, window_s: float = 10.0, n: int = 3, wait_s: float = 60.0, log=None) -> dict:
    """Sleeping-GPU floor from the raw NVML energy counter.

    Must run before this process (or any other) holds a CUDA context on the GPU: a
    Zeus window syncs with torch and would create one, which keeps the GPU at P0.
    Waits up to ``wait_s`` for P8, then averages ``n`` windows of ``window_s``.
    """
    t_end = time.perf_counter() + wait_s
    while time.perf_counter() < t_end and snapshot(handle).pstate < 8:
        time.sleep(1.0)
    samples = []
    for i in range(n):
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        t0 = time.perf_counter()
        time.sleep(window_s)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        dt = time.perf_counter() - t0
        snap = snapshot(handle)
        samples.append(dict(cycle=i, power_w=(e1 - e0) / 1000.0 / dt, energy_j=(e1 - e0) / 1000.0, time_s=dt,
                            pstate=snap.pstate, sm_clock_mhz=snap.sm_clock_mhz, temp_c=snap.temp_c,
                            accepted=snap.pstate >= 8))
        if log:
            log(f"  p8 window {i + 1}/{n}: {samples[-1]['power_w']:.2f} W  P{snap.pstate} {snap.sm_clock_mhz} MHz")
    powers = [s["power_w"] for s in samples]
    last = samples[-1]
    notes = None if last["pstate"] >= 8 else f"GPU never reached P8 (P{last['pstate']}); is another process holding a CUDA context?"
    return dict(
        method="p8_idle", power_w=statistics.median(powers), n_samples=n,
        n_rejected=sum(1 for s in samples if not s["accepted"]),
        samples_json=json.dumps(samples), pstate=last["pstate"], sm_clock_mhz=last["sm_clock_mhz"],
        mem_clock_mhz=None, temp_c=last["temp_c"], burst_s=0.0, window_s=window_s,
        persistence_mode=int(persistence_mode_on(handle)), measured_at=_now(), notes=notes,
    )
