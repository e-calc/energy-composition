"""Zeus-based energy measurement of one kernel or a whole model.

Follows kareus ``tests/bayesian/common/hardware.py``::

    monitor = ZeusMonitor(gpu_indices=[rank])
    trial = zeus_measure(target_function=fn, zeus_monitor=monitor,
                         measurement_duration=5.0, cooldown_duration=5.0)
    energy_j, time_s = trial.energy_per_iter, trial.time_per_iter

plus: ``repeats`` trials with the median-energy trial kept, a background
``ClockSampler`` restricted to the Zeus measurement window, and an optional
torch.profiler GPU-active time. Dynamic energy is *not* stored; it is derived at
estimation time as ``energy_j - P_static * time_s``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import statistics
import time
from typing import Callable

import torch
from zeus.monitor import ZeusMonitor
from zeus.profile import measure as zeus_measure

from .capture import LayerRecord
from .config import MeasureConfig
from .gpu import ClockSampler, persistence_mode_on
from .models import precision_flags, versions
from .signature import synthesize_inputs


def make_monitor(torch_device_idx: int = 0) -> ZeusMonitor:
    return ZeusMonitor(gpu_indices=[torch_device_idx])


def kernel_target(module_or_rec: torch.nn.Module | LayerRecord, inputs=None) -> Callable[[], None]:
    """Closure running one kernel under inference mode. Accepts a ``LayerRecord`` (module or
    functional op) or a bare module plus its inputs."""
    if isinstance(module_or_rec, LayerRecord):
        return module_or_rec.target(inputs)
    module = module_or_rec
    inference = torch.inference_mode()

    def fn() -> None:
        with inference:
            module(inputs)

    return fn


def model_target(det: torch.nn.Module, x: torch.Tensor) -> Callable[[], None]:
    inference = torch.inference_mode()

    def fn() -> None:
        with inference:
            det(x)

    return fn


@dataclasses.dataclass
class TrialStats:
    energy_j: float
    time_s: float
    iterations: int
    total_energy_j: float
    total_time_s: float
    temp_before: float
    temp_after: float
    clocks: dict

    @property
    def power_w(self) -> float:
        return self.energy_j / self.time_s if self.time_s else float("nan")


def measure_callable(fn: Callable[[], None], monitor: ZeusMonitor, handle, cfg: MeasureConfig,
                     log: Callable[[str], None] | None = None) -> tuple[TrialStats, list[TrialStats]]:
    """Run ``zeus.profile.measure`` ``cfg.repeats`` times; return (median-energy trial, all trials)."""
    trials: list[TrialStats] = []
    for r in range(cfg.repeats):
        with ClockSampler(handle, cfg.clock_sample_period_s) as sampler:
            res = zeus_measure(target_function=fn, zeus_monitor=monitor, **cfg.zeus_kwargs())
            t_end = time.perf_counter()
        # The Zeus window is the tail of the call (after cooldown + warm-up).
        clocks = sampler.summary(since=t_end - res.total_time)
        ts = TrialStats(
            energy_j=float(res.energy_per_iter), time_s=float(res.time_per_iter), iterations=int(res.iterations),
            total_energy_j=float(res.total_energy), total_time_s=float(res.total_time),
            temp_before=float(res.temperature_before), temp_after=float(res.temperature_after), clocks=clocks,
        )
        trials.append(ts)
        if log:
            log(f"  repeat {r + 1}/{cfg.repeats}: {ts.energy_j * 1e3:.3f} mJ/iter, {ts.time_s * 1e6:.1f} us/iter, "
                f"{ts.iterations} iters, SM {clocks.get('sm_clock_mhz_mean')} MHz, "
                f"T {ts.temp_before:.0f}->{ts.temp_after:.0f} C")
    median = sorted(trials, key=lambda t: t.energy_j)[len(trials) // 2]
    return median, trials


def gpu_kernel_time(fn: Callable[[], None], iters: int = 50) -> float | None:
    """Sum of CUDA kernel durations per iteration (launch-gap free), via torch.profiler.

    WARNING: once torch.profiler has run in a process, every later CUDA launch in that
    process is ~30% slower (kineto/CUPTI stays hooked). Call this only after all Zeus
    measurements of the process are finished (see scripts/build_kernel_db.py).
    """
    try:
        from torch.profiler import ProfilerActivity, profile

        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        total_us = 0.0
        for ev in prof.events():
            if getattr(ev, "device_type", None) is not None and "CUDA" in str(ev.device_type):
                total_us += float(getattr(ev, "self_device_time_total", 0.0) or getattr(ev, "self_cuda_time_total", 0.0))
        return total_us / iters / 1e6
    except Exception as e:  # profiler is optional
        print(f"WARNING: gpu_kernel_time failed: {e}")
        return None


def _summary_row(med: TrialStats, trials: list[TrialStats], cfg: MeasureConfig, handle, gpu_name: str) -> dict:
    energies = [t.energy_j for t in trials]
    times = [t.time_s for t in trials]
    c = med.clocks
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    row = dict(
        gpu_name=gpu_name, freq_label=cfg.freq_label, dtype=cfg.dtype,
        energy_j=med.energy_j, time_s=med.time_s, power_w=med.power_w, iterations=med.iterations,
        n_repeats=len(trials),
        # NULL (not 0) when there was a single repeat: "not repeated" is different from "no spread".
        energy_std_j=statistics.pstdev(energies) if len(energies) > 1 else None,
        time_std_s=statistics.pstdev(times) if len(times) > 1 else None,
        sm_clock_mhz_mean=c.get("sm_clock_mhz_mean"), sm_clock_mhz_min=c.get("sm_clock_mhz_min"),
        sm_clock_mhz_max=c.get("sm_clock_mhz_max"), mem_clock_mhz_mean=c.get("mem_clock_mhz_mean"),
        pstate_max=c.get("pstate_max"), temp_before=med.temp_before, temp_after=med.temp_after,
        measurement_duration_s=cfg.measurement_duration_s, cooldown_s=cfg.cooldown_s,
        num_warmup=cfg.num_warmup, num_calibration=cfg.num_calibration,
        persistence_mode=int(persistence_mode_on(handle)),
        measured_at=now, notes=None,
    )
    row.update(versions())
    row.update(precision_flags())
    flags = []
    if med.temp_after - med.temp_before > 5:
        flags.append(f"temp_delta={med.temp_after - med.temp_before:.0f}C")
    if c.get("sm_clock_mhz_min") and c.get("sm_clock_mhz_max") and c["sm_clock_mhz_min"] < 0.95 * c["sm_clock_mhz_max"]:
        flags.append(f"sm_clock_dip={c['sm_clock_mhz_min']}/{c['sm_clock_mhz_max']}")
    if flags:
        row["notes"] = ";".join(flags)
    return row


def measure_kernel(rec: LayerRecord, monitor: ZeusMonitor, handle, cfg: MeasureConfig, gpu_name: str,
                   input_source: str = "captured", device="cuda", log=None) -> dict:
    """Isolated measurement of one layer; returns a ``kernel_energy`` row."""
    if input_source == "captured":
        inputs = rec.inputs
    elif input_source == "synth":
        inputs = synthesize_inputs(rec.input_sig, device)
    else:
        raise ValueError(input_source)
    fn = rec.target(inputs)
    med, trials = measure_callable(fn, monitor, handle, cfg, log=log)
    row = _summary_row(med, trials, cfg, handle, gpu_name)
    row.update(
        kernel_key=rec.kernel_key, kernel_type=rec.layer_type, short_desc=rec.short_desc,
        signature_json=json.dumps(rec.signature, sort_keys=True), input_spec_json=json.dumps(rec.input_sig),
        input_source=input_source, gpu_kernel_time_s=None,
    )
    return row


def measure_model_e2e(det: torch.nn.Module, x: torch.Tensor, monitor: ZeusMonitor, handle, cfg: MeasureConfig,
                      gpu_name: str, model_name: str, batch: int, imgsz: int, log=None) -> dict:
    """Same protocol applied to the full fused forward; returns a ``model_runs`` row."""
    fn = model_target(det, x)
    med, trials = measure_callable(fn, monitor, handle, cfg, log=log)
    row = _summary_row(med, trials, cfg, handle, gpu_name)
    row.update(model_name=model_name, batch=batch, imgsz=imgsz, gpu_kernel_time_s=None)
    return row
