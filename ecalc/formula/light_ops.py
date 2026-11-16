"""Price parameter-free operations from the bytes they read.

Operations without parameters (``cat``, ``add``, ``chunk``, ``split``, pooling, upsampling,
activations) are memory-bound: their time grows with the bytes moved and they draw a roughly
constant power. For each family -- same operation, same arguments, same dtype and direction --
a line is fitted to the measured rows of the database in use:

    T = overhead + bytes / bandwidth
    E = power * T

``bytes`` is the size of the inputs. Nothing here is a constant of a particular GPU: the fit
comes from the user's own measurements, and a family with fewer than ``minimum_points``
measured sizes (two by default: a line needs two points) is not estimated at all.

A size inside the fitted range is labelled ``formula``; outside it, ``extrapolated``.
"""

from __future__ import annotations

import dataclasses
import functools
import math

from ..calculus import (EXTRAPOLATED, FORMULA, MEASURED, KernelSpecification, Platform, Signature,
                        SignatureDatabase, canonical_json)


@functools.lru_cache(maxsize=None)
def light_family(specification: KernelSpecification) -> tuple[str, int] | None:
    """``(family, input bytes)`` of a parameter-free kernel; None for anything else."""
    if not specification.is_resolved or specification.has_parameters or not specification.inputs:
        return None
    signature = {k: v for k, v in specification.signature.items() if k != "input"}
    signature["input_dtypes"] = sorted({t.dtype for t in specification.inputs})
    signature["input_layouts"] = sorted({t.layout for t in specification.inputs})
    return canonical_json(signature), sum(t.byte_count for t in specification.inputs)


@dataclasses.dataclass(frozen=True)
class LightOpsFit:
    """The fitted line of one family."""

    name: str
    point_count: int
    overhead_seconds: float
    seconds_per_byte: float
    power_watts: float
    minimum_bytes: int
    maximum_bytes: int

    @property
    def bandwidth_bytes_per_second(self) -> float:
        return 1.0 / self.seconds_per_byte if self.seconds_per_byte > 0 else math.inf

    def time_seconds(self, byte_count: int) -> float:
        return self.overhead_seconds + self.seconds_per_byte * byte_count

    def describe(self) -> str:
        bandwidth = ("no size dependence" if self.seconds_per_byte == 0
                     else f"{self.bandwidth_bytes_per_second / 1e9:.1f} GB/s")
        return (f"{self.name}: {self.overhead_seconds * 1e6:.2f} us + bytes at {bandwidth}, "
                f"{self.power_watts:.1f} W, fitted on {self.point_count} measured sizes")


def fit_line(points: list[tuple[int, float, float]], name: str) -> LightOpsFit:
    """Least-squares ``T = overhead + bytes * seconds_per_byte`` over ``(bytes, T, E)`` points.

    A negative slope becomes a constant time (the mean); a negative overhead becomes a line
    through the origin. Power is total energy over total time."""
    xs = [float(p[0]) for p in points]
    ts = [p[1] for p in points]
    n = len(points)
    mean_x, mean_t = math.fsum(xs) / n, math.fsum(ts) / n
    variance = math.fsum((x - mean_x) ** 2 for x in xs)
    slope = math.fsum((x - mean_x) * (t - mean_t) for x, t in zip(xs, ts)) / variance if variance > 0 else 0.0
    if slope <= 0:
        slope, overhead = 0.0, mean_t
    else:
        overhead = mean_t - slope * mean_x
        if overhead < 0:
            overhead = 0.0
            slope = math.fsum(x * t for x, t in zip(xs, ts)) / math.fsum(x * x for x in xs)
    total_time = math.fsum(ts)
    power = math.fsum(p[2] for p in points) / total_time if total_time > 0 else 0.0
    return LightOpsFit(name, n, overhead, slope, power, int(min(xs)), int(max(xs)))


class LightOpsFormula:
    """A signature source for parameter-free kernels, fitted on a measured database."""

    def __init__(self, database: SignatureDatabase, minimum_points: int = 2) -> None:
        self.minimum_points = minimum_points
        grouped: dict[tuple[Platform, str], dict[int, list[Signature]]] = {}
        names: dict[tuple[Platform, str], str] = {}
        for signature in database.signatures():
            if signature.provenance != MEASURED:
                continue
            family = light_family(signature.kernel)
            if family is None:
                continue
            index = (signature.platform, family[0])
            grouped.setdefault(index, {}).setdefault(family[1], []).append(signature)
            names.setdefault(index, signature.kernel.name)
        self._fits: dict[tuple[Platform, str], LightOpsFit] = {}
        for index, by_size in grouped.items():
            if len(by_size) < minimum_points:
                continue
            points = [(size, s.time_seconds, s.energy_joules) for size, members in by_size.items() for s in members]
            self._fits[index] = fit_line(points, names[index])

    def fits(self, platform: Platform | None = None) -> list[LightOpsFit]:
        return [fit for (p, _), fit in self._fits.items() if platform is None or p == platform]

    def lookup(self, specification: KernelSpecification, platform: Platform) -> Signature | None:
        family = light_family(specification)
        if family is None:
            return None
        fit = self._fits.get((platform, family[0]))
        if fit is None:
            return None
        byte_count = family[1]
        time_seconds = fit.time_seconds(byte_count)
        inside = fit.minimum_bytes <= byte_count <= fit.maximum_bytes
        detail = fit.describe() + ("" if inside else
                                   f"; {byte_count} bytes is outside the fitted range "
                                   f"[{fit.minimum_bytes}, {fit.maximum_bytes}]")
        return Signature(specification, platform, time_seconds, fit.power_watts * time_seconds,
                         FORMULA if inside else EXTRAPOLATED, detail)
