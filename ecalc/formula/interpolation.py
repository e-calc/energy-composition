"""Estimate an unmeasured kernel from measured kernels of the same family.

A *family* is every kernel that differs only in batch size and in spatial extent (pixels of an
image, samples of a frame): same module, same parameters, same channel counts, same dtype,
layout and direction. Inside a family, time and energy are close to linear in both, so a
missing point is read off a chord between its two nearest measured neighbours:

- between two measured points  -> ``interpolated``
- beyond the measured range    -> ``extrapolated`` (the end chord extended; clamped at zero)
- only one measured point      -> ``extrapolated`` (scaled in proportion)

Only kernels with parameters are handled here. Parameter-free operations (cat, add, pooling,
...) are priced from the bytes they move by ``ecalc.formula.light_ops``.
"""

from __future__ import annotations

import dataclasses
import functools
import math

from ..calculus import (EXTRAPOLATED, INTERPOLATED, MEASURED, KernelSpecification, Platform, Signature,
                        SignatureDatabase, canonical_json)

_FEATURE_FIRST = ("Conv1d", "Conv2d", "ConvTranspose2d")
_FEATURE_LAST = ("Linear", "LSTM", "GRU", "RNN")
_RANK = {MEASURED: 0, INTERPOLATED: 1, EXTRAPOLATED: 2}


@dataclasses.dataclass(frozen=True)
class FamilyCoordinates:
    """Where a kernel sits inside its family."""

    family: str      # canonical JSON of the signature with batch size and extent removed
    batch_size: int
    extent: int      # product of the spatial dimensions, summed over the inputs


def _feature_axis(specification: KernelSpecification) -> int | None:
    """1 when inputs are (batch, channels, spatial...), -1 when they are (batch, ..., features).
    Decided by the first layer with parameters. None when the batch is not the first dimension."""
    for sub in specification.signature["module"]["modules"]:
        if sub["cls"] in _FEATURE_FIRST:
            return 1
        if sub["cls"] in _FEATURE_LAST:
            if sub["cls"] != "Linear" and not sub["a"].get("batch_first", False):
                return None
            return -1
    return 1


@functools.lru_cache(maxsize=None)
def family_coordinates(specification: KernelSpecification) -> FamilyCoordinates | None:
    """Family and coordinates of a kernel, or None when it cannot be interpolated."""
    if not specification.is_resolved or specification.kind != "module" or not specification.has_parameters:
        return None
    inputs = specification.inputs
    feature_axis = _feature_axis(specification)
    if not inputs or feature_axis is None or any(len(t.shape) < 2 for t in inputs):
        return None
    batch_size = inputs[0].shape[0]
    if any(t.shape[0] != batch_size for t in inputs):
        return None
    reduced = []
    extent = 0
    for tensor in inputs:
        rank = len(tensor.shape)
        axis = 1 if (rank == 2 or feature_axis == 1) else rank - 1
        extent += math.prod(d for i, d in enumerate(tensor.shape) if i not in (0, axis))
        reduced.append({"rank": rank, "feature_axis": axis, "features": tensor.shape[axis], "dtype": tensor.dtype,
                        "layout": tensor.layout})
    signature = dict(specification.signature)
    signature["input"] = {"kind": specification.signature["input"]["kind"], "family": reduced}
    return FamilyCoordinates(canonical_json(signature), batch_size, extent)


@dataclasses.dataclass(frozen=True)
class _Point:
    time_seconds: float
    energy_joules: float
    provenance: str
    detail: str


def _chord(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def resolve_along_axis(curve: dict[int, _Point], x: int, axis_name: str) -> _Point:
    """Time and energy at ``x`` on one axis of a family; ``curve`` maps the coordinate to a point."""
    if x in curve:
        return curve[x]
    xs = sorted(curve)
    if len(xs) == 1:
        only = xs[0]
        scale = x / only
        return _Point(curve[only].time_seconds * scale, curve[only].energy_joules * scale, EXTRAPOLATED,
                      f"scaled in proportion from {axis_name} {only}")
    below = [v for v in xs if v < x]
    above = [v for v in xs if v > x]
    if below and above:
        x0, x1 = below[-1], above[0]
        provenance = max(INTERPOLATED, curve[x0].provenance, curve[x1].provenance, key=_RANK.__getitem__)
        label = "chord"
    else:
        x0, x1 = (xs[-2], xs[-1]) if below else (xs[0], xs[1])
        provenance = EXTRAPOLATED
        label = "end chord extended"
    time_seconds = max(_chord(x0, curve[x0].time_seconds, x1, curve[x1].time_seconds, x), 0.0)
    energy_joules = max(_chord(x0, curve[x0].energy_joules, x1, curve[x1].energy_joules, x), 0.0)
    return _Point(time_seconds, energy_joules, provenance, f"{label} over {axis_name} between {x0} and {x1}")


def _quality(curve: dict[int, _Point], x: int) -> int:
    """3 exact, 2 bracketed, 1 beyond the range with two points, 0 a single point, -1 nothing."""
    if not curve:
        return -1
    if x in curve:
        return 3
    if len(curve) == 1:
        return 0
    return 2 if (min(curve) < x < max(curve)) else 1


class Interpolation:
    """A signature source that interpolates inside kernel families of a measured database."""

    def __init__(self, database: SignatureDatabase) -> None:
        self._families: dict[tuple[Platform, str], dict[int, dict[int, _Point]]] = {}
        for signature in database.signatures():
            if signature.provenance != MEASURED:
                continue
            coordinates = family_coordinates(signature.kernel)
            if coordinates is None:
                continue
            rows = self._families.setdefault((signature.platform, coordinates.family), {})
            rows.setdefault(coordinates.batch_size, {})[coordinates.extent] = _Point(
                signature.time_seconds, signature.energy_joules, MEASURED, "")

    def lookup(self, specification: KernelSpecification, platform: Platform) -> Signature | None:
        coordinates = family_coordinates(specification)
        if coordinates is None:
            return None
        rows = self._families.get((platform, coordinates.family))
        if not rows:
            return None
        batch_size, extent = coordinates.batch_size, coordinates.extent
        over_batch = {b: row[extent] for b, row in rows.items() if extent in row}  # same extent, other batches
        over_extent = rows.get(batch_size, {})                                    # same batch, other extents
        if _quality(over_batch, batch_size) >= max(_quality(over_extent, extent), 0):
            point = resolve_along_axis(over_batch, batch_size, "batch size")
        elif over_extent:
            point = resolve_along_axis(over_extent, extent, "extent")
        else:
            # Neither axis has a direct neighbour: bring every measured batch to this extent, then cross batches.
            at_extent = {b: resolve_along_axis(row, extent, "extent") for b, row in rows.items()}
            point = resolve_along_axis(at_extent, batch_size, "batch size")
            point = dataclasses.replace(
                point, provenance=max([point.provenance] + [p.provenance for p in at_extent.values()],
                                      key=_RANK.__getitem__),
                detail=point.detail + f", each first brought to extent {extent}")
        return Signature(specification, platform, point.time_seconds, point.energy_joules, point.provenance,
                         point.detail)
