"""Optional formula mode: estimate kernels that were not measured.

By default every kernel of a pipeline must have a measured signature. Formula mode adds two
sources behind the measured database, so a formula is used only where nothing was measured:

1. ``Interpolation``   kernels with parameters, from measured kernels of the same family at
                       other batch sizes or input sizes.
2. ``LightOpsFormula`` parameter-free operations, from a line fitted on bytes moved.

Both are computed from the database in use, and every result is labelled ``interpolated``,
``extrapolated`` or ``formula`` so it is never mistaken for a measurement. The saving is
measurement time: measure a few batch sizes, estimate the others.
"""

from __future__ import annotations

import dataclasses

from ..calculus import MEASURED, Platform, Signature, SignatureDatabase, SourceChain
from .interpolation import FamilyCoordinates, Interpolation, family_coordinates
from .light_ops import LightOpsFit, LightOpsFormula, light_family

__all__ = ["FamilyCoordinates", "HeldOutResult", "Interpolation", "LightOpsFit", "LightOpsFormula",
           "family_coordinates", "formula_sources", "leave_one_out", "light_family", "with_formulas"]


def formula_sources(database: SignatureDatabase) -> list:
    """The formula sources, in the order they are consulted."""
    return [Interpolation(database), LightOpsFormula(database)]


def with_formulas(database: SignatureDatabase) -> SourceChain:
    """``database`` first, then the formulas fitted on it."""
    return SourceChain([database, *formula_sources(database)])


@dataclasses.dataclass(frozen=True)
class HeldOutResult:
    """One measured kernel, hidden from the formulas and then estimated by them."""

    measured: Signature
    estimated: Signature

    @property
    def energy_error(self) -> float:
        return (self.estimated.energy_joules - self.measured.energy_joules) / self.measured.energy_joules

    @property
    def time_error(self) -> float:
        return (self.estimated.time_seconds - self.measured.time_seconds) / self.measured.time_seconds


def leave_one_out(database: SignatureDatabase, platform: Platform) -> list[HeldOutResult]:
    """Hide each measured signature of ``platform`` in turn and estimate it from the rest.
    Kernels the formulas cannot estimate (no family neighbour, too few sizes) are left out."""
    measured = [s for s in database.signatures(platform) if s.provenance == MEASURED]
    results = []
    for held_out in measured:
        if held_out.energy_joules <= 0 or held_out.time_seconds <= 0:
            continue
        if family_coordinates(held_out.kernel) is None and light_family(held_out.kernel) is None:
            continue
        rest = SignatureDatabase()
        for signature in measured:
            if signature is not held_out:
                rest.add_signature(signature)
        estimated = SourceChain(formula_sources(rest)).lookup(held_out.kernel, platform)
        if estimated is not None:
            results.append(HeldOutResult(held_out, estimated))
    return results
