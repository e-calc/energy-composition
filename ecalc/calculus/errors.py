"""Errors raised by the energy calculus."""

from __future__ import annotations


class EnergyCalculusError(Exception):
    """Base class of every error raised by ``ecalc.calculus``."""


class NegativeDynamicEnergy(EnergyCalculusError, ValueError):
    """A kernel's measured energy is below ``s * T``: the static power is too high for it.

    Raised rather than clamped. Clamping ``d`` to zero would invent energy and hide that the
    static power does not fit the kernel's measurement.
    """


class MissingKernel(EnergyCalculusError, LookupError):
    """A kernel of the expression has no signature in the source."""


class PlatformMismatch(EnergyCalculusError, ValueError):
    """Signatures or static power from different platforms were combined."""


class SchemaVersionError(EnergyCalculusError, ValueError):
    """A serialized document has an unknown schema or version."""
