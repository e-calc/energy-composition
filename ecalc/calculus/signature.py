"""Energy signatures: a kernel's measured (or estimated) time and energy on a platform.

A signature stores what is measured, ``(T, E)``. The form of the calculus,
``E = s*T + d``, is a view of it under a static power ``s``:

    d = E - s*T

Dynamic energy is not stored: it is derived from the pair and the platform's static power, which
is measured separately, once per GPU.
"""

from __future__ import annotations

import dataclasses
import math

from .cost import Cost
from .errors import PlatformMismatch
from .kernel import KernelSpecification

MEASURED = "measured"
INTERPOLATED = "interpolated"
EXTRAPOLATED = "extrapolated"
FORMULA = "formula"
PROVENANCES = (MEASURED, INTERPOLATED, EXTRAPOLATED, FORMULA)
"""Where a signature's numbers came from: a measurement of this exact kernel, a chord between
two measured kernels of its family, a line extended past the measured range, or a fitted formula."""


@dataclasses.dataclass(frozen=True)
class Platform:
    """The hardware a signature is valid for: a GPU model at a clock setting."""

    gpu_name: str
    frequency_label: str = "default"  # "default" = unlocked clocks; the locked megahertz otherwise

    def __str__(self) -> str:
        return f"{self.gpu_name}/{self.frequency_label}"


@dataclasses.dataclass(frozen=True)
class StaticPower:
    """A platform's static power, in watts: what the GPU draws while awake and doing nothing."""

    platform: Platform
    watts: float

    def __post_init__(self) -> None:
        watts = float(self.watts)
        if not math.isfinite(watts) or watts < 0:
            raise ValueError(f"static power must be finite and >= 0, got {watts!r}")
        object.__setattr__(self, "watts", watts)


def static_power_watts(static_power: "StaticPower | float", platform: Platform | None = None) -> float:
    """Watts of a ``StaticPower`` or a plain number. A ``StaticPower`` must belong to ``platform``
    when one is given."""
    if isinstance(static_power, StaticPower):
        if platform is not None and static_power.platform != platform:
            raise PlatformMismatch(f"static power of {static_power.platform} used with {platform}")
        return static_power.watts
    watts = float(static_power)
    if not math.isfinite(watts) or watts < 0:
        raise ValueError(f"static power must be finite and >= 0, got {watts!r}")
    return watts


@dataclasses.dataclass(frozen=True)
class Signature:
    """Time and energy of one kernel call on one platform."""

    kernel: KernelSpecification
    platform: Platform
    time_seconds: float
    energy_joules: float
    provenance: str = MEASURED
    detail: str = ""  # anchors of an interpolation, or the fitted formula

    def __post_init__(self) -> None:
        for field in ("time_seconds", "energy_joules"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{self.kernel.key}: {field} must be finite and >= 0, got {value!r}")
            object.__setattr__(self, field, value)
        if self.provenance not in PROVENANCES:
            raise ValueError(f"unknown provenance {self.provenance!r}; known: {PROVENANCES}")

    @property
    def power_watts(self) -> float:
        return self.energy_joules / self.time_seconds if self.time_seconds else float("nan")

    def static_energy(self, static_power: StaticPower | float) -> float:
        """``s * T``."""
        return static_power_watts(static_power, self.platform) * self.time_seconds

    def dynamic_energy(self, static_power: StaticPower | float) -> float:
        """``d = E - s*T``; negative when the static power is above this kernel's average power."""
        return self.energy_joules - self.static_energy(static_power)

    def decompose(self, static_power: StaticPower | float) -> tuple[float, float, float]:
        """``(s, T, d)`` with ``E = s*T + d``."""
        return static_power_watts(static_power, self.platform), self.time_seconds, self.dynamic_energy(static_power)

    def cost(self) -> Cost:
        return Cost(self.time_seconds, self.energy_joules, self.time_seconds)
