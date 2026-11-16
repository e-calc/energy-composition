"""The two composition operators of the energy calculus.

Every kernel has the signature ``E = s*T + d``: static power ``s`` times duration ``T`` plus
dynamic energy ``d``. Composition keeps that form:

    sequential(a, b, ...)   T = sum(T_i)   d = sum(d_i)     back to back
    parallel(a, b, ...)     T = max(T_i)   d = sum(d_i)     overlap: static power is paid once,
                                                            over the longest duration

A ``Cost`` does not contain ``s``. It stores what was measured -- the wall time, the sum of the
member energies, and the sum of the member durations (``busy_seconds``) -- and static power is
supplied only when energy is asked for:

    E = energy_joules + s * (time_seconds - busy_seconds)        (== s*T + d)

For a sequential cost ``time_seconds == busy_seconds``, so its energy is the plain sum of the
measured energies whatever ``s`` is. Overlap makes ``time_seconds < busy_seconds`` and saves
``s * (busy_seconds - time_seconds)``; for two kernels that is ``s * min(T_a, T_b)``.

Units: seconds, joules, watts. Sums use ``math.fsum``, so results do not depend on argument order.
"""

from __future__ import annotations

import dataclasses
import math


def _checked(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class Cost:
    """Time and energy of a kernel or of any composition of kernels.

    ``time_seconds``   wall time of the composition.
    ``energy_joules``  sum of the member kernels' energies, as measured.
    ``busy_seconds``   sum of the member kernels' durations: how long they would take back to
                       back. Equal to ``time_seconds`` for one kernel and for anything sequential
                       (the default); larger under overlap.
    """

    time_seconds: float
    energy_joules: float
    busy_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.busy_seconds is None:
            object.__setattr__(self, "busy_seconds", self.time_seconds)
        for field in ("time_seconds", "energy_joules", "busy_seconds"):
            object.__setattr__(self, field, _checked(field, getattr(self, field)))

    def static_energy(self, static_power_watts: float) -> float:
        """``s * T``."""
        return _checked("static_power_watts", static_power_watts) * self.time_seconds

    def dynamic_energy(self, static_power_watts: float) -> float:
        """``d``: the sum of the members' dynamic energies. May be negative when the static power
        is above a member's average power."""
        return self.energy_joules - _checked("static_power_watts", static_power_watts) * self.busy_seconds

    def energy(self, static_power_watts: float) -> float:
        """``E = s*T + d``, computed without the cancellation that form would cause."""
        return self.energy_joules + _checked("static_power_watts", static_power_watts) * (
            self.time_seconds - self.busy_seconds)

    def overlap_saving(self, static_power_watts: float) -> float:
        """Energy saved against running every member back to back: ``s * (sum(T_i) - T)``."""
        return _checked("static_power_watts", static_power_watts) * (self.busy_seconds - self.time_seconds)

    @property
    def has_overlap(self) -> bool:
        """True when the energy depends on the static power."""
        return self.time_seconds != self.busy_seconds


ZERO = Cost(0.0, 0.0, 0.0)
"""The identity of both operators: nothing runs."""


def sequential(*costs: Cost) -> Cost:
    """Costs run back to back: times add, energies add."""
    return Cost(
        math.fsum(c.time_seconds for c in costs),
        math.fsum(c.energy_joules for c in costs),
        math.fsum(c.busy_seconds for c in costs),
    )


def parallel(*costs: Cost) -> Cost:
    """Costs overlap, all starting together: the duration is the longest one, static power is
    paid once over it, dynamic energies add."""
    return Cost(
        max((c.time_seconds for c in costs), default=0.0),
        math.fsum(c.energy_joules for c in costs),
        math.fsum(c.busy_seconds for c in costs),
    )
