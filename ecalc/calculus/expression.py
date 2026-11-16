"""Pipelines as expressions over kernels, and their evaluation.

An ``Expression`` is a tree: ``Kernel`` leaves combined by ``Sequential`` and ``Parallel``
nodes. It says how kernels are arranged, not what they cost; ``evaluate`` folds it into a
``Cost`` by looking every leaf up in a signature source.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Iterator, Mapping, Protocol

from .cost import ZERO, Cost, parallel, sequential
from .errors import MissingKernel, NegativeDynamicEnergy, PlatformMismatch
from .kernel import KernelSpecification
from .signature import Platform, Signature, StaticPower, static_power_watts


class Expression:
    """Base class of the three node types."""


@dataclasses.dataclass(frozen=True)
class Kernel(Expression):
    """One kernel call."""

    specification: KernelSpecification
    name: str = ""


@dataclasses.dataclass(frozen=True)
class Sequential(Expression):
    """Parts run back to back."""

    parts: tuple[Expression, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))


@dataclasses.dataclass(frozen=True)
class Parallel(Expression):
    """Parts overlap, all starting together."""

    parts: tuple[Expression, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A named expression, e.g. one model at one batch size, with free-form attributes."""

    name: str
    expression: Expression
    attributes: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def kernels(expression: Expression) -> Iterator[Kernel]:
    """Every leaf, in order."""
    if isinstance(expression, Kernel):
        yield expression
    elif isinstance(expression, (Sequential, Parallel)):
        for part in expression.parts:
            yield from kernels(part)
    else:
        raise TypeError(f"not an expression: {expression!r}")


def flatten(expression: Expression) -> Expression:
    """Splice a ``Sequential`` directly inside a ``Sequential`` (and ``Parallel`` inside
    ``Parallel``) into its parent. Associativity makes the cost unchanged; group names are lost."""
    if isinstance(expression, Kernel):
        return expression
    node_type = type(expression)
    parts: list[Expression] = []
    for part in expression.parts:
        part = flatten(part)
        if isinstance(part, node_type):
            parts.extend(part.parts)
        else:
            parts.append(part)
    return node_type(tuple(parts), expression.name)


class SignatureSource(Protocol):
    """Anything that can price a kernel: a database of measurements, or a formula."""

    def lookup(self, specification: KernelSpecification, platform: Platform) -> Signature | None: ...


@dataclasses.dataclass(frozen=True)
class LeafResult:
    """One kernel of an evaluated expression. ``signature`` is None when no source could price it."""

    name: str
    specification: KernelSpecification
    signature: Signature | None
    overlapped: bool  # inside a Parallel with more than one part


@dataclasses.dataclass(frozen=True)
class ProvenanceTotal:
    kernel_count: int
    time_seconds: float
    energy_joules: float


@dataclasses.dataclass(frozen=True)
class Estimate:
    """The folded cost of an expression, with every leaf's signature kept for inspection."""

    cost: Cost
    platform: Platform
    leaves: tuple[LeafResult, ...]

    @property
    def missing(self) -> tuple[KernelSpecification, ...]:
        return tuple(leaf.specification for leaf in self.leaves if leaf.signature is None)

    @property
    def is_complete(self) -> bool:
        return not self.missing

    @property
    def time_seconds(self) -> float:
        return self.cost.time_seconds

    def energy(self, static_power: StaticPower | float, *, negative_dynamic: str = "raise") -> float:
        """Energy in joules under ``static_power``.

        ``negative_dynamic`` says what to do when an overlapped kernel's measured energy is below
        ``s * T``: ``"raise"`` (default), ``"warn"`` or ``"allow"``. A cost without overlap does
        not depend on the static power, so the check never fires for it.
        """
        if negative_dynamic not in ("raise", "warn", "allow"):
            raise ValueError(f"negative_dynamic must be 'raise', 'warn' or 'allow', got {negative_dynamic!r}")
        watts = static_power_watts(static_power, self.platform)
        if self.cost.has_overlap and negative_dynamic != "allow":
            for leaf in self.leaves:
                if not leaf.overlapped or leaf.signature is None:
                    continue
                dynamic = leaf.signature.dynamic_energy(watts)
                if dynamic < 0:
                    message = (f"{leaf.specification}: measured energy {leaf.signature.energy_joules!r} J is below "
                               f"s*T = {watts!r} W x {leaf.signature.time_seconds!r} s, so its dynamic energy "
                               f"would be {dynamic!r} J; the static power is too high for this kernel")
                    if negative_dynamic == "raise":
                        raise NegativeDynamicEnergy(message)
                    warnings.warn(message, stacklevel=2)
        return self.cost.energy(watts)

    def dynamic_energy(self, static_power: StaticPower | float) -> float:
        return self.cost.dynamic_energy(static_power_watts(static_power, self.platform))

    def static_energy(self, static_power: StaticPower | float) -> float:
        return self.cost.static_energy(static_power_watts(static_power, self.platform))

    def by_provenance(self) -> dict[str, ProvenanceTotal]:
        """Kernel count, time and measured energy per provenance; unpriced kernels under ``"missing"``."""
        groups: dict[str, list[LeafResult]] = {}
        for leaf in self.leaves:
            groups.setdefault(leaf.signature.provenance if leaf.signature else "missing", []).append(leaf)
        out = {}
        for provenance, members in groups.items():
            priced = [m.signature.cost() for m in members if m.signature is not None]
            total = sequential(*priced)
            out[provenance] = ProvenanceTotal(len(members), total.time_seconds, total.energy_joules)
        return out


def evaluate(expression: Expression, source: SignatureSource, platform: Platform, *,
             on_missing: str = "raise") -> Estimate:
    """Fold ``expression`` into a cost by pricing every kernel from ``source``.

    ``on_missing="raise"`` raises ``MissingKernel`` at the first kernel without a signature;
    ``"report"`` prices it at zero and lists it in ``Estimate.missing``.
    """
    if on_missing not in ("raise", "report"):
        raise ValueError(f"on_missing must be 'raise' or 'report', got {on_missing!r}")
    leaves: list[LeafResult] = []

    def fold(node: Expression, overlapped: bool) -> Cost:
        if isinstance(node, Kernel):
            signature = source.lookup(node.specification, platform)
            leaves.append(LeafResult(node.name, node.specification, signature, overlapped))
            if signature is None:
                if on_missing == "raise":
                    raise MissingKernel(f"no signature for {node.specification} on {platform}")
                return ZERO
            if signature.platform != platform:
                raise PlatformMismatch(f"{node.specification}: signature is for {signature.platform}, not {platform}")
            return signature.cost()
        if isinstance(node, Sequential):
            return sequential(*[fold(part, overlapped) for part in node.parts])
        if isinstance(node, Parallel):
            inside = overlapped or len(node.parts) > 1
            return parallel(*[fold(part, inside) for part in node.parts])
        raise TypeError(f"not an expression: {node!r}")

    cost = fold(expression, False)
    return Estimate(cost, platform, tuple(leaves))
