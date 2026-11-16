"""Where signatures come from: an in-memory database, and chains of sources."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

from .expression import Pipeline, SignatureSource
from .kernel import DEFAULT_PRECISION, FORWARD, PRECISIONS, KernelSpecification
from .signature import Platform, Signature, StaticPower


@dataclasses.dataclass(frozen=True)
class ModelRun:
    """An end-to-end measurement of a whole model: the reference a composition is compared with."""

    model: str
    batch_size: int
    input_size: int
    platform: Platform
    time_seconds: float
    energy_joules: float
    direction: str = FORWARD
    precision: str = DEFAULT_PRECISION

    @property
    def identity(self) -> tuple:
        return (self.model, self.batch_size, self.input_size, self.direction, self.precision, self.platform)


def pipeline_name(model: str, batch_size: int, input_size: int, level: str = "module",
                  direction: str = FORWARD, precision: str = DEFAULT_PRECISION) -> str:
    """Name under which a model's pipeline is stored in a signature database. The precision is
    part of the name only when it is not the default."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; known: {PRECISIONS}")
    name = f"{model}/batch_size={batch_size}/input_size={input_size}/{level}/{direction}"
    return name if precision == DEFAULT_PRECISION else f"{name}/{precision}"


class SignatureDatabase:
    """Signatures, static power, model pipelines and end-to-end runs, in memory.

    It is a ``SignatureSource`` that answers exact matches only; every hit is a stored signature.
    """

    def __init__(self) -> None:
        self._signatures: dict[tuple[str, Platform], Signature] = {}
        self._static_powers: dict[Platform, StaticPower] = {}
        self._pipelines: dict[str, Pipeline] = {}
        self._model_runs: dict[tuple, ModelRun] = {}

    # ---- writing -------------------------------------------------------------------------
    def add_signature(self, signature: Signature) -> None:
        index = (signature.kernel.key, signature.platform)
        existing = self._signatures.get(index)
        if existing is not None and existing.kernel != signature.kernel:
            raise ValueError(f"key collision on {signature.kernel.key}: two different kernels hash to the same key")
        self._signatures[index] = signature

    def add_static_power(self, static_power: StaticPower) -> None:
        self._static_powers[static_power.platform] = static_power

    def add_pipeline(self, pipeline: Pipeline) -> None:
        self._pipelines[pipeline.name] = pipeline

    def add_model_run(self, run: ModelRun) -> None:
        self._model_runs[run.identity] = run

    # ---- reading -------------------------------------------------------------------------
    def lookup(self, specification: KernelSpecification, platform: Platform) -> Signature | None:
        return self._signatures.get((specification.key, platform))

    def signatures(self, platform: Platform | None = None) -> list[Signature]:
        return [s for s in self._signatures.values() if platform is None or s.platform == platform]

    def platforms(self) -> list[Platform]:
        return sorted({s.platform for s in self._signatures.values()}, key=str)

    def static_power(self, platform: Platform) -> StaticPower:
        try:
            return self._static_powers[platform]
        except KeyError:
            raise LookupError(f"no static power for {platform}") from None

    def static_powers(self) -> list[StaticPower]:
        return list(self._static_powers.values())

    def pipeline(self, name: str) -> Pipeline | None:
        return self._pipelines.get(name)

    def pipelines(self) -> list[Pipeline]:
        return list(self._pipelines.values())

    def model_run(self, model: str, batch_size: int, input_size: int, platform: Platform,
                  direction: str = FORWARD, precision: str = DEFAULT_PRECISION) -> ModelRun | None:
        return self._model_runs.get((model, batch_size, input_size, direction, precision, platform))

    def model_runs(self) -> list[ModelRun]:
        return list(self._model_runs.values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SignatureDatabase):
            return NotImplemented
        return (self._signatures == other._signatures and self._static_powers == other._static_powers
                and self._pipelines == other._pipelines and self._model_runs == other._model_runs)

    def __len__(self) -> int:
        return len(self._signatures)


class SourceChain:
    """Ask each source in turn; the first one with an answer wins. Put the measured database
    first and formulas after it, so a formula is used only where nothing was measured."""

    def __init__(self, sources: Sequence[SignatureSource] | Iterable[SignatureSource]) -> None:
        self.sources = tuple(sources)

    def lookup(self, specification: KernelSpecification, platform: Platform) -> Signature | None:
        for source in self.sources:
            signature = source.lookup(specification, platform)
            if signature is not None:
                return signature
        return None
