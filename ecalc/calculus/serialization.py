"""JSON serialization of signatures, pipelines and signature databases.

Three documents, each carrying a ``schema`` string with a major version:

``ecalc.signature/1``
    One signature, with its kernel embedded.

``ecalc.pipeline/1``
    A named expression. ``kernels`` maps each key to its structural signature; expression nodes
    are ``{"kernel": key}``, ``{"sequential": [...]}`` or ``{"parallel": [...]}``, each with an
    optional ``name``.

``ecalc.signature_database/1``
    JSON Lines. The first line is the header; every later line is one record:
    ``kernel`` (key and structural signature, written once per kernel), ``signature``,
    ``static_power``, ``pipeline`` or ``model_run``. Records refer to kernels by key.

Floats are written with ``repr`` precision, so finite values round-trip bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import SchemaVersionError
from .expression import Expression, Kernel, Parallel, Pipeline, Sequential, kernels
from .kernel import FORWARD, KernelSpecification
from .signature import MEASURED, Platform, Signature, StaticPower
from .source import ModelRun, SignatureDatabase

SIGNATURE_SCHEMA = "ecalc.signature/1"
PIPELINE_SCHEMA = "ecalc.pipeline/1"
DATABASE_SCHEMA = "ecalc.signature_database/1"


def _check_schema(document: Mapping, expected: str) -> None:
    found = document.get("schema")
    if found != expected:
        raise SchemaVersionError(f"expected schema {expected!r}, found {found!r}")


def _check_keys(document: Mapping, allowed: set[str], what: str) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"{what}: unknown fields {sorted(unknown)}")


def _dumps(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ---- small records -----------------------------------------------------------------------------

def platform_to_dict(platform: Platform) -> dict:
    return {"gpu_name": platform.gpu_name, "frequency_label": platform.frequency_label,
            "precision": platform.precision}


def platform_from_dict(data: Mapping) -> Platform:
    _check_keys(data, {"gpu_name", "frequency_label", "precision"}, "platform")
    return Platform(data["gpu_name"], data.get("frequency_label", "default"), data.get("precision", "fp32"))


def _kernel_from_entry(key: str, signature: Mapping) -> KernelSpecification:
    specification = KernelSpecification.from_signature(signature)
    if specification.key != key:
        raise ValueError(f"kernel {key}: the structural signature hashes to {specification.key}")
    return specification


def _signature_fields(signature: Signature) -> dict:
    out = {"key": signature.kernel.key, "platform": platform_to_dict(signature.platform),
           "time_seconds": signature.time_seconds, "energy_joules": signature.energy_joules,
           "provenance": signature.provenance}
    if signature.detail:
        out["detail"] = signature.detail
    return out


def _signature_from_fields(data: Mapping, kernel: KernelSpecification) -> Signature:
    return Signature(kernel, platform_from_dict(data["platform"]), float(data["time_seconds"]),
                     float(data["energy_joules"]), data.get("provenance", MEASURED), data.get("detail", ""))


def signature_to_dict(signature: Signature) -> dict:
    return {"schema": SIGNATURE_SCHEMA, "kernel": signature.kernel.signature, **_signature_fields(signature)}


def signature_from_dict(data: Mapping) -> Signature:
    _check_schema(data, SIGNATURE_SCHEMA)
    _check_keys(data, {"schema", "key", "kernel", "platform", "time_seconds", "energy_joules", "provenance",
                       "detail"}, "signature")
    return _signature_from_fields(data, _kernel_from_entry(data["key"], data["kernel"]))


def static_power_to_dict(static_power: StaticPower) -> dict:
    return {"platform": platform_to_dict(static_power.platform), "method": static_power.method,
            "watts": static_power.watts}


def static_power_from_dict(data: Mapping) -> StaticPower:
    return StaticPower(platform_from_dict(data["platform"]), data["method"], float(data["watts"]))


def model_run_to_dict(run: ModelRun) -> dict:
    return {"model": run.model, "batch_size": run.batch_size, "input_size": run.input_size,
            "direction": run.direction, "platform": platform_to_dict(run.platform),
            "time_seconds": run.time_seconds, "energy_joules": run.energy_joules}


def model_run_from_dict(data: Mapping) -> ModelRun:
    return ModelRun(data["model"], int(data["batch_size"]), int(data["input_size"]),
                    platform_from_dict(data["platform"]), float(data["time_seconds"]),
                    float(data["energy_joules"]), data.get("direction", FORWARD))


# ---- expressions and pipelines -----------------------------------------------------------------

def expression_to_dict(expression: Expression) -> dict:
    if isinstance(expression, Kernel):
        out: dict = {"kernel": expression.specification.key}
    elif isinstance(expression, Sequential):
        out = {"sequential": [expression_to_dict(part) for part in expression.parts]}
    elif isinstance(expression, Parallel):
        out = {"parallel": [expression_to_dict(part) for part in expression.parts]}
    else:
        raise TypeError(f"not an expression: {expression!r}")
    if expression.name:
        out["name"] = expression.name
    return out


def expression_from_dict(data: Mapping, kernel_table: Mapping[str, KernelSpecification]) -> Expression:
    """Rebuild an expression; ``kernel_table`` resolves keys to specifications."""
    name = data.get("name", "")
    node_types = [k for k in ("kernel", "sequential", "parallel") if k in data]
    if len(node_types) != 1:
        raise ValueError(f"an expression node needs exactly one of kernel/sequential/parallel, got {sorted(data)}")
    _check_keys(data, {node_types[0], "name"}, "expression node")
    if "kernel" in data:
        key = data["kernel"]
        if key not in kernel_table:
            raise ValueError(f"expression refers to kernel {key}, which is not in the kernel table")
        return Kernel(kernel_table[key], name)
    node_type = Sequential if "sequential" in data else Parallel
    return node_type(tuple(expression_from_dict(part, kernel_table) for part in data[node_types[0]]), name)


def _pipeline_fields(pipeline: Pipeline) -> dict:
    return {"name": pipeline.name, "attributes": dict(pipeline.attributes),
            "expression": expression_to_dict(pipeline.expression)}


def pipeline_to_dict(pipeline: Pipeline) -> dict:
    table = {leaf.specification.key: leaf.specification.signature for leaf in kernels(pipeline.expression)}
    return {"schema": PIPELINE_SCHEMA, "kernels": table, **_pipeline_fields(pipeline)}


def pipeline_from_dict(data: Mapping) -> Pipeline:
    _check_schema(data, PIPELINE_SCHEMA)
    _check_keys(data, {"schema", "name", "attributes", "kernels", "expression"}, "pipeline")
    table = {key: _kernel_from_entry(key, signature) for key, signature in data["kernels"].items()}
    return Pipeline(data["name"], expression_from_dict(data["expression"], table), dict(data.get("attributes", {})))


# ---- signature database ------------------------------------------------------------------------

def database_to_lines(database: SignatureDatabase) -> list[str]:
    """The JSON Lines of a database: header, then kernels, signatures, static power, pipelines, runs."""
    lines = [_dumps({"schema": DATABASE_SCHEMA})]
    table: dict[str, KernelSpecification] = {}
    for signature in database.signatures():
        table.setdefault(signature.kernel.key, signature.kernel)
    for pipeline in database.pipelines():
        for leaf in kernels(pipeline.expression):
            table.setdefault(leaf.specification.key, leaf.specification)
    for key in sorted(table):
        lines.append(_dumps({"record": "kernel", "key": key, "kernel": table[key].signature}))
    for signature in sorted(database.signatures(), key=lambda s: (s.kernel.key, str(s.platform))):
        lines.append(_dumps({"record": "signature", **_signature_fields(signature)}))
    for static_power in sorted(database.static_powers(), key=lambda p: (str(p.platform), p.method)):
        lines.append(_dumps({"record": "static_power", **static_power_to_dict(static_power)}))
    for pipeline in sorted(database.pipelines(), key=lambda p: p.name):
        lines.append(_dumps({"record": "pipeline", **_pipeline_fields(pipeline)}))
    for run in sorted(database.model_runs(), key=lambda r: (r.model, r.batch_size, r.input_size, r.direction,
                                                              str(r.platform))):
        lines.append(_dumps({"record": "model_run", **model_run_to_dict(run)}))
    return lines


def database_from_lines(lines: list[str]) -> SignatureDatabase:
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise SchemaVersionError("empty signature database: the header line is missing")
    _check_schema(rows[0], DATABASE_SCHEMA)
    database = SignatureDatabase()
    table: dict[str, KernelSpecification] = {}
    for number, row in enumerate(rows[1:], start=2):
        row = dict(row)
        record = row.pop("record", None)
        if record == "kernel":
            specification = _kernel_from_entry(row["key"], row["kernel"])
            existing = table.get(row["key"])
            if existing is not None and existing != specification:
                raise ValueError(f"line {number}: key collision on {row['key']}")
            table[row["key"]] = specification
        elif record == "signature":
            if row["key"] not in table:
                raise ValueError(f"line {number}: signature refers to kernel {row['key']} before its kernel record")
            database.add_signature(_signature_from_fields(row, table[row["key"]]))
        elif record == "static_power":
            database.add_static_power(static_power_from_dict(row))
        elif record == "pipeline":
            database.add_pipeline(Pipeline(row["name"], expression_from_dict(row["expression"], table),
                                           dict(row.get("attributes", {}))))
        elif record == "model_run":
            database.add_model_run(model_run_from_dict(row))
        else:
            raise ValueError(f"line {number}: unknown record type {record!r}")
    return database


# ---- files -------------------------------------------------------------------------------------

def dump(document: Signature | Pipeline | SignatureDatabase, path: str | Path) -> Path:
    """Write a signature or pipeline as JSON, or a database as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(document, SignatureDatabase):
        path.write_text("\n".join(database_to_lines(document)) + "\n")
    elif isinstance(document, Signature):
        path.write_text(_dumps(signature_to_dict(document)) + "\n")
    elif isinstance(document, Pipeline):
        path.write_text(_dumps(pipeline_to_dict(document)) + "\n")
    else:
        raise TypeError(f"cannot serialize {type(document).__name__}")
    return path


def load(path: str | Path) -> Signature | Pipeline | SignatureDatabase:
    """Read back whatever ``dump`` wrote; the ``schema`` field decides the type."""
    text = Path(path).read_text()
    first_line = text.split("\n", 1)[0]
    try:
        head = json.loads(first_line)
    except json.JSONDecodeError:
        head = json.loads(text)  # a pretty-printed single document
    schema = head.get("schema") if isinstance(head, dict) else None
    if schema == DATABASE_SCHEMA:
        return database_from_lines(text.splitlines())
    document = json.loads(text)
    if schema == SIGNATURE_SCHEMA:
        return signature_from_dict(document)
    if schema == PIPELINE_SCHEMA:
        return pipeline_from_dict(document)
    raise SchemaVersionError(f"{path}: unknown schema {schema!r}")
