"""Serialization of signatures, pipelines and signature databases: round trips, stable keys,
and commuting with evaluation. Needs no torch."""

from __future__ import annotations

import json
import random

import pytest
from calculus_helpers import PLATFORM, operation_specification, random_expression, synthetic_database

from ecalc.calculus import (Kernel, KernelSpecification, ModelRun, Parallel, Pipeline, Platform, SchemaVersionError,
                            Sequential, Signature, SignatureDatabase, StaticPower, TensorSpecification, dump, evaluate,
                            kernel_key, kernels, load)
from ecalc.calculus.serialization import (DATABASE_SCHEMA, database_from_lines, database_to_lines,
                                          expression_from_dict, expression_to_dict, pipeline_to_dict,
                                          signature_to_dict)


def _specifications(count: int = 8):
    return [operation_specification("cat", shape=(2, 8, 16, 16 + i), dim=1) for i in range(count)]


def _database() -> SignatureDatabase:
    specifications = _specifications()
    database = synthetic_database(specifications)
    database.add_signature(Signature(specifications[0], Platform("A100", "1500"), 0.1234567891, 12.5,
                                     "interpolated", "chord over batch size between 1 and 8"))
    database.add_static_power(StaticPower(PLATFORM, 89.69807891254976))
    rng = random.Random(0)
    for name in ("first", "second"):
        database.add_pipeline(Pipeline(name, random_expression(rng, specifications), {"model": name, "batch_size": 2}))
    database.add_model_run(ModelRun("first", 2, 640, PLATFORM, 0.006055, 0.7646))
    database.add_model_run(ModelRun("first", 2, 640, PLATFORM, 0.011, 1.65, "backward"))
    return database


# ---- round trips ------------------------------------------------------------------------------

def test_signature_round_trip(tmp_path):
    for signature in _database().signatures():
        assert load(dump(signature, tmp_path / "signature.json")) == signature


def test_pipeline_round_trip(tmp_path):
    for pipeline in _database().pipelines():
        again = load(dump(pipeline, tmp_path / "pipeline.json"))
        assert again == pipeline
        assert [leaf.specification.key for leaf in kernels(again.expression)] == [
            leaf.specification.key for leaf in kernels(pipeline.expression)]


def test_database_round_trip(tmp_path):
    database = _database()
    path = dump(database, tmp_path / "database.jsonl")
    again = load(path)
    assert again == database
    assert path.read_text() == "\n".join(database_to_lines(again)) + "\n"  # canonical: dumping again is identical
    assert json.loads(path.read_text().splitlines()[0]) == {"schema": DATABASE_SCHEMA}


def test_model_fixtures_round_trip(fixture_pipelines, tmp_path):
    for name, pipeline in fixture_pipelines.items():
        assert load(dump(pipeline, tmp_path / f"{name}.json")) == pipeline


def test_floats_round_trip_bit_for_bit(tmp_path):
    rng = random.Random(1)
    for _ in range(200):
        signature = Signature(operation_specification("add"), PLATFORM, rng.uniform(1e-7, 1e-1), rng.uniform(1e-6, 30.0))
        again = load(dump(signature, tmp_path / "signature.json"))
        assert (again.time_seconds, again.energy_joules) == (signature.time_seconds, signature.energy_joules)


# ---- serialization commutes with composition and evaluation ----------------------------------

def test_evaluating_a_reloaded_pipeline_on_a_reloaded_database_gives_the_same_cost(tmp_path):
    database = _database()
    reloaded = load(dump(database, tmp_path / "database.jsonl"))
    for pipeline in database.pipelines():
        again = load(dump(pipeline, tmp_path / "pipeline.json"))
        before = evaluate(pipeline.expression, database, PLATFORM)
        after = evaluate(again.expression, reloaded, PLATFORM)
        assert after.cost == before.cost and after.energy(90) == before.energy(90)


def test_serialization_commutes_with_the_operators():
    specifications = _specifications()
    table = {s.key: s for s in specifications}
    rng = random.Random(2)
    for node_type in (Sequential, Parallel):
        parts = tuple(random_expression(rng, specifications, depth=2) for _ in range(3))
        whole = expression_to_dict(node_type(parts))
        piecewise = {node_type.__name__.lower(): [expression_to_dict(part) for part in parts]}
        assert whole == piecewise
        assert expression_from_dict(whole, table) == node_type(parts)


# ---- keys -------------------------------------------------------------------------------------

def _convolution(shape=(1, 3, 64, 64), dtype="float32", **extra) -> dict:
    tensor = {"shape": list(shape), "dtype": dtype, **extra}
    return {"sig_version": 1, "input": {"kind": "tensor", **tensor},
            "module": {"cls": "Conv", "buffers": [],
                       "modules": [{"n": "", "cls": "Conv", "a": {}},
                                   {"n": "conv", "cls": "Conv2d", "a": {"in_channels": 3, "out_channels": 16,
                                                                        "kernel_size": [3, 3], "stride": [2, 2],
                                                                        "padding": [1, 1], "dilation": [1, 1],
                                                                        "groups": 1}}],
                       "params": [{"n": "conv.weight", "s": [16, 3, 3, 3], "dt": "float32"}]}}


def test_key_does_not_depend_on_field_order():
    signature = _convolution()
    reordered = json.loads(json.dumps(signature, sort_keys=True))
    reordered = dict(reversed(list(reordered.items())))
    assert kernel_key(reordered) == kernel_key(signature)
    assert KernelSpecification.from_signature(reordered) == KernelSpecification.from_signature(signature)


def test_key_changes_with_shape_dtype_layout_and_direction():
    base = KernelSpecification.from_signature(_convolution())
    variants = [
        KernelSpecification.from_signature(_convolution(shape=(2, 3, 64, 64))),
        KernelSpecification.from_signature(_convolution(shape=(1, 3, 32, 32))),
        KernelSpecification.from_signature(_convolution(dtype="float16")),
        KernelSpecification.from_signature(_convolution(layout="channels_last")),
        base.with_direction("backward"),
        base.with_direction("backward", input_gradient=False),
    ]
    keys = {base.key} | {v.key for v in variants}
    assert len(keys) == len(variants) + 1


def test_default_layout_direction_and_precision_are_not_written():
    base = KernelSpecification.from_signature(_convolution())
    assert TensorSpecification((1, 3, 64, 64)).as_dict() == {"shape": [1, 3, 64, 64], "dtype": "float32"}
    assert "layout" not in base.canonical and "direction" not in base.canonical
    assert base.with_direction("backward").with_direction("forward") == base
    assert "precision" not in base.canonical and base.precision == "fp32"
    assert base.with_precision("fp32") == base
    assert base.inputs == (TensorSpecification((1, 3, 64, 64)),)
    assert KernelSpecification.from_signature(_convolution(layout="channels_last")).inputs[0].layout == "channels_last"


def test_fp32_is_the_only_precision_and_another_one_would_get_its_own_keys(monkeypatch):
    from ecalc.calculus import PRECISIONS, pipeline_name
    from ecalc.calculus import kernel as kernel_module, source as source_module

    base = KernelSpecification.from_signature(_convolution())
    unresolved = KernelSpecification.from_key("0123456789abcdef", "a kernel known only by key")
    assert PRECISIONS == ("fp32",)
    for unsupported in ("fp32_strict", "fp16"):
        with pytest.raises(ValueError, match="unknown precision"):
            base.with_precision(unsupported)
        with pytest.raises(ValueError, match="unknown precision"):
            pipeline_name("m", 1, 640, precision=unsupported)
    # the slot is kept: once a precision is supported it is part of the kernel and of the pipeline name
    monkeypatch.setattr(kernel_module, "PRECISIONS", ("fp32", "fp16"))
    monkeypatch.setattr(source_module, "PRECISIONS", ("fp32", "fp16"))
    half = base.with_precision("fp16")
    assert half.precision == "fp16" and half.key != base.key and half.with_precision("fp32") == base
    assert half.with_direction("backward").precision == "fp16"
    assert half.description == "Conv[3->16,k3,s2]@1x3x64x64 <fp16>"
    assert unresolved.with_precision("fp16").key != unresolved.key
    assert pipeline_name("m", 1, 640) == "m/batch_size=1/input_size=640/module/forward"
    assert pipeline_name("m", 1, 640, precision="fp16") == "m/batch_size=1/input_size=640/module/forward/fp16"


def test_descriptions():
    base = KernelSpecification.from_signature(_convolution())
    assert base.description == "Conv[3->16,k3,s2]@1x3x64x64"
    assert base.with_direction("backward", input_gradient=False).description.endswith("<backward, no input gradient>")
    assert operation_specification("cat", shape=(1, 8, 4, 4), dim=1).description == "op:cat[dim=1]@1x8x4x4"
    unresolved = KernelSpecification.from_key("0123456789abcdef", "a kernel known only by key")
    assert unresolved.key == "0123456789abcdef" and not unresolved.is_resolved
    assert unresolved.description == "a kernel known only by key"


def test_a_kernel_known_only_by_its_key_round_trips(tmp_path):
    unresolved = KernelSpecification.from_key("0123456789abcdef", "legacy row")
    database = SignatureDatabase()
    database.add_signature(Signature(unresolved, PLATFORM, 0.5, 100.0))
    database.add_pipeline(Pipeline("legacy", Sequential((Kernel(unresolved),))))
    assert load(dump(database, tmp_path / "database.jsonl")) == database


# ---- errors -----------------------------------------------------------------------------------

def test_unknown_schema_is_rejected(tmp_path):
    signature = _database().signatures()[0]
    document = signature_to_dict(signature)
    document["schema"] = "ecalc.signature/2"
    path = tmp_path / "future.json"
    path.write_text(json.dumps(document))
    with pytest.raises(SchemaVersionError):
        load(path)
    with pytest.raises(SchemaVersionError):
        database_from_lines(['{"schema": "something.else/1"}'])


def test_a_tampered_kernel_is_rejected(tmp_path):
    pipeline = _database().pipelines()[0]
    document = pipeline_to_dict(pipeline)
    key = next(iter(document["kernels"]))
    document["kernels"][key]["input"]["items"][0]["shape"][0] += 1  # no longer hashes to its key
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="hashes to"):
        load(path)


def test_unknown_fields_and_dangling_references_are_rejected():
    table = {s.key: s for s in _specifications(1)}
    with pytest.raises(ValueError, match="not in the kernel table"):
        expression_from_dict({"kernel": "ffffffffffffffff"}, table)
    with pytest.raises(ValueError, match="unknown fields"):
        expression_from_dict({"sequential": [], "repeat": 3}, table)
    with pytest.raises(ValueError, match="exactly one"):
        expression_from_dict({"name": "empty"}, table)
    lines = database_to_lines(_database())
    signature_line = next(i for i, line in enumerate(lines) if '"record":"signature"' in line)
    with pytest.raises(ValueError, match="before its kernel record"):
        database_from_lines([lines[0], lines[signature_line]])
    # a static power record written with a measurement method, as older exports did
    with pytest.raises(ValueError, match="unknown fields"):
        database_from_lines([lines[0], json.dumps({
            "record": "static_power", "method": "p8_idle", "watts": 23.4,
            "platform": {"gpu_name": "A40", "frequency_label": "default"}})])
    # precision is part of the kernel, not of the platform
    with pytest.raises(ValueError, match="unknown fields"):
        database_from_lines([lines[0], json.dumps({
            "record": "static_power", "watts": 90.0,
            "platform": {"gpu_name": "A40", "frequency_label": "default", "precision": "fp32"}})])


def test_a_key_collision_is_refused():
    first, second = _specifications(2)
    database = SignatureDatabase()
    database.add_signature(Signature(first, PLATFORM, 0.5, 100.0))
    impostor = KernelSpecification.from_key(first.key, "different structure, same key")
    with pytest.raises(ValueError, match="collision"):
        database.add_signature(Signature(impostor, PLATFORM, 0.5, 100.0))
    database.add_signature(Signature(second, PLATFORM, 0.5, 100.0))
    assert len(database) == 2
