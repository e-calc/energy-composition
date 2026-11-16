"""The ``ecalc`` command on a small synthetic database built from the structure fixtures.
Needs no torch and no GPU."""

from __future__ import annotations

import json

import pytest
from calculus_helpers import PLATFORM, synthetic_signature

from ecalc.calculus import (ModelRun, Pipeline, SignatureDatabase, StaticPower, dump, evaluate, kernels, load,
                            pipeline_name)
from ecalc.command_line import main


@pytest.fixture()
def database_path(fixture_pipelines, tmp_path):
    """yolo26n (batch 1) and rf_vgg (batch 4), forward, with synthetic costs and a reference run."""
    database = SignatureDatabase()
    database.add_static_power(StaticPower(PLATFORM, "active_idle", 90.0))
    database.add_static_power(StaticPower(PLATFORM, "p8_idle", 23.0))
    for stem, model, size in (("yolo26n_module_forward", "yolo26n", 640), ("rf_vgg_module_forward", "rf_vgg", 1024)):
        source = fixture_pipelines[stem]
        batch_size = source.attributes["batch_size"]
        database.add_pipeline(Pipeline(pipeline_name(model, batch_size, size), source.expression,
                                       dict(source.attributes)))
        for leaf in kernels(source.expression):
            database.add_signature(synthetic_signature(leaf.specification))
        estimate = evaluate(source.expression, database, PLATFORM)
        database.add_model_run(ModelRun(model, batch_size, size, PLATFORM, estimate.time_seconds * 1.25,
                                        estimate.energy(90.0) * 1.25))
    return dump(database, tmp_path / "signature_database.jsonl")


def run(capsys, *arguments) -> tuple[int, str, str]:
    code = main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_estimate_reports_energy_time_and_the_reference(capsys, database_path):
    code, out, _ = run(capsys, "estimate", "--model", "yolo26n", "--database", str(database_path))
    assert code == 0
    assert "yolo26n: batch size 1, input size 640, module level, forward" in out
    assert "measured 24 kernels" in out and "composition error -20.0% energy, -20.0% time" in out


def test_estimate_json_matches_the_library(capsys, database_path):
    code, out, _ = run(capsys, "estimate", "--model", "rf_vgg", "--batch-size", "4", "--database", str(database_path),
                       "--json")
    (result,) = json.loads(out)
    database = load(database_path)
    pipeline = database.pipeline(pipeline_name("rf_vgg", 4, 1024))
    estimate = evaluate(pipeline.expression, database, PLATFORM)
    assert code == 0 and result["kernels"] == 11 and result["pipeline_source"] == "database"
    assert result["time_seconds"] == estimate.time_seconds
    assert result["energy_joules"] == estimate.energy(90.0)
    assert result["static_energy_joules"] + result["dynamic_energy_joules"] == pytest.approx(result["energy_joules"])
    assert result["overlap_saving_joules"] == 0.0


def test_static_power_by_method_or_watts_does_not_change_a_sequential_total(capsys, database_path):
    energies = []
    for static_power in ("active_idle", "p8_idle", "150"):
        _, out, _ = run(capsys, "estimate", "--model", "yolo26n", "--database", str(database_path), "--json",
                        "--static-power", static_power)
        (result,) = json.loads(out)
        energies.append(result["energy_joules"])
    assert len(set(energies)) == 1
    assert result["static_power_watts"] == 150.0 and result["static_power_method"] == "user"


def test_kernel_table(capsys, database_path):
    _, out, _ = run(capsys, "estimate", "--model", "rf_vgg", "--batch-size", "4", "--database", str(database_path),
                    "--kernels")
    assert out.count("measured ") >= 11 and "ConvPool[2->64" in out


def test_compose_overlap_saving_is_static_power_times_the_shorter_time(capsys, database_path, tmp_path):
    names = [pipeline_name("yolo26n", 1, 640), pipeline_name("rf_vgg", 4, 1024)]
    document = {"schema": "ecalc.pipeline/1", "name": "two models sharing one GPU",
                "expression": {"parallel": [{"pipeline": names[0]}, {"pipeline": names[1]}]}}
    path = tmp_path / "two_models.json"
    path.write_text(json.dumps(document))
    code, out, _ = run(capsys, "compose", str(path), "--database", str(database_path), "--json")
    result = json.loads(out)
    database = load(database_path)
    times = [evaluate(database.pipeline(n).expression, database, PLATFORM).time_seconds for n in names]
    assert code == 0 and result["kernels"] == 35
    assert result["time_seconds"] == max(times)
    assert result["overlap_saving_joules"] == pytest.approx(90.0 * min(times), rel=1e-12)
    _, text, _ = run(capsys, "compose", str(path), "--database", str(database_path))
    assert "overlap saves" in text


def test_compose_reports_an_unknown_pipeline(capsys, database_path, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "ecalc.pipeline/1", "name": "bad",
                                "expression": {"pipeline": "no/such/pipeline"}}))
    code, _, err = run(capsys, "compose", str(path), "--database", str(database_path))
    assert code == 2 and "no pipeline named" in err


def test_plan_counts_what_is_measured(capsys, database_path):
    code, out, _ = run(capsys, "plan", "--model", "yolo26n", "--database", str(database_path), "--json")
    (report,) = json.loads(out)
    assert code == 0 and report["unique_kernels"] == 24
    assert report["by_provenance"] == {"measured": 24} and report["to_measure"] == []


def test_export_from_a_measurement_store(capsys, tmp_path):
    from ecalc import db

    store = tmp_path / "store.sqlite"
    conn = db.connect(store)
    db.upsert_kernel(conn, dict(kernel_key="k1", gpu_name="A40", freq_label="default", dtype="fp32",
                                short_desc="legacy row", signature_json="{}", energy_j=1.0, time_s=0.001))
    db.upsert_static(conn, dict(gpu_name="A40", freq_label="default", method="active_idle", power_w=90.0))
    conn.close()
    output = tmp_path / "exported.jsonl"
    code, out, _ = run(capsys, "export", "--store", str(store), "--output", str(output))
    assert code == 0 and "1 signatures" in out and "keep it out of version control" in out
    exported = load(output)
    assert len(exported) == 1 and exported.static_power(PLATFORM, "active_idle").watts == 90.0


def test_helpful_errors(capsys, database_path, tmp_path):
    code, _, err = run(capsys, "estimate", "--model", "yolo26n", "--database", str(tmp_path / "missing.jsonl"))
    assert code == 2 and "no signature database" in err
    code, _, err = run(capsys, "estimate", "--model", "yolo26n", "--database", str(database_path), "--gpu", "H100")
    assert code == 2 and "no signatures for H100" in err
    code, _, err = run(capsys, "estimate", "--model", "yolo26n", "--database", str(database_path),
                       "--static-power", "no_such_method")
    assert code == 2 and "no static power" in err
    code, out, _ = run(capsys, "measure")
    assert code == 2 and "static-power" in out
