import json

from ecalc import db


def _kernel_row(key="k1", e=1.0):
    return dict(kernel_key=key, gpu_name="A40", freq_label="default", dtype="fp32", kernel_type="Conv",
                short_desc="Conv[3->16]", signature_json="{}", input_spec_json="{}", energy_j=e, time_s=0.001,
                power_w=e / 0.001, iterations=10, n_repeats=1, energy_std_j=0.0, time_std_s=0.0,
                measured_at="now")


def test_upsert_updates_not_duplicates(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    db.upsert_kernel(conn, _kernel_row(e=1.0))
    db.upsert_kernel(conn, _kernel_row(e=2.0))
    rows = conn.execute("SELECT energy_j FROM kernel_energy").fetchall()
    assert len(rows) == 1 and rows[0][0] == 2.0
    assert db.has_kernel(conn, "k1", "A40", "default", "fp32")
    assert not db.has_kernel(conn, "k1", "A100", "default", "fp32")


def test_join_reports_misses(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    db.upsert_kernel(conn, _kernel_row("k1"))
    comp = [
        dict(model_name="m", dtype="fp32", batch=1, imgsz=640, layer_idx=i, layer_type="Conv", from_json="-1",
             kernel_key=k, short_desc=k, input_spec_json="{}", output_spec_json="[]", n_params=0,
             ultralytics_version="x", captured_at="now")
        for i, k in enumerate(["k1", "k2"])
    ]
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, comp)
    rows = db.model_composition_join(conn, "m", "A40", "default", "fp32", 1, 640)
    assert [r["kernel_key"] for r in rows] == ["k1", "k2"]
    assert rows[0]["energy_j"] == 1.0 and rows[1]["energy_j"] is None
    # replace is idempotent
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, comp)
    assert conn.execute("SELECT COUNT(*) FROM model_kernels").fetchone()[0] == 2
    assert db.models_using_key(conn, "k1") == ["m"]


def test_static_and_export(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    db.upsert_static(conn, dict(gpu_name="A40", freq_label="default", power_w=70.0,
                                n_samples=3, samples_json=json.dumps([1, 2, 3]), measured_at="now"))
    assert db.get_static(conn, "A40", "default")["power_w"] == 70.0
    paths = db.export_csv(conn, tmp_path / "export")
    assert {p.name for p in paths} == {"kernel_energy.csv", "static_power.csv", "model_runs.csv", "model_kernels.csv"}
    text = (tmp_path / "export" / "static_power.csv").read_text()
    assert "active_idle" in text and text.splitlines()[0].startswith("gpu_name,")


_V1_SCHEMA = """
CREATE TABLE model_kernels (
    model_name TEXT NOT NULL, dtype TEXT NOT NULL, batch INTEGER NOT NULL, imgsz INTEGER NOT NULL,
    layer_idx INTEGER NOT NULL,
    layer_type TEXT, from_json TEXT, kernel_key TEXT NOT NULL, short_desc TEXT,
    input_spec_json TEXT, output_spec_json TEXT, n_params INTEGER,
    ultralytics_version TEXT, captured_at TEXT,
    PRIMARY KEY (model_name, dtype, batch, imgsz, layer_idx)
);
CREATE INDEX idx_model_kernels_key ON model_kernels(kernel_key);
"""


def _comp(level, n, model="m"):
    return [
        dict(model_name=model, dtype="fp32", batch=1, imgsz=640, level=level, layer_idx=i, parent_layer_idx=i // 2,
             path="" if level == "module" else f"p{i}", layer_type="Conv", from_json="-1", kernel_key=f"{level}{i}",
             short_desc="d", input_spec_json="{}", output_spec_json="[]", n_params=0, ultralytics_version="x",
             captured_at="now")
        for i in range(n)
    ]


def test_migration_from_v1_schema(tmp_path):
    import sqlite3

    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(p)
    c.executescript(_V1_SCHEMA)
    c.execute("INSERT INTO model_kernels (model_name, dtype, batch, imgsz, layer_idx, layer_type, kernel_key) "
              "VALUES ('m', 'fp32', 1, 640, 7, 'Conv', 'k7')")
    c.commit()
    c.close()
    conn = db.connect(p)
    cols = db.columns(conn, "model_kernels")
    assert "level" in cols and "parent_layer_idx" in cols and "path" in cols
    rows = conn.execute("SELECT level, layer_idx, parent_layer_idx, kernel_key FROM model_kernels").fetchall()
    assert [tuple(r) for r in rows] == [("module", 7, 7, "k7")]
    assert db.PK["model_kernels"] == ("model_name", "dtype", "batch", "imgsz", "level", "direction", "layer_idx")
    assert conn.execute("SELECT direction FROM model_kernels").fetchone()[0] == "forward"
    # idempotent
    conn.close()
    conn = db.connect(p)
    assert conn.execute("SELECT COUNT(*) FROM model_kernels").fetchone()[0] == 1


def test_levels_are_independent(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, _comp("module", 3), level="module")
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, _comp("leaf", 5), level="leaf")
    assert conn.execute("SELECT COUNT(*) FROM model_kernels WHERE level='module'").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM model_kernels WHERE level='leaf'").fetchone()[0] == 5
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, _comp("leaf", 2), level="leaf")
    assert conn.execute("SELECT COUNT(*) FROM model_kernels WHERE level='module'").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM model_kernels WHERE level='leaf'").fetchone()[0] == 2
    j = db.model_composition_join(conn, "m", "A40", "default", "fp32", 1, 640, level="leaf")
    assert [r["layer_idx"] for r in j] == [0, 1] and j[0]["parent_layer_idx"] == 0 and j[1]["path"] == "p1"
    assert db.models_using_key(conn, "leaf1", "leaf") == ["m"] and db.models_using_key(conn, "leaf1", "module") == []
    assert db.list_models(conn, "module") == ["m"]


_BEFORE_DIRECTION_SCHEMA = """
CREATE TABLE model_runs (
    model_name TEXT NOT NULL, gpu_name TEXT NOT NULL, freq_label TEXT NOT NULL, dtype TEXT NOT NULL,
    batch INTEGER NOT NULL, imgsz INTEGER NOT NULL, energy_j REAL, time_s REAL,
    PRIMARY KEY (model_name, gpu_name, freq_label, dtype, batch, imgsz)
);
CREATE TABLE model_kernels (
    model_name TEXT NOT NULL, dtype TEXT NOT NULL, batch INTEGER NOT NULL, imgsz INTEGER NOT NULL,
    level TEXT NOT NULL DEFAULT 'module', layer_idx INTEGER NOT NULL, parent_layer_idx INTEGER, path TEXT,
    layer_type TEXT, from_json TEXT, kernel_key TEXT NOT NULL, short_desc TEXT,
    input_spec_json TEXT, output_spec_json TEXT, n_params INTEGER, ultralytics_version TEXT, captured_at TEXT,
    PRIMARY KEY (model_name, dtype, batch, imgsz, level, layer_idx)
);
CREATE INDEX idx_model_kernels_key ON model_kernels(kernel_key);
"""


def test_migration_adds_direction_to_the_keys(tmp_path):
    import sqlite3

    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(p)
    c.executescript(_BEFORE_DIRECTION_SCHEMA)
    c.execute("INSERT INTO model_runs VALUES ('m', 'A40', 'default', 'fp32', 1, 640, 0.7, 0.006)")
    c.execute("INSERT INTO model_kernels (model_name, dtype, batch, imgsz, level, layer_idx, layer_type, kernel_key) "
              "VALUES ('m', 'fp32', 1, 640, 'module', 0, 'Conv', 'k0')")
    c.commit()
    c.close()
    for _ in range(2):  # idempotent
        conn = db.connect(p)
        assert db.get_model_run(conn, "m", "A40", "default", "fp32", 1, 640)["energy_j"] == 0.7
        assert db.get_model_run(conn, "m", "A40", "default", "fp32", 1, 640, "backward") is None
        assert [r["direction"] for r in db.model_composition_join(conn, "m", "A40", "default", "fp32", 1, 640)] == ["forward"]
        assert "signature_json" in db.columns(conn, "model_kernels")
        conn.close()


def test_directions_are_independent(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, _comp("module", 3))
    backward = [dict(r, kernel_key=f"b{r['layer_idx']}") for r in _comp("module", 4)]
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, backward, direction="backward")
    assert len(db.model_composition_join(conn, "m", "A40", "default", "fp32", 1, 640)) == 3
    rows = db.model_composition_join(conn, "m", "A40", "default", "fp32", 1, 640, direction="backward")
    assert [r["kernel_key"] for r in rows] == ["b0", "b1", "b2", "b3"]
    for direction, energy in (("forward", 1.0), ("backward", 2.5)):
        db.upsert_model_run(conn, dict(model_name="m", gpu_name="A40", freq_label="default", dtype="fp32", batch=1,
                                       imgsz=640, direction=direction, energy_j=energy, time_s=0.01))
    assert db.get_model_run(conn, "m", "A40", "default", "fp32", 1, 640)["energy_j"] == 1.0
    assert db.get_model_run(conn, "m", "A40", "default", "fp32", 1, 640, "backward")["energy_j"] == 2.5


def test_store_as_a_signature_database(tmp_path):
    from ecalc.calculus import KernelSpecification, Platform, evaluate, load

    signature = {"sig_version": 1, "op": {"name": "add", "args": [], "kwargs": {}},
                 "input": {"kind": "list", "items": [{"shape": [1, 4], "dtype": "float32"}] * 2}}
    key = KernelSpecification.from_signature(signature).key
    conn = db.connect(tmp_path / "t.sqlite")
    db.upsert_kernel(conn, dict(_kernel_row(key, e=0.5), signature_json=json.dumps(signature)))
    db.upsert_kernel(conn, _kernel_row("legacy", e=1.0))  # its stored structure does not hash to its key
    db.upsert_static(conn, dict(gpu_name="A40", freq_label="default", power_w=90.0))
    # a row left by an older store under another definition is ignored
    conn.execute("INSERT INTO static_power (gpu_name, freq_label, method, power_w) VALUES ('A40', 'default', 'p8_idle', 23.0)")
    rows = [dict(r, kernel_key=k, signature_json=s) for r, (k, s) in zip(
        _comp("module", 3), [(key, json.dumps(signature)), ("legacy", None), ("unmeasured", None)])]
    db.replace_model_kernels(conn, "m", "fp32", 1, 640, rows)
    db.upsert_model_run(conn, dict(model_name="m", gpu_name="A40", freq_label="default", dtype="fp32", batch=1,
                                   imgsz=640, energy_j=1.6, time_s=0.0021))
    database = db.signature_database(conn)
    platform = Platform("A40")
    assert len(database) == 2 and database.static_power(platform).watts == 90.0
    pipeline = database.pipeline("m/batch_size=1/input_size=640/module/forward")
    estimate = evaluate(pipeline.expression, database, platform, on_missing="report")
    assert [leaf.signature is not None for leaf in estimate.leaves] == [True, True, False]
    assert estimate.leaves[0].specification.is_resolved and not estimate.leaves[1].specification.is_resolved
    assert estimate.cost.energy_joules == 1.5
    assert database.model_run("m", 1, 640, platform).energy_joules == 1.6
    assert load(db.export_signature_database(conn, tmp_path / "export.jsonl")) == database


def test_precision_of_a_stored_composition_is_carried_by_its_pipeline(tmp_path):
    from ecalc.calculus import KernelSpecification, Platform, evaluate, load, pipeline_name

    signature = {"sig_version": 1, "op": {"name": "add", "args": [], "kwargs": {}},
                 "input": {"kind": "list", "items": [{"shape": [1, 4], "dtype": "float32"}] * 2}}
    base = KernelSpecification.from_signature(signature)
    conn = db.connect(tmp_path / "t.sqlite")
    # the second precision is one an older store may hold; it is no longer supported and is left out
    for precision, energy in (("fp32", 0.5), ("fp32_strict", 0.75)):
        db.upsert_kernel(conn, dict(_kernel_row(base.key, e=energy), dtype=precision, signature_json=json.dumps(signature)))
        row = dict(_comp("module", 1)[0], dtype=precision, kernel_key=base.key, signature_json=json.dumps(signature))
        db.replace_model_kernels(conn, "m", precision, 1, 640, [row])
        db.upsert_model_run(conn, dict(model_name="m", gpu_name="A40", freq_label="default", dtype=precision, batch=1,
                                       imgsz=640, energy_j=energy, time_s=0.001))
    database = db.signature_database(conn)
    platform = Platform("A40")
    assert database.platforms() == [platform] and len(database) == 1
    assert [p.name for p in database.pipelines()] == [pipeline_name("m", 1, 640)]
    (pipeline,) = database.pipelines()
    assert pipeline.attributes["precision"] == "fp32"
    (leaf,) = evaluate(pipeline.expression, database, platform).leaves
    assert leaf.specification == base and leaf.specification.precision == "fp32" and leaf.signature.energy_joules == 0.5
    assert [run.energy_joules for run in database.model_runs()] == [0.5]
    assert load(db.export_signature_database(conn, tmp_path / "export.jsonl")) == database
