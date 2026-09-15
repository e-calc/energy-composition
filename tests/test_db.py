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
    db.upsert_static(conn, dict(gpu_name="A40", freq_label="default", method="active_idle", power_w=70.0,
                                n_samples=3, samples_json=json.dumps([1, 2, 3]), measured_at="now"))
    assert db.get_static(conn, "A40", "default", "active_idle")["power_w"] == 70.0
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
    assert db.PK["model_kernels"] == ("model_name", "dtype", "batch", "imgsz", "level", "layer_idx")
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
