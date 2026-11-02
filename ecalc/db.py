"""SQLite kernel-energy database (stdlib sqlite3, WAL, upserts, resumable).

Tables
  kernel_energy  PK (kernel_key, gpu_name, freq_label, dtype)  one isolated Zeus measurement per kernel
  static_power   PK (gpu_name, freq_label, method)             static power definitions
  model_runs     PK (model_name, gpu_name, freq_label, dtype, batch, imgsz)  end-to-end ground truth
  model_kernels  PK (model_name, dtype, batch, imgsz, level, layer_idx)      model composition per level

``kernel_energy`` is level-agnostic: a kernel key identifies a callable on given input shapes,
whatever composition it appears in. ``level`` ("module" | "leaf") lives only in ``model_kernels``;
at leaf level ``layer_idx`` is the execution index and ``parent_layer_idx`` the top-level layer.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS kernel_energy (
    kernel_key TEXT NOT NULL, gpu_name TEXT NOT NULL, freq_label TEXT NOT NULL, dtype TEXT NOT NULL,
    kernel_type TEXT, short_desc TEXT, signature_json TEXT, input_spec_json TEXT,
    energy_j REAL, time_s REAL, power_w REAL, iterations INTEGER, n_repeats INTEGER,
    energy_std_j REAL, time_std_s REAL, gpu_kernel_time_s REAL,
    sm_clock_mhz_mean REAL, sm_clock_mhz_min REAL, sm_clock_mhz_max REAL, mem_clock_mhz_mean REAL,
    pstate_max INTEGER, temp_before REAL, temp_after REAL,
    measurement_duration_s REAL, cooldown_s REAL, num_warmup INTEGER, num_calibration INTEGER,
    warmup_settle_s REAL,
    input_source TEXT, persistence_mode INTEGER,
    torch_version TEXT, ultralytics_version TEXT, zeus_version TEXT, driver_version TEXT,
    tf32_conv INTEGER, tf32_matmul INTEGER, measured_at TEXT, notes TEXT,
    PRIMARY KEY (kernel_key, gpu_name, freq_label, dtype)
);
CREATE TABLE IF NOT EXISTS static_power (
    gpu_name TEXT NOT NULL, freq_label TEXT NOT NULL, method TEXT NOT NULL,
    power_w REAL, n_samples INTEGER, n_rejected INTEGER, samples_json TEXT,
    pstate INTEGER, sm_clock_mhz REAL, mem_clock_mhz REAL, temp_c REAL,
    burst_s REAL, window_s REAL, model_name TEXT, persistence_mode INTEGER,
    driver_version TEXT, measured_at TEXT, notes TEXT,
    PRIMARY KEY (gpu_name, freq_label, method)
);
CREATE TABLE IF NOT EXISTS model_runs (
    model_name TEXT NOT NULL, gpu_name TEXT NOT NULL, freq_label TEXT NOT NULL, dtype TEXT NOT NULL,
    batch INTEGER NOT NULL, imgsz INTEGER NOT NULL,
    energy_j REAL, time_s REAL, power_w REAL, iterations INTEGER, n_repeats INTEGER,
    energy_std_j REAL, time_std_s REAL, gpu_kernel_time_s REAL,
    sm_clock_mhz_mean REAL, sm_clock_mhz_min REAL, sm_clock_mhz_max REAL, mem_clock_mhz_mean REAL,
    pstate_max INTEGER, temp_before REAL, temp_after REAL,
    measurement_duration_s REAL, cooldown_s REAL, num_warmup INTEGER, num_calibration INTEGER,
    warmup_settle_s REAL,
    persistence_mode INTEGER,
    torch_version TEXT, ultralytics_version TEXT, zeus_version TEXT, driver_version TEXT,
    tf32_conv INTEGER, tf32_matmul INTEGER, measured_at TEXT, notes TEXT,
    PRIMARY KEY (model_name, gpu_name, freq_label, dtype, batch, imgsz)
);
CREATE TABLE IF NOT EXISTS model_kernels (
    model_name TEXT NOT NULL, dtype TEXT NOT NULL, batch INTEGER NOT NULL, imgsz INTEGER NOT NULL,
    level TEXT NOT NULL DEFAULT 'module', layer_idx INTEGER NOT NULL,
    parent_layer_idx INTEGER, path TEXT,
    layer_type TEXT, from_json TEXT, kernel_key TEXT NOT NULL, short_desc TEXT,
    input_spec_json TEXT, output_spec_json TEXT, n_params INTEGER,
    ultralytics_version TEXT, captured_at TEXT,
    PRIMARY KEY (model_name, dtype, batch, imgsz, level, layer_idx)
);
CREATE INDEX IF NOT EXISTS idx_model_kernels_key ON model_kernels(kernel_key);
"""

PK = {
    "kernel_energy": ("kernel_key", "gpu_name", "freq_label", "dtype"),
    "static_power": ("gpu_name", "freq_label", "method"),
    "model_runs": ("model_name", "gpu_name", "freq_label", "dtype", "batch", "imgsz"),
    "model_kernels": ("model_name", "dtype", "batch", "imgsz", "level", "layer_idx"),
}


def _migrate_model_kernels(conn: sqlite3.Connection) -> bool:
    """Schema v1 -> v2: add ``level``/``parent_layer_idx``/``path`` and put ``level`` in the PK.
    SQLite cannot alter a primary key in place, so the table is rebuilt in one transaction.
    Existing rows become level='module' with parent_layer_idx = layer_idx. Idempotent."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(model_kernels)")]
    if not cols or "level" in cols:
        return False
    conn.executescript("""
    BEGIN;
    ALTER TABLE model_kernels RENAME TO model_kernels_v1;
    DROP INDEX IF EXISTS idx_model_kernels_key;
    CREATE TABLE model_kernels (
        model_name TEXT NOT NULL, dtype TEXT NOT NULL, batch INTEGER NOT NULL, imgsz INTEGER NOT NULL,
        level TEXT NOT NULL DEFAULT 'module', layer_idx INTEGER NOT NULL,
        parent_layer_idx INTEGER, path TEXT,
        layer_type TEXT, from_json TEXT, kernel_key TEXT NOT NULL, short_desc TEXT,
        input_spec_json TEXT, output_spec_json TEXT, n_params INTEGER,
        ultralytics_version TEXT, captured_at TEXT,
        PRIMARY KEY (model_name, dtype, batch, imgsz, level, layer_idx)
    );
    INSERT INTO model_kernels (model_name, dtype, batch, imgsz, level, layer_idx, parent_layer_idx, path,
        layer_type, from_json, kernel_key, short_desc, input_spec_json, output_spec_json, n_params,
        ultralytics_version, captured_at)
    SELECT model_name, dtype, batch, imgsz, 'module', layer_idx, layer_idx, '',
        layer_type, from_json, kernel_key, short_desc, input_spec_json, output_spec_json, n_params,
        ultralytics_version, captured_at FROM model_kernels_v1;
    DROP TABLE model_kernels_v1;
    CREATE INDEX IF NOT EXISTS idx_model_kernels_key ON model_kernels(kernel_key);
    COMMIT;
    """)
    return True


# Columns added after the first schema: (table, column, type). NULL in old rows.
ADDED_COLUMNS = (("kernel_energy", "warmup_settle_s", "REAL"), ("model_runs", "warmup_settle_s", "REAL"))


def _add_columns(conn: sqlite3.Connection) -> None:
    for table, col, typ in ADDED_COLUMNS:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if cols and col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    conn.commit()


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _migrate_model_kernels(conn)
    conn.executescript(SCHEMA)
    _add_columns(conn)
    return conn


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _upsert(conn: sqlite3.Connection, table: str, row: dict) -> None:
    cols = [c for c in columns(conn, table) if c in row]
    missing = [k for k in PK[table] if k not in row]
    if missing:
        raise ValueError(f"{table}: missing primary key columns {missing}")
    pk = PK[table]
    upd = [c for c in cols if c not in pk]
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
        f"ON CONFLICT({', '.join(pk)}) DO UPDATE SET " + ", ".join(f"{c}=excluded.{c}" for c in upd)
    )
    conn.execute(sql, [row[c] for c in cols])


def upsert_kernel(conn, row: dict, commit: bool = True) -> None:
    _upsert(conn, "kernel_energy", row)
    if commit:
        conn.commit()


def has_kernel(conn, kernel_key: str, gpu_name: str, freq_label: str, dtype: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM kernel_energy WHERE kernel_key=? AND gpu_name=? AND freq_label=? AND dtype=?",
        (kernel_key, gpu_name, freq_label, dtype),
    ).fetchone()
    return r is not None


def get_kernel(conn, kernel_key: str, gpu_name: str, freq_label: str, dtype: str) -> dict | None:
    r = conn.execute(
        "SELECT * FROM kernel_energy WHERE kernel_key=? AND gpu_name=? AND freq_label=? AND dtype=?",
        (kernel_key, gpu_name, freq_label, dtype),
    ).fetchone()
    return dict(r) if r else None


def set_kernel_gpu_time(conn, kernel_key: str, gpu_name: str, freq_label: str, dtype: str, t: float | None) -> None:
    conn.execute(
        "UPDATE kernel_energy SET gpu_kernel_time_s=? WHERE kernel_key=? AND gpu_name=? AND freq_label=? AND dtype=?",
        (t, kernel_key, gpu_name, freq_label, dtype))
    conn.commit()


def set_model_run_gpu_time(conn, model_name, gpu_name, freq_label, dtype, batch, imgsz, t: float | None) -> None:
    conn.execute(
        "UPDATE model_runs SET gpu_kernel_time_s=? WHERE model_name=? AND gpu_name=? AND freq_label=? AND dtype=? "
        "AND batch=? AND imgsz=?", (t, model_name, gpu_name, freq_label, dtype, batch, imgsz))
    conn.commit()


def upsert_static(conn, row: dict, commit: bool = True) -> None:
    _upsert(conn, "static_power", row)
    if commit:
        conn.commit()


def get_static(conn, gpu_name: str, freq_label: str, method: str) -> dict | None:
    r = conn.execute(
        "SELECT * FROM static_power WHERE gpu_name=? AND freq_label=? AND method=?",
        (gpu_name, freq_label, method),
    ).fetchone()
    return dict(r) if r else None


def list_static(conn, gpu_name: str, freq_label: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM static_power WHERE gpu_name=? AND freq_label=? ORDER BY method", (gpu_name, freq_label))]


def upsert_model_run(conn, row: dict, commit: bool = True) -> None:
    _upsert(conn, "model_runs", row)
    if commit:
        conn.commit()


def get_model_run(conn, model_name, gpu_name, freq_label, dtype, batch, imgsz) -> dict | None:
    r = conn.execute(
        "SELECT * FROM model_runs WHERE model_name=? AND gpu_name=? AND freq_label=? AND dtype=? AND batch=? AND imgsz=?",
        (model_name, gpu_name, freq_label, dtype, batch, imgsz),
    ).fetchone()
    return dict(r) if r else None


def replace_model_kernels(conn, model_name: str, dtype: str, batch: int, imgsz: int, rows: list[dict],
                          level: str = "module") -> None:
    """Replace the composition of one model at one level; other levels are untouched."""
    conn.execute(
        "DELETE FROM model_kernels WHERE model_name=? AND dtype=? AND batch=? AND imgsz=? AND level=?",
        (model_name, dtype, batch, imgsz, level),
    )
    for r in rows:
        r = dict(r)
        r.setdefault("level", level)
        if r["level"] != level:
            raise ValueError(f"row level {r['level']!r} != {level!r}")
        _upsert(conn, "model_kernels", r)
    conn.commit()


def model_composition_join(conn, model_name: str, gpu_name: str, freq_label: str, dtype: str,
                           batch: int, imgsz: int, level: str = "module") -> list[dict]:
    """Composition rows LEFT JOINed with kernel_energy; misses have energy_j NULL."""
    rows = conn.execute(
        """
        SELECT mk.level, mk.layer_idx, mk.parent_layer_idx, mk.path, mk.layer_type, mk.from_json, mk.kernel_key,
               mk.short_desc, mk.n_params,
               ke.energy_j, ke.time_s, ke.power_w, ke.energy_std_j, ke.time_std_s, ke.iterations, ke.n_repeats,
               ke.gpu_kernel_time_s, ke.sm_clock_mhz_mean, ke.input_source,
               ke.measurement_duration_s, ke.cooldown_s, ke.num_warmup, ke.num_calibration, ke.warmup_settle_s
        FROM model_kernels mk
        LEFT JOIN kernel_energy ke
          ON ke.kernel_key = mk.kernel_key AND ke.gpu_name = ? AND ke.freq_label = ? AND ke.dtype = ?
        WHERE mk.model_name = ? AND mk.dtype = ? AND mk.batch = ? AND mk.imgsz = ? AND mk.level = ?
        ORDER BY mk.layer_idx
        """,
        (gpu_name, freq_label, dtype, model_name, dtype, batch, imgsz, level),
    ).fetchall()
    return [dict(r) for r in rows]


def models_using_key(conn, kernel_key: str, level: str | None = None) -> list[str]:
    if level is None:
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT model_name FROM model_kernels WHERE kernel_key=? ORDER BY model_name", (kernel_key,))]
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT model_name FROM model_kernels WHERE kernel_key=? AND level=? ORDER BY model_name",
        (kernel_key, level))]


def list_models(conn, level: str | None = None) -> list[str]:
    if level is None:
        return [r[0] for r in conn.execute("SELECT DISTINCT model_name FROM model_kernels ORDER BY model_name")]
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT model_name FROM model_kernels WHERE level=? ORDER BY model_name", (level,))]


def export_csv(conn, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for table in PK:
        cols = columns(conn, table)
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        p = out_dir / f"{table}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
        paths.append(p)
    return paths
