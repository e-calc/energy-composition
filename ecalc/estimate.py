"""Compose a model's energy from the kernel DB and validate against the end-to-end run.

    E_dyn(k)      = e_k - P * t_k
    E_pred_sum    = sum_k E_dyn(k) + P * T_sum      (== sum_k e_k, independent of P)
    E_pred_hybrid = sum_k E_dyn(k) + P * T_e2e      (depends on the static-power definition)
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from pathlib import Path

from . import db


@dataclasses.dataclass
class ModelEstimate:
    model_name: str
    gpu_name: str
    freq_label: str
    dtype: str
    static_method: str
    static_w: float
    layers: list[dict]
    hits: int
    misses: int
    T_sum: float
    E_sum: float
    E_dyn_sum: float
    E_pred_sum: float
    E_pred_hybrid: float | None
    E_e2e: float | None
    T_e2e: float | None
    e2e_row: dict | None
    level: str = "module"
    n_unique_keys: int = 0

    @property
    def complete(self) -> bool:
        return self.misses == 0

    def rel_err(self, pred: float | None, ref: float | None) -> float | None:
        if pred is None or ref is None or ref == 0:
            return None
        return (pred - ref) / ref

    @property
    def energy_err_sum(self):
        return self.rel_err(self.E_pred_sum, self.E_e2e)

    @property
    def energy_err_hybrid(self):
        return self.rel_err(self.E_pred_hybrid, self.E_e2e)

    @property
    def time_err(self):
        return self.rel_err(self.T_sum, self.T_e2e)


def resolve_static_power(conn, gpu_name: str, freq_label: str, method: str, static_w: float | None = None) -> float:
    if method == "user":
        if static_w is None:
            raise ValueError("--static-w required with --static-method user")
        return float(static_w)
    row = db.get_static(conn, gpu_name, freq_label, method)
    if row is None:
        raise LookupError(f"no static_power row for ({gpu_name}, {freq_label}, {method}); run scripts/measure_static.py")
    return float(row["power_w"])


def estimate_model(conn, model_name: str, gpu_name: str, freq_label: str, dtype: str, batch: int, imgsz: int,
                   static_method: str = "active_idle", static_w: float | None = None,
                   level: str = "module") -> ModelEstimate:
    P = resolve_static_power(conn, gpu_name, freq_label, static_method, static_w)
    rows = db.model_composition_join(conn, model_name, gpu_name, freq_label, dtype, batch, imgsz, level=level)
    if not rows:
        raise LookupError(f"no {level}-level composition for {model_name}; run scripts/build_kernel_db.py --level {level}")
    layers = []
    for r in rows:
        r = dict(r)
        hit = r["energy_j"] is not None
        r["hit"] = hit
        r["dyn_energy_j"] = (r["energy_j"] - P * r["time_s"]) if hit else None
        r["static_energy_j"] = (P * r["time_s"]) if hit else None
        r["shared_with"] = [m for m in db.models_using_key(conn, r["kernel_key"], level) if m != model_name]
        layers.append(r)
    hits = sum(1 for r in layers if r["hit"])
    T_sum = sum(r["time_s"] for r in layers if r["hit"])
    E_sum = sum(r["energy_j"] for r in layers if r["hit"])
    E_dyn = sum(r["dyn_energy_j"] for r in layers if r["hit"])
    e2e = db.get_model_run(conn, model_name, gpu_name, freq_label, dtype, batch, imgsz)
    E_e2e = e2e["energy_j"] if e2e else None
    T_e2e = e2e["time_s"] if e2e else None
    for r in layers:
        r["energy_share"] = (r["energy_j"] / E_sum) if (r["hit"] and E_sum) else None
    return ModelEstimate(
        model_name=model_name, gpu_name=gpu_name, freq_label=freq_label, dtype=dtype,
        static_method=static_method, static_w=P, layers=layers, hits=hits, misses=len(layers) - hits,
        T_sum=T_sum, E_sum=E_sum, E_dyn_sum=E_dyn, E_pred_sum=E_dyn + P * T_sum,
        E_pred_hybrid=(E_dyn + P * T_e2e) if T_e2e is not None else None,
        E_e2e=E_e2e, T_e2e=T_e2e, e2e_row=e2e, level=level, n_unique_keys=len({r["kernel_key"] for r in layers}),
    )


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:+.1f}%"


def _mj(v) -> str:
    return "n/a" if v is None else f"{1e3 * v:.3f}"


def _us(v) -> str:
    return "n/a" if v is None else f"{1e6 * v:.1f}"


def summary_lines(est: ModelEstimate) -> list[str]:
    return [
        f"model={est.model_name} level={est.level} gpu={est.gpu_name} freq={est.freq_label} dtype={est.dtype} "
        f"static={est.static_method} P={est.static_w:.2f} W  kernels hit {est.hits}/{est.hits + est.misses} "
        f"({est.n_unique_keys} unique keys)",
        f"  T_sum   = {_us(est.T_sum)} us   T_e2e = {_us(est.T_e2e)} us   err {_pct(est.time_err)}",
        f"  E_sum   = {_mj(est.E_pred_sum)} mJ  (= sum e_k; dyn {_mj(est.E_dyn_sum)} + static {_mj(est.static_w * est.T_sum)})",
        f"  E_hyb   = {_mj(est.E_pred_hybrid)} mJ  (= sum E_dyn + P*T_e2e)",
        f"  E_e2e   = {_mj(est.E_e2e)} mJ   err sum {_pct(est.energy_err_sum)}   err hybrid {_pct(est.energy_err_hybrid)}",
    ]


def per_layer_table(est: ModelEstimate) -> str:
    leaf = est.level == "leaf"
    pre_hdr = f"{'seq':>3} {'par':>3} {'path':12} " if leaf else f"{'idx':>3} "
    hdr = pre_hdr + f"{'type':9} {'key':16} {'us':>9} {'mJ':>9} {'dyn mJ':>9} {'W':>7} {'share':>6}  desc / shared"
    out = [hdr]
    for r in est.layers:
        pre = (f"{r['layer_idx']:>3} {r['parent_layer_idx']:>3} {(r['path'] or '')[:12]:12} " if leaf
               else f"{r['layer_idx']:>3} ")
        if r["hit"]:
            out.append(
                pre + f"{r['layer_type']:9} {r['kernel_key']} {_us(r['time_s']):>9} {_mj(r['energy_j']):>9} "
                f"{_mj(r['dyn_energy_j']):>9} {r['power_w']:>7.1f} {100 * r['energy_share']:>5.1f}%  {r['short_desc']}"
                + (f"  <- {','.join(r['shared_with'])}" if r["shared_with"] else "")
            )
        else:
            out.append(pre + f"{r['layer_type']:9} {r['kernel_key']} {'MISS':>9}  {r['short_desc']}")
    return "\n".join(out)


def kernel_reuse_matrix(conn, model_names: list[str], dtype: str, batch: int, imgsz: int,
                        level: str = "module") -> list[dict]:
    rows = conn.execute(
        "SELECT kernel_key, short_desc, model_name FROM model_kernels WHERE dtype=? AND batch=? AND imgsz=? AND level=?",
        (dtype, batch, imgsz, level),
    ).fetchall()
    by_key: dict[str, dict] = {}
    for r in rows:
        d = by_key.setdefault(r["kernel_key"], {"kernel_key": r["kernel_key"], "short_desc": r["short_desc"],
                                                **{m: 0 for m in model_names}})
        if r["model_name"] in d:
            d[r["model_name"]] += 1
    return sorted(by_key.values(), key=lambda d: d["short_desc"])


def write_report(estimates: list[ModelEstimate], conn, out_dir: str | Path, model_names: list[str],
                 dtype: str, batch: int, imgsz: int, tag: str = "") -> tuple[Path, Path]:
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    level = estimates[0].level
    name = f"estimate_{tag or estimates[0].static_method}_{level}_{stamp}"
    md = out_dir / f"{name}.md"
    csvp = out_dir / f"{name}_layers.csv"
    lines = [f"# Kernel-DB energy estimate, level={level} ({stamp})", "",
             "E_pred_sum = sum_k e_k = sum_k (e_k - P t_k) + P sum_k t_k ; E_pred_hybrid = sum_k (e_k - P t_k) + P T_e2e",
             ""]
    lines += ["| model | P (W) | hits | keys | T_sum (us) | T_e2e (us) | T err | E_sum (mJ) | E_hyb (mJ) | E_e2e (mJ) | E err sum | E err hyb |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in estimates:
        lines.append(f"| {e.model_name} | {e.static_w:.1f} | {e.hits}/{e.hits + e.misses} | {e.n_unique_keys} | {_us(e.T_sum)} | {_us(e.T_e2e)} | "
                     f"{_pct(e.time_err)} | {_mj(e.E_pred_sum)} | {_mj(e.E_pred_hybrid)} | {_mj(e.E_e2e)} | "
                     f"{_pct(e.energy_err_sum)} | {_pct(e.energy_err_hybrid)} |")
    lines.append("")
    for s in db.list_static(conn, estimates[0].gpu_name, estimates[0].freq_label):
        lines.append(f"- static_power[{s['method']}] = {s['power_w']:.2f} W (n={s['n_samples']}, {s['notes'] or ''})")
    lines.append("")
    for e in estimates:
        lines += [f"## {e.model_name}", "", "```"] + summary_lines(e) + ["", per_layer_table(e), "```", ""]
    matrix = kernel_reuse_matrix(conn, model_names, dtype, batch, imgsz, level)
    lines += ["## Kernel reuse (occurrences per model)", "",
              "| key | desc | " + " | ".join(model_names) + " |", "|---|---|" + "---|" * len(model_names)]
    for d in matrix:
        lines.append(f"| {d['kernel_key']} | {d['short_desc']} | " + " | ".join(str(d[m]) for m in model_names) + " |")
    md.write_text("\n".join(lines) + "\n")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "layer_idx", "parent_layer_idx", "path", "layer_type", "kernel_key", "short_desc",
                    "hit", "time_s", "energy_j", "dyn_energy_j", "static_energy_j", "power_w", "energy_share", "shared_with"])
        for e in estimates:
            for r in e.layers:
                w.writerow([e.model_name, e.level, r["layer_idx"], r["parent_layer_idx"], r["path"], r["layer_type"],
                            r["kernel_key"], r["short_desc"], int(r["hit"]),
                            r["time_s"], r["energy_j"], r["dyn_energy_j"], r["static_energy_j"], r["power_w"],
                            r["energy_share"], ";".join(r["shared_with"])])
    return md, csvp


def plot_estimates(estimates: list[ModelEstimate], out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), tight_layout=True)
    names = [e.model_name for e in estimates]
    ax = axes[0]
    xs = range(len(names))
    dyn = [1e3 * e.E_dyn_sum for e in estimates]
    sta = [1e3 * e.static_w * e.T_sum for e in estimates]
    ax.bar(xs, sta, label="static  (P * sum t_k)", color="tab:gray")
    ax.bar(xs, dyn, bottom=sta, label="sum dynamic  (sum e_k - P t_k)", color="tab:orange")
    e2e = [1e3 * e.E_e2e if e.E_e2e is not None else 0 for e in estimates]
    ax.plot(xs, e2e, "k_", markersize=25, markeredgewidth=2, label="measured end-to-end")
    for x, e in zip(xs, estimates):
        if e.energy_err_sum is not None:
            top = max(1e3 * e.E_pred_sum, 1e3 * e.E_e2e)
            ax.annotate(f"{100 * e.energy_err_sum:+.1f}%", (x, top), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.12)
    ax.set_xticks(list(xs), names)
    ax.set_ylabel("energy per forward (mJ)")
    ax.set_title(f"[{estimates[0].level}] E_sum = sum e_k vs measured  (P = {estimates[0].static_w:.1f} W); label = E_sum error")
    ax.legend()
    ax = axes[1]
    ax.bar([x - 0.2 for x in xs], [1e6 * e.T_sum for e in estimates], width=0.4, label="sum t_k")
    ax.bar([x + 0.2 for x in xs], [1e6 * e.T_e2e if e.T_e2e else 0 for e in estimates], width=0.4, label="T_e2e")
    for x, e in zip(xs, estimates):
        if e.time_err is not None:
            top = max(1e6 * e.T_sum, 1e6 * e.T_e2e)
            ax.annotate(f"{100 * e.time_err:+.1f}%", (x, top), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.12)
    ax.set_xticks(list(xs), names)
    ax.set_ylabel("time per forward (us)")
    ax.set_title(f"[{estimates[0].level}] sum t_k vs measured; label = time error")
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------------------------
# Granularity comparison: leaf composition vs module-level rows, and DB size per level.
# ---------------------------------------------------------------------------------------------

def _row_cost_s(r: dict) -> float | None:
    """Approximate wall time spent measuring one kernel_energy row (all repeats)."""
    if r.get("measurement_duration_s") is None:
        return None
    n = r.get("n_repeats") or 1
    per = (r["measurement_duration_s"] or 0) + (r["cooldown_s"] or 0) + (r.get("warmup_settle_s") or 0)
    if r.get("time_s") is not None:
        per += ((r.get("num_warmup") or 0) * 2 + (r.get("num_calibration") or 0)) * r["time_s"]
    return n * per


def compare_levels(conn, model_name: str, gpu_name: str, freq_label: str, dtype: str, batch: int, imgsz: int) -> dict:
    """Per top-level layer: module-level row vs the sum of its leaf-level rows.

    Returns {"blocks": [rows...], "total": row}. Each row has module e/t/gpu_t, leaf sums, counts and
    ratios; None where either side is missing."""
    mod = {r["layer_idx"]: r for r in db.model_composition_join(conn, model_name, gpu_name, freq_label, dtype,
                                                                batch, imgsz, level="module")}
    leaf = db.model_composition_join(conn, model_name, gpu_name, freq_label, dtype, batch, imgsz, level="leaf")
    if not mod or not leaf:
        raise LookupError(f"{model_name}: need both module and leaf compositions in the DB")
    groups: dict[int, list[dict]] = {}
    for r in leaf:
        groups.setdefault(r["parent_layer_idx"], []).append(r)
    blocks = []
    for idx in sorted(mod):
        m = mod[idx]
        ls = groups.get(idx, [])
        hit_all = all(r["energy_j"] is not None for r in ls) and bool(ls)
        e_leaf = sum(r["energy_j"] for r in ls) if hit_all else None
        t_leaf = sum(r["time_s"] for r in ls) if hit_all else None
        g_leaf = (sum(r["gpu_kernel_time_s"] for r in ls) if hit_all and all(r["gpu_kernel_time_s"] is not None for r in ls)
                  else None)
        row = dict(
            layer_idx=idx, layer_type=m["layer_type"], short_desc=m["short_desc"],
            e_mod=m["energy_j"], t_mod=m["time_s"], g_mod=m["gpu_kernel_time_s"],
            n_leaves=sum(1 for r in ls if not r["layer_type"].startswith("op:")),
            n_ops=sum(1 for r in ls if r["layer_type"].startswith("op:")),
            n_keys=len({r["kernel_key"] for r in ls}),
            e_leaf=e_leaf, t_leaf=t_leaf, g_leaf=g_leaf,
        )
        row["e_ratio"] = (e_leaf / m["energy_j"]) if (e_leaf is not None and m["energy_j"]) else None
        row["t_ratio"] = (t_leaf / m["time_s"]) if (t_leaf is not None and m["time_s"]) else None
        row["g_ratio"] = (g_leaf / m["gpu_kernel_time_s"]) if (g_leaf is not None and m["gpu_kernel_time_s"]) else None
        blocks.append(row)

    def _tot(k):
        vals = [b[k] for b in blocks if b[k] is not None]
        return sum(vals) if vals else None

    total = dict(layer_idx=-1, layer_type="TOTAL", short_desc=model_name,
                 e_mod=_tot("e_mod"), t_mod=_tot("t_mod"), g_mod=_tot("g_mod"),
                 n_leaves=_tot("n_leaves"), n_ops=_tot("n_ops"), n_keys=len({r["kernel_key"] for r in leaf}),
                 e_leaf=_tot("e_leaf"), t_leaf=_tot("t_leaf"), g_leaf=_tot("g_leaf"))
    for k in ("e", "t", "g"):
        a, b = total[f"{k}_leaf"], total[f"{k}_mod"]
        total[f"{k}_ratio"] = (a / b) if (a is not None and b) else None
    return {"model_name": model_name, "blocks": blocks, "total": total}


def _ratio(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def compare_levels_table(cmp: dict) -> str:
    hdr = (f"{'idx':>3} {'type':8} {'leaves':>6} {'ops':>4} {'keys':>4} {'E_mod mJ':>9} {'E_leaf mJ':>9} {'E ratio':>8} "
           f"{'t_mod us':>9} {'t_leaf us':>9} {'t ratio':>8} {'gpu ratio':>9}  desc")
    out = [hdr]
    for b in cmp["blocks"] + [cmp["total"]]:
        out.append(f"{b['layer_idx']:>3} {b['layer_type']:8} {b['n_leaves']:>6} {b['n_ops']:>4} {b['n_keys']:>4} "
                   f"{_mj(b['e_mod']):>9} {_mj(b['e_leaf']):>9} {_ratio(b['e_ratio']):>8} "
                   f"{_us(b['t_mod']):>9} {_us(b['t_leaf']):>9} {_ratio(b['t_ratio']):>8} {_ratio(b['g_ratio']):>9}  "
                   f"{b['short_desc']}")
    return "\n".join(out)


def db_size_summary(conn, model_names: list[str], gpu_name: str, freq_label: str, dtype: str, batch: int,
                    imgsz: int, levels: tuple[str, ...] = ("module", "leaf")) -> list[dict]:
    """Per level and model: instances, unique keys, keys shared with other models in the list, keys new
    when models are added in the given order, and measurement cost of the rows (from stored durations)."""
    out = []
    for level in levels:
        keys_by_model: dict[str, set[str]] = {}
        inst: dict[str, int] = {}
        cost: dict[str, float] = {}
        measured: dict[str, int] = {}
        for m in model_names:
            rows = db.model_composition_join(conn, m, gpu_name, freq_label, dtype, batch, imgsz, level=level)
            keys_by_model[m] = {r["kernel_key"] for r in rows}
            inst[m] = len(rows)
            seen = {}
            for r in rows:
                seen.setdefault(r["kernel_key"], r)
            measured[m] = sum(1 for r in seen.values() if r["energy_j"] is not None)
            cost[m] = sum(c for c in (_row_cost_s(r) for r in seen.values()) or [] if c)
        cum: set[str] = set()
        all_keys: set[str] = set().union(*keys_by_model.values()) if keys_by_model else set()
        for m in model_names:
            ks = keys_by_model[m]
            others = set().union(*(keys_by_model[o] for o in model_names if o != m)) if len(model_names) > 1 else set()
            out.append(dict(level=level, model_name=m, instances=inst[m], unique_keys=len(ks),
                            measured_keys=measured[m], shared_with_others=len(ks & others),
                            new_keys=len(ks - cum), cost_min=cost[m] / 60.0))
            cum |= ks
        # cost of the union of all keys at this level
        union_cost = 0.0
        seen_u: set[str] = set()
        for m in model_names:
            for r in db.model_composition_join(conn, m, gpu_name, freq_label, dtype, batch, imgsz, level=level):
                if r["kernel_key"] in seen_u:
                    continue
                seen_u.add(r["kernel_key"])
                c = _row_cost_s(r)
                union_cost += c or 0.0
        out.append(dict(level=level, model_name="ALL", instances=sum(inst.values()), unique_keys=len(all_keys),
                        measured_keys=None, shared_with_others=sum(1 for k in all_keys
                                                                   if sum(k in v for v in keys_by_model.values()) > 1),
                        new_keys=len(all_keys), cost_min=union_cost / 60.0))
    return out


def db_size_table(rows: list[dict]) -> str:
    hdr = f"{'level':7} {'model':9} {'inst':>5} {'keys':>5} {'meas':>5} {'shared':>6} {'new':>5} {'cost min':>9}"
    out = [hdr]
    for r in rows:
        out.append(f"{r['level']:7} {r['model_name']:9} {r['instances']:>5} {r['unique_keys']:>5} "
                   f"{'-' if r['measured_keys'] is None else r['measured_keys']:>5} {r['shared_with_others']:>6} "
                   f"{r['new_keys']:>5} {r['cost_min']:>9.1f}")
    return "\n".join(out)


def write_compare_report(cmps: list[dict], size_rows: list[dict], level_estimates: dict[str, list[ModelEstimate]],
                         out_dir: str | Path, tag: str = "") -> tuple[Path, Path]:
    """Markdown + CSV: per-size errors at both levels, DB-size table, per-block tables."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"compare_levels_{tag + '_' if tag else ''}{stamp}"
    md = out_dir / f"{name}.md"
    csvp = out_dir / f"{name}_blocks.csv"
    lines = [f"# Module-level vs leaf-level composition ({stamp})", ""]
    lines += ["## Whole-model errors per level", "",
              "| model | level | kernels | keys | T_sum (us) | T_e2e (us) | T err | E_sum (mJ) | E_hyb (mJ) | E_e2e (mJ) | E err sum | E err hyb |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    models = [c["model_name"] for c in cmps]
    for m in models:
        for level, ests in level_estimates.items():
            for e in ests:
                if e.model_name != m:
                    continue
                lines.append(f"| {m} | {level} | {e.hits}/{e.hits + e.misses} | {e.n_unique_keys} | {_us(e.T_sum)} | "
                             f"{_us(e.T_e2e)} | {_pct(e.time_err)} | {_mj(e.E_pred_sum)} | {_mj(e.E_pred_hybrid)} | "
                             f"{_mj(e.E_e2e)} | {_pct(e.energy_err_sum)} | {_pct(e.energy_err_hybrid)} |")
    lines += ["", "## Database size per level", "",
              "| level | model | instances | unique keys | measured | shared with other sizes | new keys (cumulative order) | measurement cost (min) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in size_rows:
        lines.append(f"| {r['level']} | {r['model_name']} | {r['instances']} | {r['unique_keys']} | "
                     f"{'-' if r['measured_keys'] is None else r['measured_keys']} | {r['shared_with_others']} | "
                     f"{r['new_keys']} | {r['cost_min']:.1f} |")
    lines.append("")
    for c in cmps:
        lines += [f"## Per-block comparison: {c['model_name']}", "",
                  "E ratio = sum of leaf energies / module-level energy of the same layer (same for t and profiler GPU time).",
                  "", "```", compare_levels_table(c), "```", ""]
    md.write_text("\n".join(lines) + "\n")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["model", "layer_idx", "layer_type", "short_desc", "n_leaves", "n_ops", "n_keys", "e_mod", "e_leaf", "e_ratio",
                "t_mod", "t_leaf", "t_ratio", "g_mod", "g_leaf", "g_ratio"]
        w.writerow(cols)
        for c in cmps:
            for b in c["blocks"] + [c["total"]]:
                w.writerow([c["model_name"]] + [b[k] for k in cols[1:]])
    return md, csvp


def plot_level_comparison(cmps: list[dict], size_rows: list[dict], level_estimates: dict[str, list[ModelEstimate]],
                          out_path: str | Path, block_model: str | None = None) -> Path:
    """Three panels: per-block E/t ratio for one model, E_sum error per size and level, unique keys per level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), tight_layout=True)
    c = next((c for c in cmps if c["model_name"] == block_model), cmps[0])
    ax = axes[0]
    blocks = [b for b in c["blocks"] if b["n_leaves"] + b["n_ops"] > 1]  # composite layers only
    xs0 = list(range(len(blocks)))
    er = [100 * (b["e_ratio"] - 1) if b["e_ratio"] is not None else float("nan") for b in blocks]
    tr = [100 * (b["t_ratio"] - 1) if b["t_ratio"] is not None else float("nan") for b in blocks]
    ax.bar([i - 0.2 for i in xs0], er, width=0.4, label="energy: sum leaves vs module row")
    ax.bar([i + 0.2 for i in xs0], tr, width=0.4, label="time: sum leaves vs module row")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(xs0, [f"L{b['layer_idx']} {b['layer_type']}" for b in blocks], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("(sum of leaves - module row) / module row  [%]")
    ax.set_title(f"{c['model_name']}: composite layers, leaf composition vs module row")
    ax.legend(fontsize=8)

    ax = axes[1]
    models = [c["model_name"] for c in cmps]
    xs = range(len(models))
    levels = list(level_estimates)
    w = 0.8 / max(1, len(levels))
    for j, level in enumerate(levels):
        by = {e.model_name: e for e in level_estimates[level]}
        vals = [100 * by[m].energy_err_sum if m in by and by[m].energy_err_sum is not None else float("nan") for m in models]
        ax.bar([x + (j - (len(levels) - 1) / 2) * w for x in xs], vals, width=w, label=f"{level}: E_sum error")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(xs), models)
    ax.set_ylabel("(E_sum - E_e2e) / E_e2e  [%]")
    ax.set_title("energy composition error per level")
    ax.legend(fontsize=8)

    ax = axes[2]
    for j, level in enumerate(levels):
        vals = [next((r["unique_keys"] for r in size_rows if r["level"] == level and r["model_name"] == m), 0) for m in models]
        ax.bar([x + (j - (len(levels) - 1) / 2) * w for x in xs], vals, width=w, label=f"{level}")
    tot = {level: next((r["unique_keys"] for r in size_rows if r["level"] == level and r["model_name"] == "ALL"), 0)
           for level in levels}
    ax.set_xticks(list(xs), models)
    ax.set_ylabel("unique kernel keys")
    ax.set_title("DB rows per model  (all sizes: " + ", ".join(f"{k} {v}" for k, v in tot.items()) + ")")
    ax.legend(fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
