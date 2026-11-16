"""The ``ecalc`` command.

    ecalc estimate   energy and time of a model from a signature database
    ecalc compose    energy and time of any pipeline file (sequential / parallel tree)
    ecalc plan       which kernels of a model are measured, fillable by formula, or missing
    ecalc inspect    capture a model's kernels on CPU; optionally write its pipeline file
    ecalc export     write the measurement store as a signature database (JSON Lines)
    ecalc measure    static-power | kernels | model      (needs a CUDA GPU)

``estimate``, ``compose`` and ``plan`` need neither a GPU nor torch when the pipeline is in the
database. A model that is not in the database is captured on CPU, which needs torch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import config as C
from .calculus import (DEFAULT_PRECISION, DIRECTIONS, FORWARD, MEASURED, PRECISIONS, Estimate, Kernel, Pipeline,
                       Platform, SignatureDatabase, StaticPower, dump, evaluate, kernels, load, pipeline_name)
from .calculus.errors import EnergyCalculusError
from .calculus.serialization import expression_to_dict, pipeline_from_dict

YOLO26_MODELS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")
RFML_MODELS = ("rf_vgg", "rf_resnet", "rf_lstm")
MODEL_GROUPS = {"all": YOLO26_MODELS, "yolo26": YOLO26_MODELS, "rfml": RFML_MODELS}
DEFAULT_INPUT_SIZE = {**{m: 640 for m in YOLO26_MODELS}, **{m: 1024 for m in RFML_MODELS}}


class CommandError(Exception):
    """A problem to report to the user without a traceback."""


# ---- database, platform, static power ----------------------------------------------------------

def default_database_path() -> Path:
    """``$ECALC_DATABASE``, else the exported signature database, else the measurement store."""
    if os.environ.get("ECALC_DATABASE"):
        return Path(os.environ["ECALC_DATABASE"])
    for candidate in (C.SIGNATURE_DATABASE_PATH, C.DB_PATH):
        if candidate.exists():
            return candidate
    return C.SIGNATURE_DATABASE_PATH


def open_database(path: str | Path | None) -> SignatureDatabase:
    """A signature database from a JSON Lines export or straight from a SQLite measurement store."""
    path = Path(path) if path else default_database_path()
    if not path.exists():
        raise CommandError(
            f"no signature database at {path}. Measure one (`ecalc measure kernels`, then `ecalc export`) "
            f"or pass --database <file>.")
    if path.suffix in (".sqlite", ".sqlite3", ".db"):
        from . import db

        return db.signature_database(db.connect(path))
    document = load(path)
    if not isinstance(document, SignatureDatabase):
        raise CommandError(f"{path} is not a signature database")
    return document


def choose_platform(database: SignatureDatabase, gpu: str | None, frequency_label: str) -> Platform:
    platforms = database.platforms()
    if gpu is None:
        names = sorted({p.gpu_name for p in platforms})
        if len(names) != 1:
            raise CommandError(f"--gpu is required: the database has {names or 'no signatures'}")
        gpu = names[0]
    platform = Platform(gpu, frequency_label)
    if platform not in platforms:
        raise CommandError(f"no signatures for {platform}; the database has {[str(p) for p in platforms]}")
    return platform


def stored_static_power(database: SignatureDatabase, platform: Platform) -> StaticPower:
    try:
        return database.static_power(platform)
    except LookupError as error:
        raise CommandError(f"{error} in the database. Run `ecalc measure static-power`.") from None


def source_for(database: SignatureDatabase, fill: str | None):
    if fill == "formula":
        from .formula import with_formulas

        return with_formulas(database)
    return database


# ---- pipelines ---------------------------------------------------------------------------------

def parse_models(text: str) -> list[str]:
    return list(MODEL_GROUPS[text]) if text in MODEL_GROUPS else [m.strip() for m in text.split(",") if m.strip()]


def capture_pipeline(model: str, batch_size: int, input_size: int, level: str, direction: str,
                     precision: str = DEFAULT_PRECISION) -> Pipeline:
    """Capture a model's kernels on CPU. Needs torch (and Ultralytics for YOLO26)."""
    try:
        import torch

        from .capture import capture, model_composition
        from .db import composition_expression
        from .models import load_model, model_input
    except ImportError as error:
        raise CommandError(f"{model} (batch size {batch_size}, input size {input_size}, {level}, {direction}) is not "
                           f"in the database, and capturing it needs torch: {error}. "
                           f"Install with `pip install 'ecalc[capture]'`.") from None
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    network = load_model(model, "cpu")
    records = capture(network, model_input(model, batch_size, input_size, "cpu"), level=level, direction=direction)
    rows = model_composition(records, model, precision, batch_size, input_size, "")
    return Pipeline(pipeline_name(model, batch_size, input_size, level, direction, precision),
                    composition_expression(rows),
                    {"model": model, "batch_size": batch_size, "input_size": input_size, "level": level,
                     "direction": direction, "precision": precision, "captured_with": _capture_versions()})


def _capture_versions() -> dict:
    """Versions of the libraries a capture depends on; kernel keys can change between them."""
    import importlib.metadata

    versions = {}
    for package in ("torch", "ultralytics"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def find_pipeline(database: SignatureDatabase, model: str, batch_size: int, input_size: int | None, level: str,
                  direction: str, precision: str = DEFAULT_PRECISION) -> tuple[Pipeline, bool]:
    """The model's pipeline and whether it came from the database (True) or a fresh CPU capture (False)."""
    if input_size is None:
        input_size = DEFAULT_INPUT_SIZE.get(model)
        if input_size is None:
            sizes = {p.attributes.get("input_size") for p in database.pipelines() if p.attributes.get("model") == model}
            if len(sizes) != 1:
                raise CommandError(f"--input-size is required for {model}")
            input_size = sizes.pop()
    stored = database.pipeline(pipeline_name(model, batch_size, input_size, level, direction, precision))
    if stored is not None:
        return stored, True
    return capture_pipeline(model, batch_size, input_size, level, direction, precision), False


def load_pipeline_file(path: str | Path, database: SignatureDatabase) -> Pipeline:
    """A pipeline file. Besides ``{"kernel": key}`` nodes it may hold ``{"pipeline": name}`` nodes,
    which stand for a pipeline stored in the database."""
    document = json.loads(Path(path).read_text())

    def resolve(node: dict) -> dict:
        if "pipeline" in node:
            stored = database.pipeline(node["pipeline"])
            if stored is None:
                known = "\n  ".join(sorted(p.name for p in database.pipelines()))
                raise CommandError(f"{path}: no pipeline named {node['pipeline']!r} in the database. Known:\n  {known}")
            for leaf in kernels(stored.expression):
                document["kernels"].setdefault(leaf.specification.key, leaf.specification.signature)
            inlined = expression_to_dict(stored.expression)
            inlined["name"] = node.get("name", node["pipeline"])
            return inlined
        for node_type in ("sequential", "parallel"):
            if node_type in node:
                return {**node, node_type: [resolve(part) for part in node[node_type]]}
        return node

    document.setdefault("kernels", {})
    document["expression"] = resolve(document["expression"])
    return pipeline_from_dict(document)


# ---- reporting ---------------------------------------------------------------------------------

def _millijoules(value: float) -> str:
    return f"{value * 1e3:,.1f} mJ"


def _microseconds(value: float) -> str:
    return f"{value * 1e6:,.1f} us"


def _percent(value: float) -> str:
    return f"{100 * value:+.1f}%"


def summarize(estimate: Estimate, static_power: StaticPower, negative_dynamic: str = "raise") -> dict:
    """Numbers of an evaluated expression as a plain dictionary (also the ``--json`` output)."""
    energy = estimate.energy(static_power, negative_dynamic=negative_dynamic)
    in_sequence = estimate.cost.energy_joules
    provenance = {name: {"kernels": total.kernel_count, "time_seconds": total.time_seconds,
                         "energy_joules": total.energy_joules} for name, total in estimate.by_provenance().items()}
    return {
        "platform": str(estimate.platform),
        "static_power_watts": static_power.watts,
        "kernels": len(estimate.leaves), "unique_kernels": len({leaf.specification.key for leaf in estimate.leaves}),
        "missing_kernels": len(estimate.missing),
        "time_seconds": estimate.time_seconds, "energy_joules": energy,
        "static_energy_joules": estimate.static_energy(static_power),
        "dynamic_energy_joules": estimate.dynamic_energy(static_power),
        "sequential_time_seconds": estimate.cost.busy_seconds, "sequential_energy_joules": in_sequence,
        "overlap_saving_joules": in_sequence - energy,
        "provenance": provenance,
    }


def print_summary(title: str, summary: dict, reference: dict | None = None, show_time: bool = True) -> None:
    """``show_time=False`` prints energy only (``ecalc estimate``); times stay in the ``--json`` output."""
    print(title)
    print(f"  platform            {summary['platform']}, static power {summary['static_power_watts']:.1f} W")
    print(f"  kernels             {summary['kernels']} ({summary['unique_kernels']} unique)")
    if show_time:
        print(f"  time                {_microseconds(summary['time_seconds'])}")
    print(f"  energy              {_millijoules(summary['energy_joules'])}")
    print(f"    static  s*T       {_millijoules(summary['static_energy_joules'])}")
    print(f"    dynamic d         {_millijoules(summary['dynamic_energy_joules'])}")
    if summary["overlap_saving_joules"] != 0:
        print(f"  without overlap     {_millijoules(summary['sequential_energy_joules'])} in "
              f"{_microseconds(summary['sequential_time_seconds'])}; overlap saves "
              f"{_millijoules(summary['overlap_saving_joules'])}")
    parts = [f"{name} {entry['kernels']} kernels, {_millijoules(entry['energy_joules'])}"
             for name, entry in sorted(summary["provenance"].items())]
    print(f"  by provenance       {'; '.join(parts)}")
    if summary["missing_kernels"]:
        print(f"  WARNING             {summary['missing_kernels']} kernels have no signature and are priced at zero; "
              f"measure them or pass --fill formula")
    if reference and not show_time:
        print(f"  measured end to end {_millijoules(reference['energy_joules'])}")
        print(f"    composition error {_percent(reference['energy_error'])}")
    elif reference:
        print(f"  measured end to end {_millijoules(reference['energy_joules'])} in "
              f"{_microseconds(reference['time_seconds'])}")
        print(f"    composition error {_percent(reference['energy_error'])} energy, "
              f"{_percent(reference['time_error'])} time")
        print(f"    with measured time {_millijoules(reference['hybrid_energy_joules'])} "
              f"(dynamic + static power x measured time), error {_percent(reference['hybrid_energy_error'])}")


def print_kernel_table(estimate: Estimate, static_power: StaticPower, show_time: bool = True) -> None:
    def time_column(text: str) -> str:
        return f" {text:>10}" if show_time else ""

    print(f"  {'kernel':16}{time_column('us')} {'mJ':>10} {'dyn mJ':>10} {'W':>7}  {'provenance':12} description")
    for leaf in estimate.leaves:
        signature = leaf.signature
        if signature is None:
            print(f"  {leaf.specification.key:16}{time_column('-')} {'-':>10} {'-':>10} {'-':>7}  {'missing':12} "
                  f"{leaf.specification.description}")
            continue
        print(f"  {leaf.specification.key:16}{time_column(f'{signature.time_seconds * 1e6:.1f}')} "
              f"{signature.energy_joules * 1e3:>10.3f} "
              f"{signature.dynamic_energy(static_power) * 1e3:>10.3f} {signature.power_watts:>7.1f}  "
              f"{signature.provenance:12} {leaf.specification.description}"
              + (f"  [{signature.detail}]" if signature.provenance != MEASURED and signature.detail else ""))


# ---- commands ----------------------------------------------------------------------------------

def add_database_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", default=None,
                        help="signature database (.jsonl export) or measurement store (.sqlite); default "
                             "$ECALC_DATABASE, else data/signature_database.jsonl, else data/ecalc.sqlite")
    parser.add_argument("--gpu", default=None, help="GPU name in the database (default: the only one present)")
    parser.add_argument("--frequency-label", default="default")
    parser.add_argument("--fill", choices=["formula"], default=None,
                        help="estimate unmeasured kernels by interpolation and the light-op formula")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", "--models", dest="models", required=True,
                        help="model name, comma list, 'all' (YOLO26 sizes) or 'rfml'")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", "--image-size", dest="input_size", type=int, default=None,
                        help="YOLO image side in pixels (default 640) or RFML frame length (default 1024)")
    parser.add_argument("--level", choices=list(C.LEVELS), default="module")
    parser.add_argument("--direction", choices=list(DIRECTIONS), default=FORWARD)
    parser.add_argument("--precision", choices=list(PRECISIONS), default=DEFAULT_PRECISION,
                        help="numeric precision the model runs at; fp32 (with TF32) is the only one supported")


def estimate_command(args) -> int:
    database = open_database(args.database)
    platform = choose_platform(database, args.gpu, args.frequency_label)
    static_power = stored_static_power(database, platform)
    source = source_for(database, args.fill)
    results = []
    for model in parse_models(args.models):
        pipeline, stored = find_pipeline(database, model, args.batch_size, args.input_size, args.level, args.direction,
                                         args.precision)
        estimate = evaluate(pipeline.expression, source, platform, on_missing="report")
        summary = summarize(estimate, static_power)
        attributes = pipeline.attributes
        run = database.model_run(model, attributes["batch_size"], attributes["input_size"], platform, args.direction,
                                 args.precision)
        reference = None
        if run is not None and estimate.is_complete:
            hybrid = summary["dynamic_energy_joules"] + static_power.watts * run.time_seconds
            reference = {"energy_joules": run.energy_joules, "time_seconds": run.time_seconds,
                         "energy_error": summary["energy_joules"] / run.energy_joules - 1,
                         "time_error": summary["time_seconds"] / run.time_seconds - 1,
                         "hybrid_energy_joules": hybrid, "hybrid_energy_error": hybrid / run.energy_joules - 1}
        results.append({"pipeline": pipeline.name, **attributes, "pipeline_source": "database" if stored else "captured",
                        **summary, "measured_end_to_end": reference})
        if args.json:
            continue
        title = (f"{model}: batch size {attributes['batch_size']}, input size {attributes['input_size']}, "
                 f"{args.level} level, {args.direction}"
                 + ("" if args.precision == DEFAULT_PRECISION else f", {args.precision}") + ("" if stored else "  (captured on CPU, not in the database)"))
        print_summary(title, summary, reference, show_time=False)
        if args.kernels:
            print_kernel_table(estimate, static_power, show_time=False)
        print()
    if args.json:
        print(json.dumps(results, indent=2))
    return 1 if any(r["missing_kernels"] for r in results) else 0


def compose_command(args) -> int:
    database = open_database(args.database)
    platform = choose_platform(database, args.gpu, args.frequency_label)
    static_power = stored_static_power(database, platform)
    pipeline = load_pipeline_file(args.pipeline, database)
    estimate = evaluate(pipeline.expression, source_for(database, args.fill), platform, on_missing="report")
    summary = summarize(estimate, static_power, negative_dynamic="allow" if args.allow_negative_dynamic else "raise")
    if args.json:
        print(json.dumps({"pipeline": pipeline.name, **summary}, indent=2))
    else:
        print_summary(pipeline.name, summary)
        if args.kernels:
            print_kernel_table(estimate, static_power)
    return 1 if summary["missing_kernels"] else 0


def plan_command(args) -> int:
    """What ``ecalc measure kernels --fill formula`` would measure for these models."""
    from .formula import with_formulas

    database = open_database(args.database)
    platform = choose_platform(database, args.gpu, args.frequency_label)
    source = with_formulas(database)
    report = []
    for model in parse_models(args.models):
        pipeline, _ = find_pipeline(database, model, args.batch_size, args.input_size, args.level, args.direction,
                                    args.precision)
        unique: dict[str, Kernel] = {}
        for leaf in kernels(pipeline.expression):
            unique.setdefault(leaf.specification.key, leaf)
        counts: dict[str, int] = {}
        to_measure = []
        for leaf in unique.values():
            signature = source.lookup(leaf.specification, platform)
            label = "missing" if signature is None else signature.provenance
            counts[label] = counts.get(label, 0) + 1
            if signature is None or signature.provenance == "extrapolated":
                to_measure.append(leaf.specification.description)
        report.append({"pipeline": pipeline.name, "unique_kernels": len(unique), "by_provenance": counts,
                       "to_measure": to_measure})
        if not args.json:
            print(f"{pipeline.name}: {len(unique)} unique kernels")
            for label in ("measured", "interpolated", "formula", "extrapolated", "missing"):
                if counts.get(label):
                    print(f"  {label:13} {counts[label]}")
            print(f"  to measure    {len(to_measure)} (missing or extrapolated)" + (":" if to_measure else ""))
            for description in to_measure:
                print(f"    {description}")
            print()
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


def inspect_command(args) -> int:
    for model in parse_models(args.models):
        input_size = args.input_size or DEFAULT_INPUT_SIZE.get(model)
        if input_size is None:
            raise CommandError(f"--input-size is required for {model}")
        pipeline = capture_pipeline(model, args.batch_size, input_size, args.level, args.direction, args.precision)
        leaves = list(kernels(pipeline.expression))
        print(f"{pipeline.name}: {len(leaves)} kernels, {len({leaf.specification.key for leaf in leaves})} unique")
        for leaf in leaves:
            print(f"  {leaf.name:28} {leaf.specification.key}  {leaf.specification.description}")
        if args.output:
            target = Path(args.output)
            if len(parse_models(args.models)) > 1:
                target = target.with_name(f"{target.stem}_{model}{target.suffix}")
            print(f"  wrote {dump(pipeline, target)}")
    return 0


def export_command(args) -> int:
    from . import db

    store = Path(args.store)
    if not store.exists():
        raise CommandError(f"no measurement store at {store}")
    conn = db.connect(store)
    path = db.export_signature_database(conn, args.output)
    database = db.signature_database(conn)
    print(f"wrote {path}: {len(database)} signatures, {len(database.static_powers())} static power records, "
          f"{len(database.pipelines())} pipelines, {len(database.model_runs())} end-to-end runs")
    print("This file is measured data for one GPU; keep it out of version control.")
    return 0


MEASURE_COMMANDS = {"static-power": "static_power_main", "kernels": "kernels_main", "model": "model_main"}


def measure_command(arguments: list[str]) -> int:
    """``ecalc measure <what> [options]``; the measurement commands parse their own options."""
    if not arguments or arguments[0] in ("-h", "--help") or arguments[0] not in MEASURE_COMMANDS:
        print(f"usage: ecalc measure {{{','.join(MEASURE_COMMANDS)}}} [options]\n"
              f"       ecalc measure <command> --help")
        return 0 if arguments and arguments[0] in ("-h", "--help") else 2
    try:
        from . import measurement_commands
    except ImportError as error:
        raise CommandError(f"measuring needs torch, Zeus and NVML: {error}. "
                           f"Install with `pip install 'ecalc[measure]'`.") from None
    getattr(measurement_commands, MEASURE_COMMANDS[arguments[0]])(arguments[1:])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecalc", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    estimate = commands.add_parser("estimate", help="energy and time of a model from a signature database")
    add_model_arguments(estimate)
    add_database_arguments(estimate)
    estimate.add_argument("--kernels", action="store_true", help="also print the per-kernel table")
    estimate.set_defaults(run=estimate_command)

    compose = commands.add_parser("compose", help="energy and time of a pipeline file (sequential / parallel tree)")
    compose.add_argument("pipeline", help="pipeline JSON file (schema ecalc.pipeline/1)")
    add_database_arguments(compose)
    compose.add_argument("--kernels", action="store_true", help="also print the per-kernel table")
    compose.add_argument("--allow-negative-dynamic", action="store_true",
                         help="do not fail when an overlapped kernel draws less than the static power")
    compose.set_defaults(run=compose_command)

    plan = commands.add_parser("plan", help="which kernels are measured, fillable by formula, or missing")
    add_model_arguments(plan)
    add_database_arguments(plan)
    plan.set_defaults(run=plan_command)

    inspect = commands.add_parser("inspect", help="capture a model's kernels on CPU (needs torch)")
    add_model_arguments(inspect)
    inspect.add_argument("--output", default=None, help="write the pipeline as JSON to this file")
    inspect.set_defaults(run=inspect_command)

    export = commands.add_parser("export", help="write the measurement store as a signature database")
    export.add_argument("--store", default=str(C.DB_PATH), help="SQLite measurement store")
    export.add_argument("--output", default=str(C.SIGNATURE_DATABASE_PATH), help="JSON Lines file to write")
    export.set_defaults(run=export_command)

    commands.add_parser("measure", help="measure on a CUDA GPU: static-power | kernels | model", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    try:
        if argv[:1] == ["measure"]:
            return measure_command(argv[1:])
        args = parser.parse_args(argv)
        return args.run(args)
    except (CommandError, EnergyCalculusError) as error:
        print(f"ecalc: {error}", file=sys.stderr)
        return 2
