# e-calc: per-kernel GPU energy database (targets: YOLO26, RFML modulation classifiers)

Energy model

    E_kernel = E_dyn(kernel) + P_static * t(kernel)      measured once per (kernel, input shape, GPU, clock, dtype)
    E_model  = sum_k E_dyn(k) + P_static * sum_k t(k)

A *kernel* is measured in isolation with `zeus.profile.measure`, stored in a SQLite
database keyed by a structural signature hash + input shapes, and a model's energy is then obtained by looking up
its kernels and summing. The end-to-end fused forward is measured with the same protocol as ground truth.

Two granularities (`--level`) share one `kernel_energy` table (keys are level-agnostic). `estimate_model.py --compare-levels` prints the per-block leaf-sum vs module-row table and the DB-size table.

- `module` (default): every top-level Ultralytics module of the YOLO yaml graph (Conv, C3k2, SPPF, C2PSA, Upsample,
  Concat, Detect) is a kernel; 24 per YOLO26 model.
- `leaf`: leaf modules (Ultralytics Conv/DWConv = fused Conv2d + act, MaxPool2d, Upsample, Concat, Attention; Detect
  kept whole) plus the functional glue between them (`torch.cat`, `chunk`, `split`, residual `add`), captured with
  forward hooks and a `TorchFunctionMode`; 126-215 per YOLO26 model. 

Scope: inference, batch 1 and 8, fp32 (TF32 allowed unless `--strict-fp32`), 640x640, NMS-free (end2end) head,
one A40.

RFML family (`--models rfml`, module level only): `rf_vgg` and `rf_resnet` (O'Shea et al. 2018) and `rf_lstm`
(Rajendran et al. 2018) on RadioML 2018.01A-style 2x1024 IQ frames, 24 classes, seeded random weights. Each model
is a top-level `ModuleList` run in order (11 / 10 / 3 components); for these models `--imgsz` and the DB `imgsz`
column hold the IQ frame length (default 1024).

Study reports with results: [docs/report_yolo26_a40.md](docs/report_yolo26_a40.md) (YOLO26; data and full report
in [docs/data/yolo26/](docs/data/yolo26/)), [docs/report_rfml_a40.md](docs/report_rfml_a40.md) (RFML, batch 32 and
256; data in [docs/data/rfml/](docs/data/rfml/)).

## Setup

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync                        # Python 3.12, torch cu13, ultralytics, zeus, nvidia-ml-py ...
    sudo nvidia-smi -i 0 -pm 1     # once, needs root: GPU persistence mode (scripts refuse to measure without it)

## Run

    uv run pytest tests -m "not gpu"                    # CPU unit tests (signatures, DB)
    uv run python scripts/inspect_model.py --models all # capture only: layer table, kernel keys, reuse across sizes
    uv run python scripts/inspect_model.py --models all --level leaf   # same at leaf level (no GPU needed)
    scripts/run_all.sh                                  # static power -> kernel DB -> end-to-end -> estimate -> CSV
    LEVEL=leaf SKIP_STATIC=1 SKIP_E2E=1 scripts/run_all.sh             # leaf DB + level comparison, reusing rows
    MODELS=rfml BATCH=256 SKIP_STATIC=1 EXTRA="--repeats 3" scripts/run_all.sh   # RFML study, one batch size

Individual steps (all accept `--db`, `--repeats`, `--measurement-duration`, `--cooldown`, `--strict-fp32`, ...):

    uv run python scripts/measure_static.py --models yolo26n --auto-window --with-p8
    uv run python scripts/build_kernel_db.py --models all          # resumable, skips keys already in the DB
    uv run python scripts/build_kernel_db.py --models all --level leaf
    uv run python scripts/measure_model_e2e.py --models all
    uv run python scripts/estimate_model.py --models all --static-method active_idle --plot
    uv run python scripts/estimate_model.py --models all --level leaf --compare-levels --plot

Outputs: `data/ecalc.sqlite` (tables `kernel_energy`, `static_power`, `model_runs`, `model_kernels`),
`data/export/*.csv`, `results/reports/estimate_*.md|csv`, `results/reports/compare_levels_*.md|csv`,
`results/plots/`, `results/logs/`,
`results/static_power/decay_<gpu>.csv`.

## Layout

    ecalc/config.py       MeasureConfig, paths, scope constants
    ecalc/gpu.py          NVML: handle, snapshot, persistence mode, best-effort clock lock, ClockSampler
    ecalc/signature.py    module signature + input specs -> kernel_key; synthesize_inputs
    ecalc/models.py       YOLO26 registry, load_fused_model (end2end head), make_input, TF32 flags, family dispatch
    ecalc/rfml.py         RFML models (VGG, ResNet, LSTM), seeded loader, IQ frame inputs
    ecalc/capture.py      module level: BaseModel._predict_once routing (YOLO) or in-order ModuleList (RFML);
                          leaf level: hooks + TorchFunctionMode
    ecalc/measure.py      zeus.profile.measure wrapper with repeats + clock sampling (kernel and e2e)
    ecalc/static_power.py active_idle / post_burst / p8_idle
    ecalc/db.py           SQLite schema and API
    ecalc/estimate.py     composition, validation metrics, level comparison, markdown/CSV reports, plots
    scripts/              inspect_model, measure_static, build_kernel_db, measure_model_e2e, estimate_model, run_all.sh
    tests/                CPU tests + GPU consistency tests (`-m gpu`)
