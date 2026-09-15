#!/usr/bin/env bash
# Full study: static power -> kernel DB (all sizes) -> end-to-end -> estimates -> CSV export.
# Env: MODELS (default all), BATCH (default 1), LEVEL (module|leaf, default module),
#      SKIP_STATIC=1 / SKIP_E2E=1 to reuse stored static / end-to-end rows, EXTRA (extra flags, e.g. "--repeats 3"),
#      ALLOW_NO_PM=1 to measure without GPU persistence mode (rows flagged no_persistence_mode).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export KINETO_LOG_LEVEL="${KINETO_LOG_LEVEL:-5}"
MODELS="${MODELS:-all}"
BATCH="${BATCH:-1}"
LEVEL="${LEVEL:-module}"
EXTRA="${EXTRA:-}"
FILTER='WARNING|sudo nvidia-smi|verify with|^\{|^$|USDT'

pm=$(nvidia-smi -i 0 --query-gpu=persistence_mode --format=csv,noheader)
if [[ "$pm" != "Enabled" ]]; then
  if [[ "${ALLOW_NO_PM:-0}" == "1" ]]; then
    echo "WARNING: persistence mode is $pm; measuring anyway (ALLOW_NO_PM=1), rows flagged" >&2
    EXTRA="$EXTRA --allow-no-persistence"
  else
    echo "GPU persistence mode is $pm. Run once:  sudo nvidia-smi -i 0 -pm 1   (or set ALLOW_NO_PM=1)" >&2
    exit 1
  fi
fi

if [[ "${SKIP_STATIC:-0}" == "1" ]]; then echo "=== STATIC === skipped (SKIP_STATIC=1, reusing stored rows)"; else
echo "=== STATIC ===";   uv run python scripts/measure_static.py --models yolo26n --auto-window --with-p8 $EXTRA 2>&1 | grep -Ev "$FILTER"; fi
echo "=== KERNELS ($LEVEL) ===";  uv run python scripts/build_kernel_db.py --models "$MODELS" --batch "$BATCH" --level "$LEVEL" --profile-kernel-time $EXTRA 2>&1 | grep -Ev "$FILTER" | grep -Ev "repeat [0-9]"
if [[ "${SKIP_E2E:-0}" == "1" ]]; then echo "=== E2E === skipped (SKIP_E2E=1, reusing stored rows)"; else
echo "=== E2E ===";      uv run python scripts/measure_model_e2e.py --models "$MODELS" --batch "$BATCH" --profile-kernel-time $EXTRA 2>&1 | grep -Ev "$FILTER"; fi
CMP=""; if [[ "$LEVEL" == "leaf" ]]; then CMP="--compare-levels"; fi
echo "=== ESTIMATE active_idle ==="; uv run python scripts/estimate_model.py --models "$MODELS" --batch "$BATCH" --level "$LEVEL" --static-method active_idle --plot --tag "active_idle_b$BATCH" $CMP
uv run python -c "from ecalc import db, config as C; print(db.export_csv(db.connect(C.DB_PATH), C.EXPORT_DIR))"
echo "=== DONE ==="
