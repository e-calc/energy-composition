#!/usr/bin/env bash
# Regenerate the structure-only fixtures: which kernels each model is made of. No measured numbers.
# Needs torch and Ultralytics (CPU is enough). Run from the repository root.
set -euo pipefail
PYTHON="${PYTHON:-python}"
OUT=tests/fixtures
$PYTHON -m ecalc inspect --model yolo26n --batch-size 1 --level module --output $OUT/yolo26n_module_forward.json > /dev/null
$PYTHON -m ecalc inspect --model yolo26n --batch-size 1 --level leaf --output $OUT/yolo26n_leaf_forward.json > /dev/null
$PYTHON -m ecalc inspect --model yolo26n --batch-size 1 --level module --direction backward --output $OUT/yolo26n_module_backward.json > /dev/null
for MODEL in rf_vgg rf_resnet rf_lstm; do
  $PYTHON -m ecalc inspect --model $MODEL --batch-size 4 --output $OUT/${MODEL}_module_forward.json > /dev/null
  $PYTHON -m ecalc inspect --model $MODEL --batch-size 4 --direction backward --output $OUT/${MODEL}_module_backward.json > /dev/null
done
ls -la $OUT/*.json
