#!/usr/bin/env bash
# The Bias page's file and ratings.json from the code before the cleanup and from the cleanup checkout, the same inputs
# and date: the catalogue's words feed the Bias page's row labels, so the files must come out the same bytes.
set -uo pipefail
OUT=${1:?the folder to write into}
export OPENRAPM_CONFIG=/a/code/spmm/outputs/cleanup_proof/config.yaml PYTHONIOENCODING=utf-8
for side in before cleanup; do
  PYTHONPATH=/a/code/spmm_$side/src /a/code/spmm/.venv/Scripts/python -m openrapm site --bias \
    --out-dir "$OUT/$side" --built 2026-11-16 > "$OUT/$side.log" 2>&1 || { echo "FAILED: $side"; tail -5 "$OUT/$side.log"; exit 1; }
done
for file in bias.json ratings.json; do
  cmp -s "$OUT/before/data/$file" "$OUT/cleanup/data/$file" && echo "same bytes: $file" || { echo "DIFFERENT: $file"; exit 1; }
done
