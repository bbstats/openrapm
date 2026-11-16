#!/usr/bin/env bash
# The after side's published-way tables against the published ones, each finished season.
export PYTHONPATH=/a/code/spmm_cleanup/src
export OPENRAPM_CONFIG=A:/code/spmm/outputs/cleanup_proof/config.yaml
O=/a/code/spmm/outputs
for suffix in _step3 _step4 ""; do
  for season in $(seq 1997 2026); do
    v=$(/a/code/spmm/.venv/Scripts/python -m openrapm compare "$O/season_ratings_cu_a_prod$suffix.parquet" \
        "$O/season_ratings_product$suffix.parquet" --season "$season" 2>&1 | head -1)
    echo "product$suffix $season $v"
  done
done
