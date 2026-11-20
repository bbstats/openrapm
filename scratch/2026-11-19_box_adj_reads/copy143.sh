#!/usr/bin/env bash
# Experiment 143's tables into the published names, under the morning job's lock: the six product files, the ones
# before kept as *_before_rd143*.
#     bash scratch/2026-11-19_box_adj_reads/copy143.sh
set -euo pipefail
cd /a/code/spmm
. scripts/daily_lock.sh
take_lock "experiment 143's tables copied in" || exit 1
trap release_lock EXIT
for kind in "" _regular _step3 _step4; do
  cp -p "outputs/season_ratings_product${kind}.parquet" "outputs/season_ratings_product${kind}_before_rd143.parquet"
  cp -p "outputs/season_ratings_product143${kind}.parquet" "outputs/season_ratings_product${kind}.parquet"
done
cp -p outputs/csv/season_ratings_product_step4_multipliers.csv outputs/csv/season_ratings_product_step4_multipliers_before_rd143.csv
cp -p outputs/csv/season_ratings_product143_step4_multipliers.csv outputs/csv/season_ratings_product_step4_multipliers.csv
cp -p outputs/csv/wins_product_seasons.csv outputs/csv/wins_product_seasons_before_rd143.csv
cp -p outputs/csv/wins_product143_seasons.csv outputs/csv/wins_product_seasons.csv
for kind in "" _regular _step3 _step4; do
  cmp -s "outputs/season_ratings_product143${kind}.parquet" "outputs/season_ratings_product${kind}.parquet" || { echo "copy differs: $kind"; exit 1; }
done
echo "six product files copied; the ones before kept as *_before_rd143*"
