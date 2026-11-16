#!/usr/bin/env bash
# One side of the cleanup's proof that no rating moved (notes/DECISIONS.md, "The cleanup of 2026-11-16").
#
#     bash scratch/2026-11-16_cleanup/side.sh b      the code before, A:/code/spmm_before (the tag archive/research-2026-11)
#     bash scratch/2026-11-16_cleanup/side.sh a      the code after, A:/code/spmm_cleanup
#
# Every table goes under a scratch name, cu_<side>_..., into A:/code/spmm/outputs; both sides read the same data
# through outputs/cleanup_proof/config.yaml (make_config.py).  A step that finished leaves a marker in
# outputs/cleanup_proof/done_<side>/ and is skipped on a rerun.  compare.py reads the two sides against each other.
set -uo pipefail
S=$1
case $S in
  b) TREE=/a/code/spmm_before; SIDE=before ;;
  a) TREE=/a/code/spmm_cleanup; SIDE=after ;;
  *) echo "usage: side.sh b|a"; exit 1 ;;
esac
MAIN=/a/code/spmm
PROOF=$MAIN/outputs/cleanup_proof
DONE=$PROOF/done_$S
LOG=$PROOF/side_$S.log
mkdir -p "$DONE"
export PYTHONPATH=$TREE/src
export OPENRAPM_CONFIG=$PROOF/config.yaml
export OPENRAPM_JOBS=4
export PYTHONIOENCODING=utf-8
PY=$MAIN/.venv/Scripts/python
OUT=$MAIN/outputs
cd "$TREE" || exit 1
export DAILY_LOCK=$MAIN/outputs/daily/running.lock
. "$TREE/scripts/daily_lock.sh"

say() { echo "=== $(date '+%F %I:%M:%S %p') $*" | tee -a "$LOG"; }
# step NAME COMMAND...: run once, under the morning job's lock, a failure stops the side
step() {
  local name=$1; shift
  [ -f "$DONE/$name" ] && { say "skip $name: done before"; return 0; }
  until take_lock "the cleanup proof, side $S, $name"; do
    say "waiting: $(lock_owner) holds the lock"; sleep 60
  done
  say "start $name: $*"
  "$@" >> "$LOG" 2>&1
  local status=$?
  release_lock
  [ $status -eq 0 ] || { say "FAILED $name (exit $status)"; exit 1; }
  touch "$DONE/$name"
  say "done $name"
}
oo() { "$PY" -u -m openrapm "$@"; }

say "side $S ($SIDE) from $TREE"
# the guard lives with the proof's scripts in the cleanup checkout; PYTHONPATH decides which code it imports
"$PY" /a/code/spmm_cleanup/scratch/2026-11-16_cleanup/guard.py "$SIDE" >> "$LOG" 2>&1 \
  || { say "FAILED the guard"; exit 1; }

FOLDS=fade141_early,fade141_within
# 1. the test way, every finished season
step test_rate   oo rate --setting test --seasons 1997-2026 --name cu_${S}_test --jobs 4
step test_shrink oo shrink --base season_ratings_cu_${S}_test_step3 --setting test --folds $FOLDS --extrapolate-folds=
step test_swap   oo swap --base season_ratings_cu_${S}_test_step4 --setting test --folds $FOLDS --extrapolate-folds=
# 2. the published way, every finished season
step prod_rate   oo rate --seasons 1997-2026 --name cu_${S}_prod --jobs 4
step prod_shrink oo shrink --base season_ratings_cu_${S}_prod_step3
step prod_swap   oo swap --base season_ratings_cu_${S}_prod_step4
# 3. 2027 by the morning job's path, on copies of the published tables taken once for both sides (cu_src_morning)
copy_morning() {
  local from=$1 to=$2 suffix
  for suffix in "" _step3 _step4 _regular; do
    cp "$OUT/season_ratings_${from}${suffix}.parquet" "$OUT/season_ratings_${to}${suffix}.parquet" || return 1
  done
  cp "$OUT/csv/season_ratings_${from}_step4_multipliers.csv" "$OUT/csv/season_ratings_${to}_step4_multipliers.csv" \
    && cp "$OUT/csv/wins_${from}_seasons.csv" "$OUT/csv/wins_${to}_seasons.csv"
}
[ -f "$OUT/season_ratings_cu_src_morning.parquet" ] || copy_morning product cu_src_morning || { say "FAILED the copy"; exit 1; }
step morning_copy  copy_morning cu_src_morning cu_${S}_morning
step morning_build oo build --seasons 2027 --replace-seasons --name cu_${S}_morning
# 4. WOSA's regular-season ratings, five seasons across the eras
step wins oo wins --name cu_${S}_morning --seasons 1997,2006,2012,2020,2026 --rebuild-regular --jobs 2
# 5. held-back games on today's saved models
BASE=season_ratings_fade141_step3
step folds_plain  oo folds --models fade141 --base $BASE --seasons 2001,2020,2026 --tag cu_${S}_plain --jobs 3
step folds_cut10  oo folds --models fade141 --base $BASE --seasons 2001,2026 --tag cu_${S}_cut10 --cut 0.10 \
                     --hold-prior-scale $BASE
step folds_cut50  oo folds --models fade141 --base $BASE --seasons 2001,2026 --tag cu_${S}_cut50 --cut 0.5 \
                     --hold-prior-scale $BASE --blend-through 0.75 --blend-from 0.25
step folds_pieces oo folds --models fade141 --base $BASE --seasons 2026 --tag cu_${S}_pieces --repeats 1 --pieces
# 6. Box Adj's line, the shipped line's own command on two seasons
step line oo second-prior --models padbins127 --folds padbins127_early,padbins127_within --tag cu_${S}_sp \
            --inputs context_types_team_rates_zones_centred --target rating --seasons 2001,2026
# 7. the site's files, from the published tables
step site oo site --openrapm-decomposition --similarity --power --power-forecast --bias \
            --out-dir "$OUT/cu_${S}_site" --built 2026-11-16
say "side $S finished"
