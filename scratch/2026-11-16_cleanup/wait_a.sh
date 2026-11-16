#!/usr/bin/env bash
# Wait for the after side to end, then compare the two sides file by file and run the after side's extras.
LOG=/a/code/spmm/outputs/cleanup_proof/side_a.log
until grep -q -E 'side a finished|FAILED' "$LOG"; do sleep 60; done
if grep -q 'FAILED' "$LOG"; then
  echo "the after side FAILED: $(grep FAILED "$LOG" | tail -1)"
  exit 1
fi
cd /a/code/spmm_cleanup || exit 1
/a/code/spmm/.venv/Scripts/python scratch/2026-11-16_cleanup/compare_sides.py > /a/code/spmm/outputs/cleanup_proof/compare.txt 2>&1
echo "compare exit $?"
grep -v '^same ' /a/code/spmm/outputs/cleanup_proof/compare.txt | head -40
bash scratch/2026-11-16_cleanup/extras.sh
grep -E '^==|exit|refus|archive|reproduction' /a/code/spmm/outputs/cleanup_proof/extras.log | cut -c1-300
