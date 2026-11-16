#!/usr/bin/env bash
# Wait for the before side to end, then start the after side in the background; says which happened.
LOG=/a/code/spmm/outputs/cleanup_proof/side_b.log
START=$(wc -l < "$LOG")   # only lines written after this waiter starts count
until tail -n +"$((START + 1))" "$LOG" | grep -q -E 'side b finished|FAILED'; do sleep 30; done
if tail -n +"$((START + 1))" "$LOG" | grep -q 'FAILED'; then
  echo "the before side FAILED: $(grep FAILED "$LOG" | tail -1)"
  exit 1
fi
cd /a/code/spmm_cleanup || exit 1
nohup bash scratch/2026-11-16_cleanup/side.sh a > /a/code/spmm/outputs/cleanup_proof/side_a.nohup 2>&1 &
sleep 60
echo "before side finished at $(grep 'side b finished' "$LOG" | cut -c5-26); after side started:"
head -3 /a/code/spmm/outputs/cleanup_proof/side_a.log
