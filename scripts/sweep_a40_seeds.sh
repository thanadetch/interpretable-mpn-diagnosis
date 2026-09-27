#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# sweep_a40_seeds.sh — re-run a40_mean_over_queries at multiple seeds.
#
# Purpose (2026-05-26):
#   a40 at seed=2 produced val_qwk 0.8085 (best epoch 1) / test_qwk 0.9570.
#   Passes the test gate (>0.9476) but fails the val gate (>0.8182) by
#   ~2 ROIs. Best-epoch-1 selection is a known instability pattern (also
#   seen in a29/a35). This sweep answers a single yes/no question:
#       Is a40's val ~0.808 a stable signal across seeds,
#       or a one-seed lottery from picking a near-init checkpoint?
#
#   Decision rule (pre-registered before looking at results):
#     - median val_qwk across {0, 1, 2, 3, 42} > 0.8182 -> a40 is a real win.
#       (Update reporting to mean±std across seeds; close the search.)
#     - median val_qwk <= 0.80 -> a40 is epoch-1 lottery; close H20 family;
#       proceed to a49 multi-scale or pivot narrative to "near ceiling".
#     - in between -> Pareto / inconclusive; pivot to narrative.
#
# Usage:
#   ./scripts/sweep_a40_seeds.sh                  # seeds 0 1 3 42 (skips 2)
#   SEEDS="0 1 3 42" ./scripts/sweep_a40_seeds.sh
#
# Notes:
#   - SEED=2 is intentionally skipped (already done; row in leaderboard).
#   - STOP_ON_WIN=0 so we always run all seeds even if one beats the gate.
#   - ENFORCE_BUCKET_RULE=0 to bypass the playbook §0 rule 7 pre-flight
#     (this is a seed sweep of an existing module, not a new attempt).
#   - Each run still appends a row to results/leaderboard.csv.
#   - Per-run stdout/stderr goes to runs/a40_mean_over_queries_s<seed>.out
#     (loop_runner writes to runs/<attempt>.out, so we mv it per seed).
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

ATT="a40_mean_over_queries"
SEEDS_DEFAULT="0 1 3 42"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
TAG="a40_seed_sweep_$(date +%Y%m%d_%H%M%S)"

mkdir -p runs

log () { echo "$@" | tee -a "runs/sweep_a40.log"; }

log ""
log "============================================================"
log "$TAG started at $(date)"
log "Attempt : $ATT"
log "Seeds   : $SEEDS"
log "============================================================"

for s in $SEEDS; do
  log ""
  log "----- SEED=$s -----"
  SEED="$s" \
  STOP_ON_WIN=0 \
  ENFORCE_BUCKET_RULE=0 \
  TAG="$TAG seed=$s" \
    ./scripts/loop_runner.sh "$ATT"
  STATUS=$?

  # loop_runner writes to runs/<attempt>.out and overwrites it each call.
  # Archive a per-seed copy so we don't lose the previous one.
  if [ -f "runs/${ATT}.out" ]; then
    cp "runs/${ATT}.out" "runs/${ATT}_s${s}.out"
    log "  archived stdout -> runs/${ATT}_s${s}.out"
  fi
  log "  loop_runner exit status: $STATUS"
done

log ""
log "============================================================"
log "Sweep complete. Summarising..."
log "============================================================"
python scripts/summarize_a40_sweep.py

