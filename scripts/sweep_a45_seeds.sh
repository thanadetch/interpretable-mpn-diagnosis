#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# sweep_a45_seeds.sh — re-run a45_rank_norm_lengthnorm at multiple seeds.
#
# Purpose (2026-05-26, follow-up to a40 & baseline seed sweeps):
#   a45 at seed=2 produced val_qwk 0.8084 / test_qwk 0.9501 (best epoch 3).
#   This is the strongest thesis-narrative candidate (3-axis ablation,
#   smallest param count, best epoch >1 i.e. not init-checkpoint lottery).
#
#   This sweep + the paired baseline sweep already on disk answer one
#   defensible question:
#       At each seed in {0, 1, 2, 3, 42}, does a45 beat baseline?
#
#   Paired comparison is fair: same seed -> same data shuffle, same init
#   noise scale, same val/test cohorts. The aggregator is the only
#   independent variable per seed pair.
#
# Decision rule (pre-registered; mirrors a40 sweep refinements):
#   ≥3/5 seeds: a45 wins BOTH val_qwk and test_qwk vs baseline at the
#     same seed -> a45 is a real per-seed winner, defensible for thesis.
#   2/5 seeds joint-win -> Pareto / inconclusive; report both.
#   ≤1/5 seeds joint-win -> a45 is also seed-fragile; report as
#     "indistinguishable from baseline under val-cohort instability".
#
# Usage:
#   ./scripts/sweep_a45_seeds.sh                  # seeds 0 1 3 42 (skips 2)
#   SEEDS="0 1 3 42" ./scripts/sweep_a45_seeds.sh
#
# Notes:
#   - SEED=2 is intentionally skipped (already done; row in leaderboard_v2).
#   - STOP_ON_WIN=0 so we always run all seeds even if one beats the gate.
#   - ENFORCE_BUCKET_RULE=0 to bypass the playbook §0 rule 7 pre-flight.
#   - Each run still appends a row to results/leaderboard.csv (+_v2).
#   - Per-seed stdout archived to runs/a45_rank_norm_lengthnorm_s<seed>.out.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

ATT="a45_rank_norm_lengthnorm"
SEEDS_DEFAULT="0 1 3 42"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
TAG="a45_seed_sweep_$(date +%Y%m%d_%H%M%S)"

mkdir -p runs

log () { echo "$@" | tee -a "runs/sweep_a45.log"; }

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

  if [ -f "runs/${ATT}.out" ]; then
    cp "runs/${ATT}.out" "runs/${ATT}_s${s}.out"
    log "  archived stdout -> runs/${ATT}_s${s}.out"
  fi
  log "  loop_runner exit status: $STATUS"
done

log ""
log "============================================================"
log "Sweep complete. Running paired comparison vs baseline..."
log "============================================================"
python scripts/compare_a45.py


