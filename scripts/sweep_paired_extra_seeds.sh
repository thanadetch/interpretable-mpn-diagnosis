#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# sweep_paired_extra_seeds.sh — run BOTH SimpleGatedMIL baseline AND a45 at
# 5 new seeds, paired, so the leaderboard ends up with 10 paired seeds total
# (existing: {0, 1, 2, 3, 42}; new: {7, 13, 21, 99, 123}).
#
# Purpose (2026-05-27, follow-up to a45 seed sweep DE32):
#   The first paired audit (5 seeds, 4/5 paired) gave INDISTINCT/lose:
#       a45 joint-win count = 0/5 (test always <= baseline at same seed)
#   The val-test anti-correlation pattern (DE31) means a 5-seed sample
#   has high variance on per-seed joint-win counts. Doubling N to 10
#   tightens the distribution and rules out the "we just got unlucky"
#   counter-narrative for the writeup.
#
#   Same decision rule as sweep_a45_seeds.sh, scaled to N=10:
#       >=6/10 joint wins -> a45 is a real per-seed winner.
#       3-5/10 joint wins  -> Pareto / inconclusive.
#       <=2/10 joint wins  -> indistinguishable / a45 loses.
#
# Usage:
#   ./scripts/sweep_paired_extra_seeds.sh
#   SEEDS="7 13 21 99 123" ./scripts/sweep_paired_extra_seeds.sh
#
# Notes:
#   - WAITS for any in-flight a45 or paired sweep to finish before starting,
#     to avoid MPS contention.
#   - Each baseline run appends a row to results/leaderboard_v2.csv.
#   - Each a45 run goes through loop_runner.sh (appends to leaderboard
#     and writes stdout to runs/a45_rank_norm_lengthnorm.out; we archive
#     a per-seed copy to runs/a45_rank_norm_lengthnorm_s<seed>.out).
#   - ENFORCE_BUCKET_RULE=0 and STOP_ON_WIN=0 for the a45 calls.
#   - When all 10 paired runs (5 new seeds x 2 models) are done, runs the
#     paired comparison script.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=src

DEVICE="${DEVICE:-mps}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-50}"
SEEDS_DEFAULT="7 13 21 99 123"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
ATT="a45_rank_norm_lengthnorm"
TAG="paired_extra_$(date +%Y%m%d_%H%M%S)"

mkdir -p runs

log () { echo "$@" | tee -a "runs/sweep_paired_extra.log"; }

# ─── 1. Wait for any in-flight trainer / sweep to free MPS ───────────────
wait_for_free_mps () {
  local waited=0
  while true; do
    local pids
    pids=$(pgrep -f "train_grading_reti.py|sweep_a45_seeds\.sh|sweep_paired_extra_seeds\.sh" \
            | grep -v "^$$\$" || true)
    # Filter ourselves (this script's own pid).
    pids=$(echo "$pids" | awk -v me=$$ '$1 != me')
    if [ -z "$pids" ]; then
      log "  MPS free after waiting ${waited}s."
      return 0
    fi
    if [ $((waited % 60)) -eq 0 ]; then
      log "  waiting for in-flight processes (waited ${waited}s): $(echo $pids | tr '\n' ' ')"
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

log ""
log "============================================================"
log "$TAG started at $(date)"
log "Seeds   : $SEEDS"
log "Models  : baseline (simple) + a45 ($ATT)"
log "Config  : epochs=$EPOCHS device=$DEVICE workers=$NUM_WORKERS"
log "============================================================"

log ""
log "Step 1/3: Waiting for any in-flight trainer to finish..."
wait_for_free_mps

# ─── 2. For each new seed: baseline first, then a45 ──────────────────────
for s in $SEEDS; do
  log ""
  log "############### SEED=$s ###############"

  # ----- baseline (simple) -----
  POSTFIX_BASE="simple_virchow2_regression_s${s}"
  EXISTING=$(ls -td "experiments/reti_${POSTFIX_BASE}_"* \
                     "experiments/"*"/reti_${POSTFIX_BASE}_"* 2>/dev/null | head -1)
  if [ -n "$EXISTING" ] && [ -f "$EXISTING/test_metrics.json" ]; then
    log "  baseline SKIP (already done at $EXISTING)"
  else
    log "  --- baseline (simple) seed=$s ---"
    python src/train_grading_reti.py \
        --backbone virchow2 --data_root data \
        --epochs "$EPOCHS" --lr 1e-4 --batch_size 1 --seed "$s" \
        --num_workers "$NUM_WORKERS" --topk 0 \
        --early_stop_patience 15 \
        --formulation regression --main_metric qwk \
        --device "$DEVICE" \
        --model_type simple \
        --postfix "$POSTFIX_BASE" \
        --leaderboard \
        > "runs/baseline_${POSTFIX_BASE}.out" 2>&1
    STATUS=$?
    log "    trainer exit=$STATUS -> runs/baseline_${POSTFIX_BASE}.out"
  fi

  # ----- a45 -----
  POSTFIX_A45="${ATT}_s${s}"
  EXISTING_A45=$(ls -td "experiments/"*"/reti_novelty_attempt_virchow2_${POSTFIX_A45}_"* 2>/dev/null | head -1)
  if [ -n "$EXISTING_A45" ] && [ -f "$EXISTING_A45/test_metrics.json" ]; then
    log "  a45 SKIP (already done at $EXISTING_A45)"
  else
    log "  --- a45 ($ATT) seed=$s ---"
    SEED="$s" \
    STOP_ON_WIN=0 \
    ENFORCE_BUCKET_RULE=0 \
    TAG="$TAG seed=$s" \
      ./scripts/loop_runner.sh "$ATT"
    STATUS=$?
    if [ -f "runs/${ATT}.out" ]; then
      cp "runs/${ATT}.out" "runs/${ATT}_s${s}.out"
      log "    archived stdout -> runs/${ATT}_s${s}.out"
    fi
    log "    loop_runner exit=$STATUS"
  fi
done

# ─── 3. Paired comparison ────────────────────────────────────────────────
log ""
log "============================================================"
log "Step 3/3: Re-running paired comparison (10 seeds expected)..."
log "============================================================"
python scripts/compare_a45.py 2>&1 | tee -a runs/sweep_paired_extra.log

log ""
log "Done at $(date)."


