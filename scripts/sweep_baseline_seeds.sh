#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# sweep_baseline_seeds.sh — re-run SimpleGatedMIL locked baseline at multiple seeds.
#
# Purpose (2026-05-26, follow-up to a40 seed sweep):
#   The a40 seed sweep (scripts/sweep_a40_seeds.sh) revealed that a40 is
#   severely seed-fragile: across seeds {0, 1, 2, 3, 42} val_qwk std = 0.049
#   but test_qwk std = 0.127, with val and test ANTI-correlated. seed=2's
#   val=0.808 / test=0.957 looked Pareto-coherent but every other seed
#   showed strong val-overfit (val>0.87) -> test-collapse (test in [0.64,
#   0.89]) on the same 214-ROI val cohort.
#
#   The locked baseline (`simple + virchow2 + regression`) was characterised
#   ONLY at seed=2 (val 0.8182 / test 0.9476). We don't yet know whether
#   that result is itself a lottery from seed=2, OR whether SimpleGatedMIL
#   is genuinely more stable than a40 under the same val cohort. The
#   answer determines the entire defense narrative:
#
#     - If baseline is ALSO seed-fragile in the same direction:
#       single-seed comparison protocol is invalid for *any* aggregator on
#       this dataset; we must switch reporting to mean±std across seeds.
#
#     - If baseline is STABLE while a40 is fragile:
#       this becomes the headline finding -- "high-capacity novelties
#       overfit a 214-ROI val cohort; SimpleGatedMIL is the principled
#       choice for the locked split" -- defensible at the proposal exam.
#
# Decision rule (pre-registered before any results are seen):
#   median val_qwk and median test_qwk both stay within 0.04 of seed=2's
#   (0.8182, 0.9476) AND test std < 0.04 across seeds ->
#       STABLE. The single-seed baseline reporting in the proposal is
#       defensible as long as we report mean+/-std somewhere in §4/§5.
#   test std >= 0.08 (a40-like) ->
#       UNSTABLE. The single-seed protocol is broken for this dataset;
#       update the proposal to report median+/-IQR across seeds for ALL
#       comparisons (baseline + every novelty already audited).
#   in between ->
#       MARGINAL. Report both single-seed and multi-seed numbers; note
#       the limitation explicitly.
#
# Usage:
#   ./scripts/sweep_baseline_seeds.sh                  # seeds 0 1 3 42 (skips 2)
#   SEEDS="0 1 3 42" ./scripts/sweep_baseline_seeds.sh
#
# Notes:
#   - SEED=2 is intentionally skipped (locked baseline already characterised).
#   - Each run still appends a row to results/leaderboard_v2.csv.
#   - Per-run stdout/stderr goes to runs/baseline_simple_virchow2_s<seed>.out.
#   - Uses --leaderboard so the row lands in leaderboard_v2.csv (sidecar
#     written by the trainer when `simple` schema differs from legacy).
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=src

DEVICE="${DEVICE:-mps}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-50}"
SEEDS_DEFAULT="0 1 3 42"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
TAG="baseline_seed_sweep_$(date +%Y%m%d_%H%M%S)"

mkdir -p runs

log () { echo "$@" | tee -a "runs/sweep_baseline.log"; }

log ""
log "============================================================"
log "$TAG started at $(date)"
log "Model   : simple (SimpleGatedMIL, locked baseline head)"
log "Backbone: virchow2"
log "Seeds   : $SEEDS"
log "Config  : epochs=$EPOCHS device=$DEVICE workers=$NUM_WORKERS"
log "============================================================"

for s in $SEEDS; do
  POSTFIX="simple_virchow2_regression_s${s}"

  # Skip if a successful run for this seed already exists.
  EXISTING=$(ls -td "experiments/reti_${POSTFIX}_"* \
                     "experiments/"*"/reti_${POSTFIX}_"* 2>/dev/null | head -1)
  if [ -n "$EXISTING" ] && [ -f "$EXISTING/test_metrics.json" ]; then
    log ""
    log "----- SEED=$s : SKIP (already done at $EXISTING) -----"
    continue
  fi

  log ""
  log "----- SEED=$s -----"
  python src/train_grading_reti.py \
      --backbone virchow2 --data_root data \
      --epochs "$EPOCHS" --lr 1e-4 --batch_size 1 --seed "$s" \
      --num_workers "$NUM_WORKERS" --topk 0 \
      --early_stop_patience 15 \
      --formulation regression --main_metric qwk \
      --device "$DEVICE" \
      --model_type simple \
      --postfix "$POSTFIX" \
      --leaderboard \
      > "runs/baseline_${POSTFIX}.out" 2>&1
  STATUS=$?
  log "  trainer exit status: $STATUS  ->  runs/baseline_${POSTFIX}.out"

  if [ $STATUS -eq 0 ]; then
    EXP_DIR=$(ls -td "experiments/reti_${POSTFIX}_"* \
                       "experiments/"*"/reti_${POSTFIX}_"* 2>/dev/null | head -1)
    log "  exp_dir: $EXP_DIR"
  fi
done

log ""
log "============================================================"
log "Sweep complete. Summarising..."
log "============================================================"
python scripts/summarize_baseline_sweep.py

