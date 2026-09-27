#!/usr/bin/env bash
# -----------------------------------------------------------------------
# sweep_a51_vs_a45.sh - paired test: a51 (RNL + mid-mass) vs a45.
#
# Purpose (2026-05-27, follow-up to a45 N=10 audit DE33):
#   a45 has 1 persistent weakness - G1 recall (delta mean -4.70, loses
#   7/10 seeds). a51_rank_norm_mid_mass adds a complementary mid-rank
#   mean pool intended to recover G1 without destroying G0/G3 wins.
#
# Pre-registered decision rule (mirrors a51 docstring kill criterion):
#   WIN  -> median d(a51 - a45) macro_recall > +1.0
#           AND median d G1 recall > +2.0
#           AND median d G3 recall >= -2.0   over paired seeds
#   TIE  -> any of above misses by < 1.0
#   LOSE -> median d macro_recall <= 0  OR  destroys G3 by >= 2.0
#
# Default seed set = full N=10 paired against the a45 audit (same seeds
# as experiments/2026052[5-7]/...a45_rank_norm_lengthnorm_s*).
#
# Usage:
#   ./scripts/sweep_a51_vs_a45.sh                          # full N=10 (default)
#   STAGE=warmup ./scripts/sweep_a51_vs_a45.sh             # 5 seeds: 0,1,2,21,42
#   STAGE=extend ./scripts/sweep_a51_vs_a45.sh             # 5 seeds: 3,7,13,99,123
#   SEEDS="0 1 2 21 42" ./scripts/sweep_a51_vs_a45.sh      # custom seeds
#   WAIT_FOR_INFLIGHT=1 ./scripts/sweep_a51_vs_a45.sh      # wait for MPS first
#   RUN_COMPARE=0 ./scripts/sweep_a51_vs_a45.sh            # skip compare step
#   ATT="a51_rank_norm_mid_mass" ./scripts/sweep_a51_vs_a45.sh   # override attempt
#
# Seed selection rationale:
#   0   (a45 LOST joint, base BOTH)        <- stress test
#   1   (a45 LOST joint, base BOTH)        <- stress test
#   2   (a45 SPLIT, test only)             <- locked baseline seed
#   3   (a45 catastrophic test_qwk 0.563)  <- swing seed
#   7, 13, 99, 123                         <- a45 stable wins
#   21  (a45 BIGGEST joint win)            <- can a51 keep it?
#   42  (a45 medium joint win, G3 fragile) <- G3 generalisation
# -----------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

ATT="${ATT:-a51_rank_norm_mid_mass}"

# --- Seed selection ----------------------------------------------------
SEEDS_FULL="0 1 2 3 7 13 21 42 99 123"
SEEDS_WARMUP="0 1 2 21 42"
SEEDS_EXTEND="3 7 13 99 123"

STAGE="${STAGE:-full}"
case "$STAGE" in
  full)    SEEDS_DEFAULT="$SEEDS_FULL"   ;;
  warmup)  SEEDS_DEFAULT="$SEEDS_WARMUP" ;;
  extend)  SEEDS_DEFAULT="$SEEDS_EXTEND" ;;
  *)       echo "Unknown STAGE='$STAGE' (expected full|warmup|extend)" >&2; exit 2 ;;
esac
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"

WAIT_FOR_INFLIGHT="${WAIT_FOR_INFLIGHT:-0}"
RUN_COMPARE="${RUN_COMPARE:-1}"
TAG="a51_vs_a45_sweep_$(date +%Y%m%d_%H%M%S)"
LOG="runs/sweep_a51_vs_a45.log"

mkdir -p runs
log () { echo "$@" | tee -a "$LOG"; }

log ""
log "============================================================"
log "$TAG started at $(date)"
log "Attempt : $ATT"
log "Stage   : $STAGE"
log "Seeds   : $SEEDS"
log "Wait    : $WAIT_FOR_INFLIGHT"
log "Compare : RUN_COMPARE=$RUN_COMPARE"
log "============================================================"

# --- Optional: wait for any in-flight trainer/sweep to free MPS --------
if [ "$WAIT_FOR_INFLIGHT" = "1" ]; then
  log ""
  log "Waiting for in-flight trainer / sibling sweep to finish..."
  waited=0
  while true; do
    pids=$(pgrep -f "train_grading_reti.py|sweep_a51_vs_a45\.sh" 2>/dev/null \
             | grep -v "^$$\$" || true)
    if [ -z "$pids" ]; then
      log "  MPS free after waiting ${waited}s."
      break
    fi
    if [ $((waited % 60)) -eq 0 ]; then
      log "  waiting ${waited}s; alive: $(echo $pids | tr '\n' ' ')"
    fi
    sleep 30
    waited=$((waited + 30))
  done
fi

# --- Run each seed -----------------------------------------------------
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

# --- Paired comparison -------------------------------------------------
if [ "$RUN_COMPARE" = "1" ]; then
  log ""
  log "============================================================"
  log "Sweep complete. Running a51 vs a45 paired comparison..."
  log "============================================================"
  python scripts/compare_a51_vs_a45.py 2>&1 | tee -a "$LOG"
fi

log ""
log "Done at $(date)."
