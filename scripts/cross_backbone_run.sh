#!/usr/bin/env bash
# Cross-backbone breadth test of the seed-2 winner (warm-started gated attention).
#
# Advisor requirement: a real improvement must be BROAD (helps across backbones),
# not a single-model lottery. The warm-start (a113) cleared the seed=2 gate on
# Virchow2. This runs warm (a113) vs plain (a114 = SimpleGatedMIL) on ALL three
# backbones at seed=2 (axis + split both seed=2 -> train-only, no leakage), so we
# can read Delta = warm - plain per backbone.
#
# Usage: scripts/cross_backbone_run.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p runs
SEED=2
PAIRS="a113_warmgated_xbb a114_warmgated_xbb_off"   # warm  plain
BACKBONES="virchow2 uni2 titan"

echo "=== cross-backbone breadth test (seed=$SEED): warm(a113) vs plain(a114) on {$BACKBONES} ==="
# build the job list: one line per (backbone, module)
JOBS=$(mktemp)
for bb in $BACKBONES; do
  for id in $PAIRS; do
    echo "$bb $id" >> "$JOBS"
  done
done

cat "$JOBS" | xargs -P 6 -L 1 sh -c '
  bb="$1"; id="$2"
  echo "[start $(date +%H:%M:%S)] $bb / $id"
  python src/train_grading_reti.py \
    --backbone "$bb" --data_root data --epochs 50 --lr 1e-4 --batch_size 1 \
    --num_workers 4 --topk 0 --early_stop_patience 15 \
    --formulation regression --main_metric qwk --device cpu \
    --seed '"$SEED"' --model_type novelty_attempt --novelty_id "$id" \
    --prefix "xbb_${bb}_${id}_s'"$SEED"'" \
    > "runs/xbb_${bb}_${id}_s'"$SEED"'.out" 2>&1 \
    && echo "[done  $(date +%H:%M:%S)] $bb / $id" || echo "[FAIL ] $bb / $id"
' _
rm -f "$JOBS"
echo "=== CROSS-BACKBONE RUNS DONE (seed=$SEED) ==="