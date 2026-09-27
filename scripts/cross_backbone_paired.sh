#!/usr/bin/env bash
# Rigorous cross-backbone breadth test: warm-start (a115, PER-FOLD-MATCHED axis)
# vs plain (a114) across folds {0,1,2,3,42} on {virchow2, uni2, titan}.
#
# a115 reads --seed from argv and loads that fold's OWN train-only fibrosis axis,
# so this is leakage-safe AND a fair paired test of whether the warm-start helps
# BROADLY (across folds + backbones) or is only a seed=2 artifact.
#
# Usage: scripts/cross_backbone_paired.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p runs
SEEDS="0 1 2 3 42"
PAIRS="a115_warmgated_perfold a114_warmgated_xbb_off"   # warm  plain
BACKBONES="virchow2 uni2 titan"

echo "=== cross-backbone PAIRED test: warm(a115 per-fold) vs plain(a114) on {$BACKBONES} x folds {$SEEDS} ==="
JOBS=$(mktemp)
for bb in $BACKBONES; do for s in $SEEDS; do for id in $PAIRS; do echo "$bb $s $id" >> "$JOBS"; done; done; done

cat "$JOBS" | xargs -P 6 -L 1 sh -c '
  bb="$1"; s="$2"; id="$3"
  out="runs/xbbp_${bb}_${id}_s${s}.out"
  echo "[start $(date +%H:%M:%S)] $bb s$s / $id"
  python src/train_grading_reti.py \
    --backbone "$bb" --data_root data --epochs 50 --lr 1e-4 --batch_size 1 \
    --num_workers 4 --topk 0 --early_stop_patience 15 \
    --formulation regression --main_metric qwk --device cpu \
    --seed "$s" --model_type novelty_attempt --novelty_id "$id" \
    --prefix "xbbp_${bb}_${id}_s${s}" \
    > "$out" 2>&1 \
    && echo "[done  $(date +%H:%M:%S)] $bb s$s / $id" || echo "[FAIL ] $bb s$s / $id"
' _
rm -f "$JOBS"
echo "=== CROSS-BACKBONE PAIRED RUNS DONE ==="