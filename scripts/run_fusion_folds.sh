#!/usr/bin/env bash
# Decisive backbone-fusion test, evaluated PAIRED across patient folds.
# For each seed (= patient fold), run BOTH:
#   a68_fused_gated_mil   on data_fused (Virchow2+UNI2-h = 2816-d)
#   a69_single_gated_mil  on data       (Virchow2 = 1280-d, the ablation)
# Same aggregator (gated attention), only the input space differs -> isolates fusion.
#
# Usage:  scripts/run_fusion_folds.sh <npar> <seed1> [seed2 ...]
# Example: scripts/run_fusion_folds.sh 4 0 1 2 3 42
#
# Each run -> its own experiment dir (unique --prefix) + log runs/fusion_<id>_s<seed>.out.
# No --leaderboard (won't pollute results/).
set -u
NPAR="${1:?usage: <npar> <seed...>}"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p runs
echo "=== fusion folds: seeds=[$*] npar=$NPAR ==="

# emit "<novelty_id> <data_root>" x "<seed>" tuples
for s in "$@"; do
  echo "a68_fused_gated_mil data_fused $s"
  echo "a69_single_gated_mil data $s"
done | xargs -P "$NPAR" -I {} sh -c '
  set -- $1
  id="$1"; droot="$2"; seed="$3"
  echo "[start $(date +%H:%M:%S)] $id s$seed (data_root=$droot)"
  python src/train_grading_reti.py \
    --backbone virchow2 --data_root "$droot" --epochs 50 --lr 1e-4 --batch_size 1 \
    --num_workers 4 --topk 0 --early_stop_patience 15 \
    --formulation regression --main_metric qwk --device cpu \
    --seed "$seed" --model_type novelty_attempt --novelty_id "$id" \
    --prefix "fusion_${id}_s${seed}" \
    > "runs/fusion_${id}_s${seed}.out" 2>&1 \
    && echo "[done  $(date +%H:%M:%S)] $id s$seed" || echo "[FAIL ] $id s$seed"
' _ {}
echo "=== ALL FUSION-FOLD RUNS DONE ==="
