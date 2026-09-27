#!/usr/bin/env bash
# a158 in MULTI-CLASS (classification) on all 3 backbones, parallel.
# Matches the multi-class ablation config: formulation=classification, main_metric=macro_recall.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/a158_multiclass
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"; : > "$MASTER"
NW="${NW:-2}"

run_one () {
  local bb="$1" pfx="$2"
  echo "=== $bb START ===" >> "$MASTER"
  python src/train_grading_reti.py --backbone "$bb" --data_root data \
    --model_type novelty_attempt --novelty_id a158_diffuse_temperature_gated \
    --formulation classification --main_metric macro_recall --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers "$NW" \
    --prefix "$pfx" --postfix multiclass > "$LOGDIR/${pfx}.log" 2>&1
  local ec=$?
  local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  echo "=== $bb DONE exit=$ec dir=$d ===" >> "$MASTER"
}

run_one uni2     r49_a158cls_uni2_s2 &
run_one virchow2 r49_a158cls_virchow2_s2 &
run_one titan    r49_a158cls_titan_s2 &
wait
echo "=== A158 MULTICLASS COMPLETE ===" >> "$MASTER"
