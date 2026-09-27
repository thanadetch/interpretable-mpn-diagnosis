#!/usr/bin/env bash
set -u
cd /Users/thanadetch/projects/interpretable-mpn-diagnosis
mkdir -p logs
for trim in 0.05 0.10 0.20 0.30; do
  export A137_TRIM="$trim"
  tag=$(echo "$trim" | tr -d '.')
  log="logs/r22_a137_trim${tag}_s2.log"
  echo ">>> RUN a137 trim=$trim seed=2 -> $log"
  PYTHONPATH=src A137_TRIM="$trim" python src/train_grading_reti.py \
    --backbone virchow2 --data_root data \
    --model_type novelty_attempt --novelty_id a137_trimmed_score_pool \
    --seed 2 --formulation regression --main_metric qwk \
    --epochs 50 --lr 1e-4 --batch_size 1 --early_stop_patience 15 \
    --num_workers 4 --topk 0 --prefix "r22_a137_trim${tag}_s2" --leaderboard \
    > "$log" 2>&1
  echo "    trim=$trim exit=$? $(grep -aE 'Best Val QWK|Test QWK' "$log" | tail -2 | tr '\n' ' ')"
done
echo "ALL A137 TRIM SWEEP DONE"
