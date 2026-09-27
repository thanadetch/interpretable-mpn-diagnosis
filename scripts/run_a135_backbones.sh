#!/usr/bin/env bash
set -u; cd /Users/thanadetch/projects/interpretable-mpn-diagnosis; mkdir -p logs
for bb in uni2 titan; do
  log="logs/r25_a135_${bb}_s2.log"
  echo ">>> RUN a135 @ ${bb} seed=2 -> $log"
  PYTHONPATH=src python src/train_grading_reti.py \
    --backbone "$bb" --data_root data_bonefib \
    --model_type novelty_attempt --novelty_id a135_boneaware_calibrated_perfold \
    --seed 2 --formulation regression --main_metric qwk \
    --epochs 50 --lr 1e-4 --batch_size 1 --early_stop_patience 15 \
    --num_workers 4 --topk 0 --prefix "r25_a135_${bb}_s2" --leaderboard \
    > "$log" 2>&1
  echo "    ${bb} exit=$? $(grep -aE 'Best Val QWK|Test QWK' "$log" | tail -2 | tr '\n' ' ')"
done
echo "ALL A135 BACKBONES DONE"
