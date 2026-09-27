#!/usr/bin/env bash
set -u
cd /Users/thanadetch/projects/interpretable-mpn-diagnosis
mkdir -p logs
for mode in mean quantile topk_mean; do
  log="logs/r21_ppsp_${mode}_s2.log"
  echo ">>> RUN ppsp mode=$mode seed=2 -> $log"
  PYTHONPATH=src python src/train_grading_reti.py \
    --backbone virchow2 --data_root data \
    --model_type patch_score_pool --score_pool_mode "$mode" \
    --seed 2 --formulation regression --main_metric qwk \
    --epochs 50 --lr 1e-4 --batch_size 1 --early_stop_patience 15 \
    --num_workers 4 --topk 0 --prefix "r21_ppsp_${mode}_s2" --leaderboard \
    > "$log" 2>&1
  echo "    mode=$mode exit=$? $(grep -aE 'Best Val QWK|Test QWK' "$log" | tail -2 | tr '\n' ' ')"
done
echo "ALL PPSP MODES DONE"
