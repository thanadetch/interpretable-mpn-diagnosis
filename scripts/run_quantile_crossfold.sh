#!/usr/bin/env bash
set -u
cd /Users/thanadetch/projects/interpretable-mpn-diagnosis
mkdir -p logs
for s in 0 1 3 42; do
  log="logs/r23_ppsp_quantile_s${s}.log"
  echo ">>> RUN quantile seed=$s -> $log"
  PYTHONPATH=src python src/train_grading_reti.py \
    --backbone virchow2 --data_root data \
    --model_type patch_score_pool --score_pool_mode quantile --quantile 0.75 \
    --seed "$s" --formulation regression --main_metric qwk \
    --epochs 50 --lr 1e-4 --batch_size 1 --early_stop_patience 15 \
    --num_workers 4 --topk 0 --prefix "r23_ppsp_quantile_s${s}" --leaderboard \
    > "$log" 2>&1
  echo "    quantile s${s} exit=$? $(grep -aE 'Best Val QWK|Test QWK' "$log" | tail -2 | tr '\n' ' ')"
done
echo "ALL QUANTILE CROSSFOLD DONE"
