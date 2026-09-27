#!/usr/bin/env bash
set -u
cd /Users/thanadetch/projects/interpretable-mpn-diagnosis
mkdir -p logs
for nid in a135_boneaware_calibrated_perfold a136_boneaware_calibrated_perfold_off; do
  short=$(echo "$nid" | grep -oE '^a[0-9]+')
  for s in 0 1 2 3 42; do
    log="logs/r20_${short}_s${s}.log"
    echo ">>> RUN $short seed=$s -> $log"
    PYTHONPATH=src python src/train_grading_reti.py \
      --backbone virchow2 --data_root data_bonefib \
      --model_type novelty_attempt --novelty_id "$nid" \
      --seed "$s" --formulation regression --main_metric qwk \
      --epochs 50 --lr 1e-4 --batch_size 1 --early_stop_patience 15 \
      --num_workers 4 --topk 0 --prefix "r20_${short}_s${s}" --leaderboard \
      > "$log" 2>&1
    echo "    exit=$? $(grep -hiE 'Best Val QWK|Test QWK' "$log" | tail -2 | tr '\n' ' ')"
  done
done
echo "ALL CROSSFOLD RUNS DONE"
