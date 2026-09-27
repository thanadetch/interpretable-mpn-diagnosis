#!/usr/bin/env bash
set -u; cd /Users/thanadetch/projects/interpretable-mpn-diagnosis; mkdir -p logs
for nid in a138_spatial_network_structure a139_spatial_network_structure_off; do
  short=$(echo "$nid"|grep -oE '^a[0-9]+')
  log="logs/r26_${short}_virchow2_s2.log"
  echo ">>> RUN $short @ virchow2 seed=2 -> $log"
  PYTHONPATH=src python src/train_grading_reti.py \
    --backbone virchow2 --data_root data_struct \
    --model_type novelty_attempt --novelty_id "$nid" \
    --seed 2 --formulation regression --main_metric qwk \
    --epochs 50 --lr 1e-4 --batch_size 1 --early_stop_patience 15 \
    --num_workers 4 --topk 0 --prefix "r26_${short}_virchow2_s2" --leaderboard \
    > "$log" 2>&1
  echo "    $short exit=$? $(grep -aE 'Best Val QWK|Test QWK' "$log"|tail -2|tr '\n' ' ')"
done
echo "GATE1 DONE"
