#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/a163_Tsweep; mkdir -p "$LOGDIR"; MASTER="$LOGDIR/master.log"; : > "$MASTER"
run_one(){ local T="$1"; local pfx="r51_a163T${T/./p}_titan_s2"
  echo "=== T=$T START ===" >> "$MASTER"
  A163_T="$T" python src/train_grading_reti.py --backbone titan --data_root data \
    --model_type novelty_attempt --novelty_id a163_fixed_temperature \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$pfx" > "$LOGDIR/T${T}.log" 2>&1
  local ec=$?; local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$T" "$ec" "$d" >> "$MASTER" 2>&1 <<'PY'
import sys,json; import pandas as pd
T,ec,d=sys.argv[1],sys.argv[2],sys.argv[3]
try:
    v=float(pd.read_csv(d+"training_log.csv").val_qwk.max()); m=json.load(open(d+"test_metrics.json"))
    print(f"=== T={T} DONE exit={ec} val={v:.4f} test_qwk={m['test_qwk']:.4f} acc={m['test_accuracy']:.2f} ===")
except Exception as e: print(f"=== T={T} DONE exit={ec} ERR {e} ===")
PY
}
for T in 1.0 1.5 2.0 3.0; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge 4 ]; do sleep 4; done
  run_one "$T" &
done
wait
echo "=== SWEEP COMPLETE (cf baseline simple titan 0.9584, mean_pool titan 0.9253) ===" >> "$MASTER"
