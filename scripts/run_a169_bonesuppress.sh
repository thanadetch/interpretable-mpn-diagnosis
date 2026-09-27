#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/a169_bonesuppress; mkdir -p "$LOGDIR"; MASTER="$LOGDIR/master.log"; : > "$MASTER"
declare -A BASE=( [titan]=0.9584 [virchow2]=0.9476 [uni2]=0.9418 )
run_one(){ local bb="$1"; local pfx="r56_a169_${bb}_s2"
  echo "=== $bb START ===" >> "$MASTER"
  python src/train_grading_reti.py --backbone "$bb" --data_root data \
    --model_type novelty_attempt --novelty_id a169_bone_suppressed_gated \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$pfx" > "$LOGDIR/${bb}.log" 2>&1
  local ec=$?; local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$bb" "$ec" "$d" "${BASE[$bb]}" >> "$MASTER" 2>&1 <<'PY'
import sys,json; import pandas as pd
bb,ec,d,base=sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4])
try:
    v=float(pd.read_csv(d+"training_log.csv").val_qwk.max()); m=json.load(open(d+"test_metrics.json"))
    t=float(m["test_qwk"]); acc=float(m["test_accuracy"]); mr=float(m["test_macro_recall"])
    print(f"=== {bb} DONE exit={ec} val={v:.4f} test={t:.4f}(base {base:.4f}, {'+' if t>=base else ''}{t-base:+.4f}) acc={acc:.2f} mr={mr:.2f} ===")
except Exception as e: print(f"=== {bb} DONE exit={ec} ERR {e} ===")
PY
}
for bb in titan virchow2 uni2; do run_one "$bb" & done
wait
echo "=== A169 COMPLETE ===" >> "$MASTER"
