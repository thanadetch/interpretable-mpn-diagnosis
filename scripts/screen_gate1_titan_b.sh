#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/gate1_titan_b; mkdir -p "$LOGDIR"; MASTER="$LOGDIR/master.log"; : > "$MASTER"
NW="${NW:-2}"
JOBS=("a161_fibrosis_gated_attention r50_a161_titan_s2" "a162_blended_axis_attention r50_a162_titan_s2")
run_one(){ local nid="$1" pfx="$2"
  echo "=== $nid START ===" >> "$MASTER"
  python src/train_grading_reti.py --backbone titan --data_root data \
    --model_type novelty_attempt --novelty_id "$nid" \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers "$NW" \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  local ec=$?; local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$nid" "$ec" "$d" >> "$MASTER" 2>&1 <<'PY'
import sys,json; import pandas as pd
nid,ec,d=sys.argv[1],sys.argv[2],sys.argv[3]
try:
    v=float(pd.read_csv(d+"training_log.csv").val_qwk.max()); m=json.load(open(d+"test_metrics.json"))
    t=float(m["test_qwk"]); acc=float(m["test_accuracy"]); mr=float(m["test_macro_recall"])
    vp,tp=v>0.7902,t>0.9584
    print(f"=== {nid} DONE exit={ec} val={v:.4f} test={t:.4f} acc={acc:.2f} mr={mr:.2f} | val>0.7902:{'Y' if vp else 'N'} test>0.9584:{'Y' if tp else 'N'} => GATE1 {'PASS' if vp and tp else 'FAIL'} ===")
except Exception as e:
    print(f"=== {nid} DONE exit={ec} PARSE-ERROR {e} dir={d} ===")
PY
}
for spec in "${JOBS[@]}"; do run_one $spec & done
wait
echo "=== SCREEN COMPLETE ===" >> "$MASTER"
