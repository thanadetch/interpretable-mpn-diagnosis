#!/usr/bin/env bash
# DIAGNOSTIC T-sweep on uni2: trace uni2 test_qwk vs fixed attention temperature T (a163).
# Answers "what T (if any) would raise uni2 test above baseline 0.9418?" — a CHARACTERISATION of the
# test landscape, NOT a selectable result (real pipeline selects on val only). bash-3.2 safe.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/tsweep_uni2; mkdir -p "$LOGDIR"; MASTER="$LOGDIR/master.log"; : > "$MASTER"
echo "=== T-SWEEP uni2 (baseline simple uni2 test=0.9418; a158 learned T=0.957 test=0.9378) ===" >> "$MASTER"
# T value | tag
SPECS="0.7|T0p7 0.85|T0p85 1.0|T1p0 1.25|T1p25 1.5|T1p5 2.0|T2p0 3.0|T3p0"
wait_gpu(){ while pgrep -f "train_grading_reti.py" >/dev/null 2>&1; do sleep 8; done; }
for spec in $SPECS; do
  T="${spec%%|*}"; tag="${spec#*|}"; pfx="uni2_a163_${tag}"
  wait_gpu
  echo "=== uni2 T=$T START ===" >> "$MASTER"
  A163_T="$T" python src/train_grading_reti.py --backbone uni2 --data_root data \
    --model_type novelty_attempt --novelty_id a163_fixed_temperature \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$T" "$d" >> "$MASTER" 2>&1 <<'PY'
import sys,json,pandas as pd
T,d=sys.argv[1],sys.argv[2]
try:
    v=float(pd.read_csv(d+"training_log.csv").val_qwk.max()); m=json.load(open(d+"test_metrics.json"))
    t=float(m["test_qwk"]); acc=float(m["test_accuracy"]); mr=float(m["test_macro_recall"])
    print(f"=== uni2 T={T} DONE val={v:.4f} test={t:.4f} (base 0.9418, {t-0.9418:+.4f}) acc={acc:.2f} mr={mr:.2f} ===")
except Exception as e:
    print(f"=== uni2 T={T} ERR {e} ===")
PY
done
echo "=== T-SWEEP uni2 COMPLETE ===" >> "$MASTER"
