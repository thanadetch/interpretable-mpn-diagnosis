#!/usr/bin/env bash
# Companion multi-seed robustness audit for the THESIS ANCHOR (a215) on the GATE backbone (titan).
# Runs a215 + the 'simple' baseline on titan across seeds {0,1,3,42} (seed-2 already known:
# a215 val 0.7968/test 0.9600; baseline test 0.9584), logging val_qwk/test_qwk/acc/macro_recall.
# Produces the mean±std reviewer-proofing dataset for the thesis interpretability anchor.
set -u
cd "$(dirname "$0")/.."
LOGDIR="logs/audit_a215_titan"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
echo "=== AUDIT START a215-vs-baseline titan seeds {0,1,3,42} ===" >> "$MASTER"

wait_gpu(){ while pgrep -f "train_grading_reti.py" >/dev/null 2>&1; do sleep 8; done; }

# args: prefix model_flags...  -> echoes "val test acc mr"
run(){
  local pfx="$1"; shift
  python src/train_grading_reti.py --backbone titan --data_root data \
    "$@" \
    --formulation regression --main_metric qwk --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$d" <<'PY'
import sys,json,pandas as pd
d=sys.argv[1]
try:
    v=float(pd.read_csv(d+"training_log.csv").val_qwk.max()); m=json.load(open(d+"test_metrics.json"))
    print(f"{v:.4f} {m['test_qwk']:.4f} {m['test_accuracy']:.2f} {m['test_macro_recall']:.2f}")
except Exception as e:
    print("ERR ERR ERR ERR")
PY
}

for s in 0 1 3 42; do
  wait_gpu
  read v t acc mr <<<"$(run "audit_a215_titan_s${s}" --model_type novelty_attempt --novelty_id a215_adaptive_sparse_gated_attention_pooling --seed "$s")"
  echo "[a215 titan seed=$s] val=$v test=$t acc=$acc mr=$mr" >> "$MASTER"
  wait_gpu
  read v t acc mr <<<"$(run "audit_base_titan_s${s}" --model_type simple --seed "$s")"
  echo "[base titan seed=$s] val=$v test=$t acc=$acc mr=$mr" >> "$MASTER"
done

echo "=== AUDIT COMPLETE ===" >> "$MASTER"
