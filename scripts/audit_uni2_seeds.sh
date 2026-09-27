#!/usr/bin/env bash
# Complete the 3-backbone multi-seed matrix: uni2 was never audited multi-seed (only seed-2 from the gate).
# Runs a215(reg) + simple baseline(reg) + simple baseline(multi-class) on uni2, seeds {0,1,3,42}
# (seed-2 known: base reg test 0.9418, a215 reg 0.9262, base mc 0.9018). Logs test metrics per run;
# full metrics land in each run dir for extraction. Gate/split untouched.
set -u
cd "$(dirname "$0")/.."
LOGDIR="logs/audit_uni2"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
echo "=== AUDIT START uni2 a215/base(reg)+base(mc) seeds {0,1,3,42} ===" >> "$MASTER"

wait_gpu(){ while pgrep -f "train_grading_reti.py" >/dev/null 2>&1; do sleep 8; done; }

# args: prefix model_flags...  -> echoes "test_qwk acc macro_recall"
run(){
  local pfx="$1"; shift
  python src/train_grading_reti.py --backbone uni2 --data_root data \
    "$@" --lr 1e-4 --epochs 50 --batch_size 1 --early_stop_patience 15 \
    --topk 0 --num_workers 2 --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$d" <<'PY'
import sys,json,os
d=sys.argv[1]
try:
    m=json.load(open(os.path.join(d,"test_metrics.json")))
    print(f"{m['test_qwk']:.4f} {m['test_accuracy']:.2f} {m['test_macro_recall']:.2f}")
except Exception as e:
    print("ERR ERR ERR")
PY
}

for s in 0 1 3 42; do
  wait_gpu
  read tq acc mr <<<"$(run "audit_a215_uni2_s${s}" --model_type novelty_attempt --novelty_id a215_adaptive_sparse_gated_attention_pooling --formulation regression --main_metric qwk --seed "$s")"
  echo "[a215 uni2 reg seed=$s] test_qwk=$tq acc=$acc mr=$mr" >> "$MASTER"
  wait_gpu
  read tq acc mr <<<"$(run "audit_base_uni2_s${s}" --model_type simple --formulation regression --main_metric qwk --seed "$s")"
  echo "[base uni2 reg seed=$s] test_qwk=$tq acc=$acc mr=$mr" >> "$MASTER"
  wait_gpu
  read tq acc mr <<<"$(run "audit_mc_uni2_s${s}" --model_type simple --formulation classification --main_metric macro_recall --seed "$s")"
  echo "[mc-base uni2 seed=$s] test_qwk=$tq acc=$acc mr=$mr" >> "$MASTER"
done

echo "=== AUDIT COMPLETE ===" >> "$MASTER"
