#!/usr/bin/env bash
# Stress-test the THESIS HEADLINE (regression >> multi-class) for seed-robustness.
# The multi-seed audits showed seed-2 is a +2sigma lucky test fold, so the headline must be
# checked across seeds. We already have multi-seed REGRESSION baseline test_qwk (from the
# a238/a215 audits); this fills in the MULTI-CLASS baseline test_qwk across seeds {0,1,3,42}
# on titan + virchow2 (seed-2 known: titan 0.9080, virchow2 0.9275). Then regression-vs-multiclass
# can be compared as mean+/-std per backbone. 'simple' baseline only; gate/split untouched.
set -u
cd "$(dirname "$0")/.."
LOGDIR="logs/audit_formulation"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
echo "=== AUDIT START multiclass-baseline titan+virchow2 seeds {0,1,3,42} ===" >> "$MASTER"

wait_gpu(){ while pgrep -f "train_grading_reti.py" >/dev/null 2>&1; do sleep 8; done; }

# args: prefix backbone seed -> echoes "test_qwk test_acc test_macro_recall"
run(){
  local pfx="$1" bb="$2" seed="$3"
  python src/train_grading_reti.py --backbone "$bb" --data_root data \
    --model_type simple --formulation classification --main_metric macro_recall \
    --seed "$seed" --lr 1e-4 --epochs 50 --batch_size 1 --early_stop_patience 15 \
    --topk 0 --num_workers 2 --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$d" <<'PY'
import sys,json
d=sys.argv[1]
try:
    m=json.load(open(d+"test_metrics.json"))
    print(f"{m['test_qwk']:.4f} {m['test_accuracy']:.2f} {m['test_macro_recall']:.2f}")
except Exception as e:
    print("ERR ERR ERR")
PY
}

for bb in titan virchow2; do
  for s in 0 1 3 42; do
    wait_gpu
    read tq acc mr <<<"$(run "audit_mc_${bb}_s${s}" "$bb" "$s")"
    echo "[mc-base $bb seed=$s] test_qwk=$tq acc=$acc macro_recall=$mr" >> "$MASTER"
  done
done

echo "=== AUDIT COMPLETE ===" >> "$MASTER"
