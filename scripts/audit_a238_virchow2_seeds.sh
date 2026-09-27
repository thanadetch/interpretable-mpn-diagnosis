#!/usr/bin/env bash
# Bounded multi-seed robustness audit: is a238's UNIQUE virchow2 both-gate pass (seed-2:
# val 0.8205 > 0.8182 AND test 0.9512 > 0.9476) real, or a lucky-seed artifact?
# Runs a238 + the 'simple' baseline on virchow2 across seeds {0,1,3,42} (seed-2 already known),
# logging val_qwk/test_qwk/acc/macro_recall per run. Read-only w.r.t. the locked gate/split.
set -u
cd "$(dirname "$0")/.."
LOGDIR="logs/audit_a238_virchow2"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
echo "=== AUDIT START a238-vs-baseline virchow2 seeds {0,1,3,42} ===" >> "$MASTER"

wait_gpu(){ while pgrep -f "train_grading_reti.py" >/dev/null 2>&1; do sleep 8; done; }

# args: prefix model_flags...  -> echoes "val test acc mr"
run(){
  local pfx="$1"; shift
  python src/train_grading_reti.py --backbone virchow2 --data_root data \
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
  read v t acc mr <<<"$(run "audit_a238_virchow2_s${s}" --model_type novelty_attempt --novelty_id a238_sizecond_temperature --seed "$s")"
  echo "[a238 virchow2 seed=$s] val=$v test=$t acc=$acc mr=$mr" >> "$MASTER"
  wait_gpu
  read v t acc mr <<<"$(run "audit_base_virchow2_s${s}" --model_type simple --seed "$s")"
  echo "[base virchow2 seed=$s] val=$v test=$t acc=$acc mr=$mr" >> "$MASTER"
done

echo "=== AUDIT COMPLETE ===" >> "$MASTER"
