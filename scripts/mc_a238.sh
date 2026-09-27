#!/usr/bin/env bash
# Run a238 (size-conditioned attention temperature) under MULTI-CLASS on all 3 backbones, seed 2.
# Formulation ablation (mirror of mc_a215.sh). Self-waits for a free GPU (40s) to avoid racing on MPS.
set -u
ROOT=/Users/thanadetch/projects/interpretable-mpn-diagnosis
cd "$ROOT"
LOGDIR=logs/mc_a238; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
NID=a238_sizecond_temperature
wait_gpu(){ local n=0; while [ "$n" -lt 5 ]; do if pgrep -f "train_grading_reti.py" >/dev/null 2>&1; then n=0; else n=$((n+1)); fi; sleep 8; done; }

echo "=== MC a238 START $(date) ===" >> "$MASTER"
for bb in titan virchow2 uni2; do
  wait_gpu
  pfx=mc_a238_${bb}_s2
  echo ">>> training $pfx $(date)" >> "$MASTER"
  python src/train_grading_reti.py --backbone "$bb" --data_root data \
    --model_type novelty_attempt --novelty_id "$NID" \
    --formulation classification --main_metric macro_recall --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$bb" "$d" "$MASTER" <<'PY'
import sys, json, pandas as pd
bb, d, master = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    vmr = float(pd.read_csv(d + "training_log.csv").val_macro_recall.max())
    m = json.load(open(d + "test_metrics.json"))
    rc = m.get("test_recall_per_class", {})
    g = "/".join(f"{rc.get(k, '?')}" for k in ("G0", "G1", "G2", "G3"))
    line = (f"[mc-a238 {bb}] val_mr={vmr:.2f} | test_qwk={m['test_qwk']:.4f} "
            f"acc={m['test_accuracy']:.2f} mr={m['test_macro_recall']:.2f} | recall G0/G1/G2/G3={g}")
except Exception as e:
    line = f"[mc-a238 {bb}] ERR {e}"
print(line); open(master, "a").write(line + "\n")
PY
done
echo "=== MC a238 COMPLETE $(date) ===" >> "$MASTER"
