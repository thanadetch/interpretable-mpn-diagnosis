#!/usr/bin/env bash
# Alpha init-sweep: train a239 (init alpha=1.2) and a240 (init alpha=1.8) on titan, read CONVERGED alpha.
# Tests whether a215's alpha~1.5 is a genuine attractor (both converge toward 1.5) or just its init.
set -u
ROOT=/Users/thanadetch/projects/interpretable-mpn-diagnosis
cd "$ROOT"
LOGDIR=logs/alpha_sweep; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
wait_gpu(){ local n=0; while [ "$n" -lt 5 ]; do if pgrep -f "train_grading_reti.py" >/dev/null 2>&1; then n=0; else n=$((n+1)); fi; sleep 8; done; }

echo "=== ALPHA SWEEP START $(date) (a215 init=1.500 converged to 1.506 on titan) ===" >> "$MASTER"
for nid in a239_entmax_init_low a240_entmax_init_high; do
  wait_gpu
  pfx=asw_${nid}_titan_s2
  python src/train_grading_reti.py --backbone titan --data_root data \
    --model_type novelty_attempt --novelty_id "$nid" \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$nid" "$d" "$MASTER" <<'PY'
import sys, json, math, glob, torch, pandas as pd
nid, d, master = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    ck = sorted(glob.glob(d + "best_*.pth"))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    raw = float(sd["alpha_raw"]); alpha = 1 + 1 / (1 + math.exp(-raw))
    init = 1.2 if "low" in nid else 1.8
    v = float(pd.read_csv(d + "training_log.csv").val_qwk.max())
    m = json.load(open(d + "test_metrics.json"))
    line = f"[{nid}] init_alpha={init} -> CONVERGED alpha={alpha:.4f} (raw={raw:+.4f}) | val_qwk={v:.4f} test_qwk={m['test_qwk']:.4f}"
except Exception as e:
    line = f"[{nid}] ERR {e}"
print(line); open(master, "a").write(line + "\n")
PY
done
echo "=== ALPHA SWEEP COMPLETE $(date) ===" >> "$MASTER"
