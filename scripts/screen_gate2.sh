#!/usr/bin/env bash
# GATE 2 (only after a candidate passes GATE 1 = titan): run virchow2 AND uni2 IN PARALLEL.
# Bars (local-Mac): virchow2 val>0.8182 AND test>0.9476 ; uni2 test>0.9418.
# Usage: NID=a158_diffuse_temperature_gated PFX=r48_a158 bash scripts/screen_gate2.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
NID="${NID:?set NID=<novelty_id>}"
PFX="${PFX:?set PFX=<prefix base>}"
LOGDIR=logs/gate2_${PFX}
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
: > "$MASTER"
NW="${NW:-2}"

run_one () {
  local bb="$1" pfx="$2"
  echo "=== $bb START ===" >> "$MASTER"
  python src/train_grading_reti.py --backbone "$bb" --data_root data \
    --model_type novelty_attempt --novelty_id "$NID" \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers "$NW" \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  local ec=$?
  local d
  d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$bb" "$ec" "$d" >> "$MASTER" 2>&1 <<'PY'
import sys, json
import pandas as pd
bb, ec, d = sys.argv[1], sys.argv[2], sys.argv[3]
BAR = {"virchow2": (0.8182, 0.9476), "uni2": (None, 0.9418)}
try:
    v = float(pd.read_csv(d+"training_log.csv").val_qwk.max())
    m = json.load(open(d+"test_metrics.json")); t = float(m["test_qwk"]); acc = float(m["test_accuracy"])
    vbar, tbar = BAR[bb]
    vp = (vbar is None) or (v > vbar); tp = t > tbar
    gate = "PASS" if (vp and tp) else "FAIL"
    vtxt = f"val={v:.4f}(>{vbar}:{'Y' if (vbar is None or v>vbar) else 'N'})" if True else ""
    print(f"=== {bb} DONE exit={ec} val={v:.4f} test={t:.4f} acc={acc:.2f} | "
          f"test>{tbar}:{'Y' if tp else 'N'}"
          f"{'' if vbar is None else f' val>{vbar}:'+('Y' if v>vbar else 'N')} => {bb} {gate} ===")
except Exception as e:
    print(f"=== {bb} DONE exit={ec} PARSE-ERROR {e} dir={d} ===")
PY
}

echo "=== GATE2 $NID : virchow2 + uni2 (parallel) ===" >> "$MASTER"
run_one virchow2 "${PFX}_virchow2_s2" &
run_one uni2     "${PFX}_uni2_s2" &
wait
echo "=== GATE2 COMPLETE ===" >> "$MASTER"
