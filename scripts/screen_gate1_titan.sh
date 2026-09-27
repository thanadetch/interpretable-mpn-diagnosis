#!/usr/bin/env bash
# Screen multiple novelty candidates at GATE 1 (titan, best/binding, hardest-first) IN PARALLEL.
# GATE 1 bar (local-Mac re-base): val_qwk > 0.7902 AND test_qwk > 0.9584.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/gate1_titan
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
: > "$MASTER"
MAXJOBS="${MAXJOBS:-4}"
NW="${NW:-2}"

# "novelty_id prefix"
JOBS=(
  "a158_diffuse_temperature_gated  r47_a158_titan_s2"
  "a159_axis_shrunk_gated          r47_a159_titan_s2"
  "a160_coverage_residual_gated    r47_a160_titan_s2"
)

run_one () {
  local nid="$1" pfx="$2"
  echo "=== $nid START ===" >> "$MASTER"
  python src/train_grading_reti.py --backbone titan --data_root data \
    --model_type novelty_attempt --novelty_id "$nid" \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers "$NW" \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  local ec=$?
  local d
  d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$nid" "$ec" "$d" >> "$MASTER" 2>&1 <<'PY'
import sys, json
import pandas as pd
nid, ec, d = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    v = float(pd.read_csv(d+"training_log.csv").val_qwk.max())
    m = json.load(open(d+"test_metrics.json"))
    t = float(m["test_qwk"]); acc = float(m["test_accuracy"])
    vp, tp = v > 0.7902, t > 0.9584
    gate = "PASS" if (vp and tp) else "FAIL"
    print(f"=== {nid} DONE exit={ec} val={v:.4f} test={t:.4f} acc={acc:.2f} | "
          f"val>{0.7902}:{'Y' if vp else 'N'} test>{0.9584}:{'Y' if tp else 'N'} => GATE1 {gate} ===")
except Exception as e:
    print(f"=== {nid} DONE exit={ec} PARSE-ERROR {e} dir={d} ===")
PY
}

echo "=== SCREEN GATE1(titan) ${#JOBS[@]} candidates, ${MAXJOBS} lanes, nw=${NW} | bar val>0.7902 AND test>0.9584 ===" >> "$MASTER"
for spec in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 4; done
  run_one $spec &
done
wait
echo "=== SCREEN COMPLETE ===" >> "$MASTER"
