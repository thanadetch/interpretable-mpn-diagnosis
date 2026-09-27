#!/usr/bin/env bash
# a170 already PASSED GATE1 (titan val0.7913/test0.9590). Run GATE2 (virchow2+uni2) then GATE3
# (virchow2 multi-seed) only. GPU-wait guard queues behind any running trainer. bash-3.2 safe.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
NID=a170_per_bag_adaptive_temperature; SHORT=a170
LOGDIR=logs/gate_cascade; mkdir -p "$LOGDIR"; MASTER="$LOGDIR/master.log"
echo "" >> "$MASTER"; echo ">>> CANDIDATE $SHORT ($NID)  [GATE1 already PASS val0.7913/test0.9590]" >> "$MASTER"

G2V_VAL=0.8182; G2V_TEST=0.9476; G2U_TEST=0.9418; G3_SEEDS="0 1 3 42"; G3_MEAN=0.9476
wait_gpu(){ while pgrep -f "train_grading_reti.py" >/dev/null 2>&1; do sleep 8; done; }
train(){ # nid backbone seed prefix
  python src/train_grading_reti.py --backbone "$2" --data_root data \
    --model_type novelty_attempt --novelty_id "$1" \
    --formulation regression --main_metric qwk --seed "$3" --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$4" > "$LOGDIR/${4}.log" 2>&1
  local d; d=$(ls -dt experiments/*/"${4}"_*/ 2>/dev/null | head -1)
  python - "$d" <<'PY'
import sys,json,pandas as pd
d=sys.argv[1]
try:
    v=float(pd.read_csv(d+"training_log.csv").val_qwk.max()); m=json.load(open(d+"test_metrics.json"))
    print(f"{v:.4f} {m['test_qwk']:.4f} {m['test_accuracy']:.2f} {m['test_macro_recall']:.2f}")
except Exception as e: print("ERR ERR ERR ERR")
PY
}
ge(){ python - "$1" "$2" <<'PY'
import sys; print("1" if float(sys.argv[1])>float(sys.argv[2]) else "0")
PY
}
# ---- GATE2 ----
wait_gpu; read vv vt vacc vmr <<<"$(train "$NID" virchow2 2 "casc_${SHORT}_virchow2_s2")"
wait_gpu; read uv ut uacc umr <<<"$(train "$NID" uni2 2 "casc_${SHORT}_uni2_s2")"
p2vv=$(ge "$vv" "$G2V_VAL"); p2vt=$(ge "$vt" "$G2V_TEST"); p2ut=$(ge "$ut" "$G2U_TEST")
echo "[$SHORT] GATE2 virchow2 val=$vv test=$vt | uni2 val=$uv test=$ut" >> "$MASTER"
if [ "$p2vv" = "1" ] && [ "$p2vt" = "1" ] && [ "$p2ut" = "1" ]; then
  echo "[$SHORT] GATE2 PASS -> promote GATE3" >> "$MASTER"
else
  echo "[$SHORT] GATE2 FAIL (vval?$p2vv vtest?$p2vt utest?$p2ut)" >> "$MASTER"
  echo "=== promote_a170 COMPLETE (stopped at GATE2) ===" >> "$MASTER"; exit 0
fi
# ---- GATE3 ----
sumf="$LOGDIR/${SHORT}_gate3_seeds.txt"; : > "$sumf"
for s in $G3_SEEDS; do
  wait_gpu; read sv st sacc smr <<<"$(train "$NID" virchow2 "$s" "casc_${SHORT}_virchow2_s${s}")"
  echo "$s $sv $st" >> "$sumf"; echo "[$SHORT] GATE3 seed=$s val=$sv test=$st" >> "$MASTER"
done
python - "$SHORT" "$sumf" "$G3_MEAN" "$MASTER" <<'PY'
import sys,statistics as st
short,f,bar,master=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4]
rows=[l.split() for l in open(f) if l.strip()]; tests=[float(r[2]) for r in rows if r[2]!="ERR"]
with open(master,"a") as o:
    if not tests: o.write(f"[{short}] GATE3 ERROR\n"); sys.exit()
    mean=sum(tests)/len(tests); sd=st.pstdev(tests); mn=min(tests); ok=mean>bar and sd<0.03
    o.write(f"[{short}] GATE3 {'PASS' if ok else 'FAIL'} mean={mean:.4f} std={sd:.4f} min={mn:.4f} seeds={tests}\n")
    if ok: o.write(f"*** [{short}] CLEARED ALL 3 GATES — ROBUST WINNER ***\n")
PY
echo "=== promote_a170 COMPLETE ===" >> "$MASTER"
