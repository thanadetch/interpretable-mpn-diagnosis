#!/usr/bin/env bash
# Autonomous gate cascade: GATE1(titan) -> GATE2(virchow2,uni2) -> GATE3(virchow2 multi-seed).
# Each candidate auto-promotes only if it clears the current gate. bash-3.2 safe (no declare -A / wait -n).
# Usage: bash scripts/gate_cascade.sh a172_embedding_noise_regularized_gated a173_... a174_...
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/gate_cascade; mkdir -p "$LOGDIR"; MASTER="$LOGDIR/master.log"
echo "=== CASCADE START (candidates: $*) ===" >> "$MASTER"

# GATE bars (seed-2 local-Mac baselines)
G1_VAL=0.7902; G1_TEST=0.9584          # titan
G2V_VAL=0.8182; G2V_TEST=0.9476        # virchow2
G2U_TEST=0.9418                        # uni2
G3_SEEDS="0 1 3 42"                    # virchow2 robustness
G3_MEAN=0.9476                         # candidate mean test must exceed single-seed baseline

# wait until GPU is free (no other trainer running) to avoid MPS thrash
wait_gpu(){ while pgrep -f "train_grading_reti.py" >/dev/null 2>&1; do sleep 8; done; }

# run one training; echo "val test acc mr" to stdout
train(){ # nid backbone seed prefix
  local nid="$1" bb="$2" seed="$3" pfx="$4"
  python src/train_grading_reti.py --backbone "$bb" --data_root data \
    --model_type novelty_attempt --novelty_id "$nid" \
    --formulation regression --main_metric qwk --seed "$seed" --lr 1e-4 --epochs 50 \
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

ge(){ python - "$1" "$2" <<'PY'
import sys; print("1" if float(sys.argv[1])>float(sys.argv[2]) else "0")
PY
}

for nid in "$@"; do
  short=$(echo "$nid" | grep -oE "^a[0-9]+")
  echo "" >> "$MASTER"; echo ">>> CANDIDATE $short ($nid)" >> "$MASTER"
  # ---- GATE1: titan ----
  wait_gpu
  read v t acc mr <<<"$(train "$nid" titan 2 "casc_${short}_titan_s2")"
  if [ "$v" = "ERR" ]; then echo "[$short] GATE1 titan ERROR (see log)" >> "$MASTER"; continue; fi
  p1v=$(ge "$v" "$G1_VAL"); p1t=$(ge "$t" "$G1_TEST")
  if [ "$p1v" = "1" ] && [ "$p1t" = "1" ]; then
    echo "[$short] GATE1 titan PASS  val=$v test=$t acc=$acc mr=$mr  -> promote GATE2" >> "$MASTER"
  else
    echo "[$short] GATE1 titan FAIL  val=$v (>$G1_VAL?$p1v) test=$t (>$G1_TEST?$p1t) acc=$acc mr=$mr" >> "$MASTER"
    continue
  fi
  # ---- GATE2: virchow2 + uni2 ----
  wait_gpu
  read vv vt vacc vmr <<<"$(train "$nid" virchow2 2 "casc_${short}_virchow2_s2")"
  wait_gpu
  read uv ut uacc umr <<<"$(train "$nid" uni2 2 "casc_${short}_uni2_s2")"
  p2vv=$(ge "$vv" "$G2V_VAL"); p2vt=$(ge "$vt" "$G2V_TEST"); p2ut=$(ge "$ut" "$G2U_TEST")
  echo "[$short] GATE2 virchow2 val=$vv test=$vt | uni2 val=$uv test=$ut" >> "$MASTER"
  if [ "$p2vv" = "1" ] && [ "$p2vt" = "1" ] && [ "$p2ut" = "1" ]; then
    echo "[$short] GATE2 PASS (v:val>$G2V_VAL&test>$G2V_TEST, u:test>$G2U_TEST) -> promote GATE3" >> "$MASTER"
  else
    echo "[$short] GATE2 FAIL (vval?$p2vv vtest?$p2vt utest?$p2ut)" >> "$MASTER"
    continue
  fi
  # ---- GATE3: virchow2 multi-seed robustness ----
  sumf="$LOGDIR/${short}_gate3_seeds.txt"; : > "$sumf"
  for s in $G3_SEEDS; do
    wait_gpu
    read sv st sacc smr <<<"$(train "$nid" virchow2 "$s" "casc_${short}_virchow2_s${s}")"
    echo "$s $sv $st" >> "$sumf"
    echo "[$short] GATE3 seed=$s val=$sv test=$st" >> "$MASTER"
  done
  python - "$short" "$sumf" "$G3_MEAN" "$MASTER" <<'PY'
import sys,statistics as st
short,f,bar,master=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4]
rows=[l.split() for l in open(f) if l.strip()]
tests=[float(r[2]) for r in rows if r[2]!="ERR"]
with open(master,"a") as o:
    if not tests: o.write(f"[{short}] GATE3 ERROR no seeds\n"); sys.exit()
    mean=sum(tests)/len(tests); sd=st.pstdev(tests); mn=min(tests)
    ok = mean>bar and sd<0.03
    o.write(f"[{short}] GATE3 {'PASS' if ok else 'FAIL'} mean={mean:.4f} std={sd:.4f} min={mn:.4f} (bar mean>{bar} & std<0.03) seeds={tests}\n")
    if ok: o.write(f"*** [{short}] CLEARED ALL 3 GATES — ROBUST WINNER CANDIDATE ***\n")
PY
done
echo "" >> "$MASTER"; echo "=== CASCADE COMPLETE ===" >> "$MASTER"
