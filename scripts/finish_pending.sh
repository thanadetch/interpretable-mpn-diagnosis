#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/finish_pending; mkdir -p "$LOGDIR"; MASTER="$LOGDIR/master.log"; : > "$MASTER"
MAXJOBS=4
# spec: nid|backbone|prefix|base_test|tag
JOBS=(
"a167_patch_dropout_gated|virchow2|r55b_a167_virchow2_s2|0.9476|a167-GATE2"
"a167_patch_dropout_gated|uni2|r55b_a167_uni2_s2|0.9418|a167-GATE2"
"a169_bone_suppressed_gated|titan|r56b_a169_titan_s2|0.9584|a169-bonesuppress"
"a169_bone_suppressed_gated|virchow2|r56b_a169_virchow2_s2|0.9476|a169-bonesuppress"
"a169_bone_suppressed_gated|uni2|r56b_a169_uni2_s2|0.9418|a169-bonesuppress"
"a166_nuisance_pc_removed|titan|r53b_a166_titan_s2|0.9584|a166-GATE1"
)
run_one(){
  local spec="$1"
  local nid="${spec%%|*}"; local rest="${spec#*|}"
  local bb="${rest%%|*}"; rest="${rest#*|}"
  local pfx="${rest%%|*}"; rest="${rest#*|}"
  local base="${rest%%|*}"; local tag="${rest#*|}"
  echo "=== [$tag] $bb START ===" >> "$MASTER"
  python src/train_grading_reti.py --backbone "$bb" --data_root data \
    --model_type novelty_attempt --novelty_id "$nid" \
    --formulation regression --main_metric qwk --seed 2 --lr 1e-4 --epochs 50 \
    --batch_size 1 --early_stop_patience 15 --topk 0 --num_workers 2 \
    --prefix "$pfx" > "$LOGDIR/${pfx}.log" 2>&1
  local ec=$?; local d; d=$(ls -dt experiments/*/"${pfx}"_*/ 2>/dev/null | head -1)
  python - "$tag" "$bb" "$ec" "$d" "$base" >> "$MASTER" 2>&1 <<'PY'
import sys,json; import pandas as pd
tag,bb,ec,d,base=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5])
try:
    v=float(pd.read_csv(d+"training_log.csv").val_qwk.max()); m=json.load(open(d+"test_metrics.json"))
    t=float(m["test_qwk"]); acc=float(m["test_accuracy"]); mr=float(m["test_macro_recall"])
    print(f"=== [{tag}] {bb} DONE exit={ec} val={v:.4f} test={t:.4f} (base {base:.4f}, {t-base:+.4f}) acc={acc:.2f} mr={mr:.2f} ===")
except Exception as e:
    print(f"=== [{tag}] {bb} DONE exit={ec} ERR {e} ===")
PY
}
for spec in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 4; done
  run_one "$spec" &
done
wait
echo "=== FINISH_PENDING COMPLETE ===" >> "$MASTER"
