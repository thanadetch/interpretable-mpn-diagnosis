#!/usr/bin/env bash
# Multi-seed audit of a298 (2-head independent ensemble) — the genuine 2/3 high-water candidate.
# Question: is a298's seed-2 result (uni2+virchow2 PASS, titan FAIL) robust across seeds, or a
# lucky-seed-2 fold like a274/a215/a238? Runs seeds {0,1,3,42} x {uni2,titan,virchow2} (seed-2 known).
# Analysis only; locked split/test; selection on val_qwk. Writes a summary table at the end.
set -u
cd /Users/thanadetch/projects/interpretable-mpn-diagnosis
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
NID=a298_two_head_ensemble_avg
OUTDIR=logs/audit_a298
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"
echo "backbone,seed,val_qwk,test_qwk,best_epoch" > "$SUMMARY"

for BB in uni2 titan virchow2; do
  for S in 0 1 3 42; do
    PFX="a298_audit_${BB}_s${S}"
    echo "=== training $BB seed=$S ===" | tee -a "$OUTDIR/master.log"
    $PY src/train_grading_reti.py --backbone "$BB" --model_type novelty_attempt --novelty_id "$NID" \
      --data_root data --epochs 50 --lr 1e-4 --batch_size 1 --seed "$S" --num_workers 2 --topk 0 \
      --hidden_dim 128 --early_stop_patience 15 --formulation regression --main_metric qwk \
      --prefix "$PFX" >> "$OUTDIR/master.log" 2>&1
    # locate the freshest run dir for this prefix and extract metrics
    D=$(ls -dt experiments/*/${PFX}_* 2>/dev/null | head -1)
    if [ -n "$D" ] && [ -f "$D/test_metrics.json" ]; then
      $PY - "$D" "$BB" "$S" "$SUMMARY" <<'PYEOF'
import json, sys
D, BB, S, SUMMARY = sys.argv[1:5]
v = json.load(open(f"{D}/val_metrics.json")); t = json.load(open(f"{D}/test_metrics.json"))
row = f"{BB},{S},{v['val_qwk']:.4f},{t['test_qwk']:.4f},{v.get('best_epoch')}"
open(SUMMARY, "a").write(row + "\n")
print("  -> " + row)
PYEOF
    else
      echo "${BB},${S},NA,NA,NA" >> "$SUMMARY"
      echo "  -> $BB s$S FAILED (no metrics)" | tee -a "$OUTDIR/master.log"
    fi
  done
done

echo "=== AUDIT COMPLETE ===" | tee -a "$OUTDIR/master.log"
cat "$SUMMARY" | tee -a "$OUTDIR/master.log"
