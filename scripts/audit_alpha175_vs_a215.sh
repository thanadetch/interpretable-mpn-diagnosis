#!/usr/bin/env bash
# Multi-seed paired comparison: a215 (alpha=1.5) vs a308 (alpha=1.75), to settle whether the
# seed-2 edge of alpha=1.75 over a215 survives averaging or is single-seed noise.
# Seeds {0,1,3,42} x {titan,virchow2,uni2} x {a215, a308}. seed-2 already known. Analysis only.
set -u
cd /Users/thanadetch/projects/interpretable-mpn-diagnosis
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
OUT=logs/audit_alpha175; mkdir -p "$OUT"
SUM="$OUT/summary.csv"; echo "model,backbone,seed,val_qwk,test_qwk,best_epoch" > "$SUM"
for NID in a215_adaptive_sparse_gated_attention_pooling a308_alpha175; do
  TAG=$([ "$NID" = "a308_alpha175" ] && echo a175 || echo a215)
  for BB in titan virchow2 uni2; do
    for S in 0 1 3 42; do
      PFX="cmp_${TAG}_${BB}_s${S}"
      $PY src/train_grading_reti.py --backbone "$BB" --model_type novelty_attempt --novelty_id "$NID" \
        --data_root data --epochs 50 --lr 1e-4 --batch_size 1 --seed "$S" --num_workers 2 --topk 0 \
        --hidden_dim 128 --early_stop_patience 15 --formulation regression --main_metric qwk \
        --prefix "$PFX" >> "$OUT/master.log" 2>&1
      D=$(ls -dt experiments/*/${PFX}_* 2>/dev/null | head -1)
      if [ -n "$D" ] && [ -f "$D/test_metrics.json" ]; then
        $PY - "$D" "$TAG" "$BB" "$S" "$SUM" <<'PYEOF'
import json,sys
D,TAG,BB,S,SUM=sys.argv[1:6]
v=json.load(open(f"{D}/val_metrics.json")); t=json.load(open(f"{D}/test_metrics.json"))
open(SUM,"a").write(f"{TAG},{BB},{S},{v['val_qwk']:.4f},{t['test_qwk']:.4f},{v.get('best_epoch')}\n")
PYEOF
      else echo "${TAG},${BB},${S},NA,NA,NA" >> "$SUM"; fi
    done
  done
done
echo "=== DONE ===" >> "$OUT/master.log"; cat "$SUM" >> "$OUT/master.log"
