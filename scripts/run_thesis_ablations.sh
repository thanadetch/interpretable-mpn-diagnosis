#!/usr/bin/env bash
# Thesis ablation grid: 12 standard + 12 resize(no_patch) = 24 runs, PARALLEL.
# {uni2,virchow2,titan} x {mean_pool,simple} x {regression,classification}, seed=2.
# device=auto (MPS) kept identical to the validated baseline run; num_workers lowered
# to 2 (proven irrelevant to results: shuffle is seeded in the main process, __getitem__
# has no randomness) purely to reduce process oversubscription under parallelism.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOGDIR=logs/thesis_ablation
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
: > "$MASTER"
MAXJOBS="${MAXJOBS:-5}"
NW="${NW:-2}"

run_one () {
  local idx="$1" pfx="$2" bb="$3" mt="$4" form="$5" metric="$6" postfix="$7"
  local tag="${idx}_${pfx}_${bb}_${mt}_${postfix}"
  echo "=== RUN ${idx}/24 START: bb=${bb} model=${mt} form=${form} metric=${metric} ===" >> "$MASTER"
  python src/train_grading_reti.py \
    --backbone "$bb" --model_type "$mt" \
    --epochs 50 --lr 1e-4 --formulation "$form" \
    --main_metric "$metric" --seed 2 --num_workers "$NW" \
    --prefix "$pfx" --postfix "$postfix" \
    > "$LOGDIR/${tag}.log" 2>&1
  local ec=$?
  local rundir
  rundir=$(ls -dt experiments/*/"${pfx}_reti_${mt}_${bb}_${postfix}"_*/ 2>/dev/null | head -1)
  local tqwk="?" tacc="?"
  if [ -n "$rundir" ] && [ -f "${rundir}test_metrics.json" ]; then
    tqwk=$(python -c "import json;print(round(json.load(open('${rundir}test_metrics.json')).get('test_qwk',float('nan')),4))" 2>/dev/null)
    tacc=$(python -c "import json;print(round(json.load(open('${rundir}test_metrics.json')).get('test_accuracy',float('nan')),4))" 2>/dev/null)
  fi
  echo "=== RUN ${idx}/24 DONE exit=${ec} test_qwk=${tqwk} test_acc=${tacc} dir=${rundir} ===" >> "$MASTER"
}

# idx prefix backbone model formulation metric postfix
JOBS=(
  "01 01 uni2 mean_pool regression qwk regression"
  "02 02 uni2 simple regression qwk regression"
  "03 03 virchow2 mean_pool regression qwk regression"
  "04 04 virchow2 simple regression qwk regression"
  "05 05 titan mean_pool regression qwk regression"
  "06 06 titan simple regression qwk regression"
  "07 07 uni2 mean_pool classification macro_recall multiclass"
  "08 08 uni2 simple classification macro_recall multiclass"
  "09 09 virchow2 mean_pool classification macro_recall multiclass"
  "10 10 virchow2 simple classification macro_recall multiclass"
  "11 11 titan mean_pool classification macro_recall multiclass"
  "12 12 titan simple classification macro_recall multiclass"
  "13 01 uni2_no_patch mean_pool regression qwk regression"
  "14 02 uni2_no_patch simple regression qwk regression"
  "15 03 virchow2_no_patch mean_pool regression qwk regression"
  "16 04 virchow2_no_patch simple regression qwk regression"
  "17 05 titan_no_patch mean_pool regression qwk regression"
  "18 06 titan_no_patch simple regression qwk regression"
  "19 07 uni2_no_patch mean_pool classification macro_recall multiclass"
  "20 08 uni2_no_patch simple classification macro_recall multiclass"
  "21 09 virchow2_no_patch mean_pool classification macro_recall multiclass"
  "22 10 virchow2_no_patch simple classification macro_recall multiclass"
  "23 11 titan_no_patch mean_pool classification macro_recall multiclass"
  "24 12 titan_no_patch simple classification macro_recall multiclass"
)

echo "=== LAUNCHING ${#JOBS[@]} runs, ${MAXJOBS} lanes, num_workers=${NW} ===" >> "$MASTER"
for spec in "${JOBS[@]}"; do
  # throttle: wait until a lane frees (bash 3.2 compatible — no `wait -n`)
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 4; done
  run_one $spec &
done
wait
echo "=== ALL 24 RUNS COMPLETE ===" >> "$MASTER"
