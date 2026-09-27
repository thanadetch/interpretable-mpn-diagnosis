#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# loop_runner.sh — single, parameterized batch runner for novelty attempts.
#
# Replaces the per-batch loop_runner_v{1..4}.sh files. Same hyperparameters
# as the locked baseline
# (experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342):
#   --backbone virchow2 --epochs 50 --lr 1e-4 --batch_size 1
#   --num_workers $NUM_WORKERS --topk 0 --early_stop_patience 15
#   --formulation regression --main_metric qwk
# Only --model_type / --novelty_id / --postfix change between attempts.
#
# Usage:
#   # 1) Pass attempts as positional args (most common):
#   ./scripts/loop_runner.sh a36_my_novelty a37_other_novelty
#
#   # 2) Read attempts from a batch file (one aNN_name per line, # comments OK):
#   ./scripts/loop_runner.sh --batch scripts/batches/v5.txt
#
#   # 3) Auto-discover: every aNN module under src/models/novelty_attempts/
#   #    that doesn't yet have a completed _s${SEED}_ experiment dir.
#   ./scripts/loop_runner.sh --auto
#
# Knobs (env vars):
#   DEVICE            auto | cuda | mps | cpu          (default: mps)
#   NUM_WORKERS                                          (default: 2)
#   EPOCHS                                               (default: 50)
#   SEED                                                 (default: 2)
#   TARGET_VAL_QWK    val_qwk threshold to "beat baseline"  (default: 0.8182)
#   TARGET_TEST_QWK   test_qwk threshold                    (default: 0.9476)
#   STOP_ON_WIN       1 = exit on first BEATS_BASELINE     (default: 1)
#   TAG               optional label written into the loop log
#
# Per-attempt artifact: runs/<attempt>.out (full stdout+stderr).
# Global log:           runs/loop_runner.log (appended).
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=src
DEVICE="${DEVICE:-mps}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-2}"
TARGET_VAL_QWK="${TARGET_VAL_QWK:-0.8182}"
TARGET_TEST_QWK="${TARGET_TEST_QWK:-0.9476}"
STOP_ON_WIN="${STOP_ON_WIN:-1}"
TAG="${TAG:-}"

mkdir -p runs

# ── Parse args ────────────────────────────────────────────────────────
ATTEMPTS=()
MODE="args"
if [ $# -ge 1 ] && [ "$1" = "--batch" ]; then
  [ $# -ge 2 ] || { echo "ERROR: --batch requires a file path"; exit 2; }
  BATCH_FILE="$2"; MODE="batch:$BATCH_FILE"
  [ -f "$BATCH_FILE" ] || { echo "ERROR: batch file not found: $BATCH_FILE"; exit 2; }
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [ -z "$line" ] && continue
    ATTEMPTS+=("$line")
  done < "$BATCH_FILE"
elif [ $# -ge 1 ] && [ "$1" = "--auto" ]; then
  MODE="auto"
  for f in src/models/novelty_attempts/a*.py; do
    name="$(basename "$f" .py)"
    [ "$name" = "__init__" ] && continue
    EXISTING=$(ls -td "experiments/reti_novelty_attempt_virchow2_${name}_s${SEED}_"* \
                       "experiments/"*"/reti_novelty_attempt_virchow2_${name}_s${SEED}_"* 2>/dev/null | head -1)
    if [ -z "$EXISTING" ] || [ ! -f "$EXISTING/test_metrics.json" ]; then
      ATTEMPTS+=("$name")
    fi
  done
elif [ $# -ge 1 ]; then
  ATTEMPTS=("$@")
else
  cat <<EOF
Usage:
  $0 a36_foo a37_bar ...
  $0 --batch scripts/batches/v5.txt
  $0 --auto
EOF
  exit 2
fi

if [ "${#ATTEMPTS[@]}" -eq 0 ]; then
  echo "No attempts to run (mode=$MODE)."; exit 0
fi

# ── Pre-flight: enforce philosophy-bucket diversity (playbook §0 rule 7) ──
# Added 2026-05-25. Set ENFORCE_BUCKET_RULE=0 to bypass for ad-hoc work.
ENFORCE_BUCKET_RULE="${ENFORCE_BUCKET_RULE:-1}"
if [ "$ENFORCE_BUCKET_RULE" = "1" ]; then
  python - "${ATTEMPTS[@]}" <<'PY' || { echo "Pre-flight aborted (set ENFORCE_BUCKET_RULE=0 to bypass)"; exit 3; }
import json, pathlib, re, sys

repo = pathlib.Path(__file__).resolve().parent if "__file__" in dir() else pathlib.Path.cwd()
# When invoked via the shell heredoc, __file__ may be missing — use cwd.
repo = pathlib.Path.cwd()
nov_dir = repo / "src" / "models" / "novelty_attempts"
notes = repo / "NOVELTY_NOTES.md"

# Load the cap from frontmatter (PyYAML present in the project venv).
cap = 5
try:
    import yaml
    text = notes.read_text()
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end > 0:
            fm = yaml.safe_load(text[4:end]) or {}
            cap = int((fm.get("search_config") or {}).get("max_consecutive_in_bucket", 5))
except Exception:
    pass

def bucket_of(name: str) -> str:
    p = nov_dir / f"{name}.py"
    if not p.exists():
        return "missing"
    head = p.read_text()[:4000]
    m = re.search(r"Philosophy bucket:\s*([a-z_]+)", head, flags=re.IGNORECASE)
    return m.group(1).strip().lower() if m else "untagged"

attempts = sys.argv[1:]
buckets = [bucket_of(a) for a in attempts]
print(f"[pre-flight] attempts -> buckets: {list(zip(attempts, buckets))}")

# Last `cap` aNN modules already on disk (by aNN id), excluding the new attempts.
def aid(name: str) -> int:
    m = re.match(r"a(\d+)_", name)
    return int(m.group(1)) if m else -1

disk_modules = sorted(
    (p.stem for p in nov_dir.glob("a*.py") if p.is_file() and p.name != "__init__.py"),
    key=aid,
)
# Drop any attempts already on disk (avoid double-counting).
disk_modules = [m for m in disk_modules if m not in attempts]
window = (disk_modules + attempts)[-cap:]
window_buckets = [bucket_of(m) for m in window]
print(f"[pre-flight] consecutive window (last {cap}) -> {list(zip(window, window_buckets))}")

unique = {b for b in window_buckets if b not in ("untagged", "missing", "")}
# Block only when EVERY slot in the window has a known bucket tag AND
# they all agree. Untagged/missing slots are treated as "unknown" and
# therefore cannot establish a violation on their own (we don't know
# which bucket they would have belonged to).
all_known = all(b not in ("untagged", "missing", "") for b in window_buckets)
if len(window_buckets) >= cap and all_known and len(unique) == 1:
    bucket = next(iter(unique))
    print(
        f"[pre-flight] BLOCKED — §0 rule 7 would be violated: last {cap} "
        f"consecutive modules all in bucket '{bucket}'. "
        "Pick at least one module from an unexplored bucket per playbook §12.4.",
        file=sys.stderr,
    )
    sys.exit(1)

# Soft warning for untagged attempts (template requires the tag).
for name, b in zip(attempts, buckets):
    if b in ("untagged", "missing"):
        print(
            f"[pre-flight] WARNING: '{name}' is {b} — add a "
            "'Philosophy bucket: <name>' line near the top of its docstring "
            "(playbook §3 template).",
            file=sys.stderr,
        )
print("[pre-flight] OK — bucket diversity rule satisfied.")
PY
fi
# ── End pre-flight ──

log_global () { echo "$@" | tee -a runs/loop_runner.log; }

baseline_check () {
  local d="$1"
  if [ ! -f "$d/val_metrics.json" ] || [ ! -f "$d/test_metrics.json" ]; then
    echo "MISSING METRICS"; return 2
  fi
  python - "$d" "$TARGET_VAL_QWK" "$TARGET_TEST_QWK" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
target_v = float(sys.argv[2]); target_t = float(sys.argv[3])
v = json.loads((d/"val_metrics.json").read_text())
t = json.loads((d/"test_metrics.json").read_text())
val_qwk  = v.get("val_qwk",  float("nan"))
test_qwk = t.get("test_qwk", float("nan"))
val_acc  = v.get("val_accuracy",  float("nan"))
test_acc = t.get("test_accuracy", float("nan"))
status = "BEATS_BASELINE" if (val_qwk > target_v and test_qwk > target_t) else "NO_BEAT"
print(f"VAL_QWK={val_qwk:.4f} (>{target_v}? {val_qwk>target_v})  "
      f"TEST_QWK={test_qwk:.4f} (>{target_t}? {test_qwk>target_t})  "
      f"VAL_ACC={val_acc:.2f}  TEST_ACC={test_acc:.2f}  ::  {status}")
PY
}

log_global ""
log_global "============================================================"
log_global "loop_runner${TAG:+ [$TAG]} started at $(date)"
log_global "Mode    : $MODE"
log_global "Attempts: ${ATTEMPTS[*]}"
log_global "Targets : val_qwk > $TARGET_VAL_QWK  AND  test_qwk > $TARGET_TEST_QWK"
log_global "Config  : device=$DEVICE workers=$NUM_WORKERS epochs=$EPOCHS seed=$SEED"
log_global "============================================================"

for ATT in "${ATTEMPTS[@]}"; do
  POSTFIX="${ATT}_s${SEED}"
  EXISTING=$(ls -td "experiments/reti_novelty_attempt_virchow2_${POSTFIX}_"* \
                     "experiments/"*"/reti_novelty_attempt_virchow2_${POSTFIX}_"* 2>/dev/null | head -1)
  if [ -n "$EXISTING" ] && [ -f "$EXISTING/test_metrics.json" ]; then
    log_global ""
    log_global "> Skipping $ATT (already done): $EXISTING"
    RESULT=$(baseline_check "$EXISTING")
    log_global "  $RESULT"
    if [ "$STOP_ON_WIN" = "1" ] && echo "$RESULT" | grep -q "BEATS_BASELINE"; then
      log_global "SUCCESS: '$ATT' already beat baseline. Stopping loop."
      exit 0
    fi
    continue
  fi

  log_global ""
  log_global "> Attempt: $ATT  (postfix=$POSTFIX)"
  python src/train_grading_reti.py \
      --backbone virchow2 --data_root data \
      --epochs "$EPOCHS" --lr 1e-4 --batch_size 1 --seed "$SEED" \
      --num_workers "$NUM_WORKERS" --topk 0 \
      --early_stop_patience 15 \
      --formulation regression --main_metric qwk \
      --device "$DEVICE" \
      --model_type novelty_attempt --novelty_id "$ATT" \
      --postfix "$POSTFIX" \
      --leaderboard \
      > "runs/${ATT}.out" 2>&1
  STATUS=$?
  if [ $STATUS -ne 0 ]; then
    log_global "  Run FAILED (exit=$STATUS); see runs/${ATT}.out"
    continue
  fi
  EXP_DIR=$(ls -td "experiments/reti_novelty_attempt_virchow2_${POSTFIX}_"* \
                    "experiments/"*"/reti_novelty_attempt_virchow2_${POSTFIX}_"* 2>/dev/null | head -1)
  if [ -z "$EXP_DIR" ]; then
    log_global "  Cannot find experiment dir for ${POSTFIX}"; continue
  fi
  RESULT=$(baseline_check "$EXP_DIR")
  log_global "  $RESULT"
  log_global "  exp_dir: $EXP_DIR"
  if [ "$STOP_ON_WIN" = "1" ] && echo "$RESULT" | grep -q "BEATS_BASELINE"; then
    log_global ""
    log_global "SUCCESS: '$ATT' beats the baseline. Stopping loop."
    log_global "Best experiment dir: $EXP_DIR"
    exit 0
  fi
done

log_global ""
log_global "Batch finished. ${#ATTEMPTS[@]} attempt(s) processed."
exit 0

