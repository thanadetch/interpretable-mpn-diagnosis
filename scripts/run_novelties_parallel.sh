#!/usr/bin/env bash
# Run many novelty_attempt modules in PARALLEL at one seed (fast seed-2 screen).
# Frozen-feature training is light + I/O-bound and the feature dir is RAM-cached,
# so several runs overlap well on this 10-core machine.
#
# Usage:
#   scripts/run_novelties_parallel.sh <prefix> <seed> <npar> <id1> [id2 ...]
# Example:
#   scripts/run_novelties_parallel.sh screen 2 4 a58_foo a59_bar a60_baz
#
# Each run writes its own experiment dir (via --prefix) and a log to
# runs/<prefix>_<id>_s<seed>.out. No --leaderboard (won't pollute results/).
set -u
PREFIX="${1:?usage: <prefix> <seed> <npar> <id...>}"; SEED="${2:?need seed}"; NPAR="${3:?need npar}"
shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p runs
export PREFIX SEED
echo "=== parallel run: prefix=$PREFIX seed=$SEED npar=$NPAR n_modules=$# ==="
printf '%s\n' "$@" | xargs -P "$NPAR" -I {} sh -c '
  id="$1"
  echo "[start $(date +%H:%M:%S)] $id"
  python src/train_grading_reti.py \
    --backbone virchow2 --data_root data --epochs 50 --lr 1e-4 --batch_size 1 \
    --num_workers 4 --topk 0 --early_stop_patience 15 \
    --formulation regression --main_metric qwk --device cpu \
    --seed "$SEED" --model_type novelty_attempt --novelty_id "$id" --prefix "${PREFIX}_${id}_s${SEED}" \
    > "runs/${PREFIX}_${id}_s${SEED}.out" 2>&1 \
    && echo "[done  $(date +%H:%M:%S)] $id" || echo "[FAIL ] $id"
' _ {}
echo "=== ALL PARALLEL RUNS DONE (prefix=$PREFIX seed=$SEED) ==="
