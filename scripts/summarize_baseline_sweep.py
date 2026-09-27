#!/usr/bin/env python3
"""Summarise ABMIL locked-baseline seed sweep (seeds 0/1/2/3/42).

Scans `results/leaderboard_v2.csv` for rows where the run_name matches
`reti_simple_virchow2_regression_s<seed>`. Includes the existing seed=2
locked-baseline row (from experiments/20260523/04_...) so the table covers
{0, 1, 2, 3, 42}.

Pre-registered decision rule from `scripts/sweep_baseline_seeds.sh`:
  test std < 0.04 AND medians within 0.04 of seed=2 -> STABLE.
  test std >= 0.08                                   -> UNSTABLE.
  otherwise                                          -> MARGINAL.

Read-only; never writes to disk.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
LEADERBOARDS = [
    REPO / "results" / "leaderboard_v2.csv",
    REPO / "results" / "leaderboard.csv",
]
# Locked baseline seed=2 numbers from copilot-instructions / batch-20 era.
LOCKED_SEED2_VAL = 0.8182
LOCKED_SEED2_TEST = 0.9476
LOCKED_SEED2_PATH = (
    "experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342"
)


def load_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for lb in LEADERBOARDS:
        if not lb.exists():
            continue
        with lb.open() as f:
            for r in csv.DictReader(f):
                r["_source"] = lb.name
                rows.append(r)
    return rows


def latest_row_per_seed(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    """Match rows whose run_name endswith _simple_virchow2_regression_s<seed>.

    The baseline trainer uses --model_type simple, so run_name is
    `reti_<postfix>` where postfix = `simple_virchow2_regression_s<seed>`.
    """
    out: Dict[int, Dict[str, str]] = {}
    needle = "_simple_virchow2_regression_s"
    for row in rows:
        run_name = row.get("run_name", "")
        # Skip novelty_attempt rows that happen to contain "simple" anywhere.
        if row.get("model_type", "") not in ("simple", ""):
            continue
        if needle not in run_name:
            continue
        suffix = run_name.split(needle, 1)[1]
        try:
            seed = int(suffix)
        except ValueError:
            continue
        prev = out.get(seed)
        if prev is None or row["timestamp"] > prev["timestamp"]:
            out[seed] = row
    return out


def row_test_metrics(row: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ("test_qwk", "test_accuracy", "test_mae"):
        v = row.get(k, "")
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    if "test_qwk" in out:
        return out
    exp_dir = row.get("experiment_dir", "")
    p = Path(exp_dir) / "test_metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return out


def fmt(x: Optional[float], width: int = 6, prec: int = 4) -> str:
    if x is None:
        return "  --  ".rjust(width)
    return f"{x:>{width}.{prec}f}"


def main() -> int:
    rows = load_rows()
    by_seed = latest_row_per_seed(rows)

    # Always include the locked seed=2 baseline (its row in leaderboard_v2.csv
    # may have a different run_name from the postfix convention used here).
    if 2 not in by_seed:
        by_seed[2] = {
            "best_epoch": "5",
            "val_qwk": str(LOCKED_SEED2_VAL),
            "test_qwk": str(LOCKED_SEED2_TEST),
            "test_accuracy": "85.33",
            "experiment_dir": str(REPO / LOCKED_SEED2_PATH),
            "_source": "locked-baseline-hardcoded",
        }

    if not by_seed:
        print("[summarize_baseline_sweep] No matching rows found.")
        return 1

    print()
    print(f"Baseline (SimpleGatedMIL) seed-sweep summary (locked: "
          f"val>{LOCKED_SEED2_VAL}  test>{LOCKED_SEED2_TEST} at seed=2)")
    print("-" * 102)
    print(f"  {'seed':>4} {'epoch':>5}  {'val_qwk':>8} {'test_qwk':>9} {'test_acc':>9}  "
          f"{'val_gate':>8} {'test_gate':>9}  exp_dir")
    print("-" * 102)

    val_qwks: List[float] = []
    test_qwks: List[float] = []
    for seed in sorted(by_seed):
        r = by_seed[seed]
        exp_dir = r.get("experiment_dir", "")
        try:
            val_qwk = float(r.get("val_qwk", "nan"))
        except ValueError:
            val_qwk = float("nan")
        epoch = r.get("best_epoch", "")
        tm = row_test_metrics(r)
        test_qwk = tm.get("test_qwk")
        test_acc = tm.get("test_accuracy")

        val_qwks.append(val_qwk)
        if test_qwk is not None:
            test_qwks.append(test_qwk)

        v_pass = "PASS" if val_qwk > LOCKED_SEED2_VAL else "fail"
        t_pass = "PASS" if (test_qwk is not None and test_qwk > LOCKED_SEED2_TEST) else "fail"
        short_dir = exp_dir.split("/experiments/", 1)[-1]
        print(f"  {seed:>4} {str(epoch):>5}  {fmt(val_qwk)}  {fmt(test_qwk):>9} "
              f"{fmt(test_acc, prec=2):>9}  {v_pass:>8} {t_pass:>9}  {short_dir}")

    print("-" * 102)
    if val_qwks:
        med_v = statistics.median(val_qwks)
        mean_v = statistics.mean(val_qwks)
        sd_v = statistics.pstdev(val_qwks) if len(val_qwks) > 1 else 0.0
        print(f"  val_qwk  : median={med_v:.4f}  mean={mean_v:.4f}  std={sd_v:.4f}  "
              f"n={len(val_qwks)}")
    if test_qwks:
        med_t = statistics.median(test_qwks)
        mean_t = statistics.mean(test_qwks)
        sd_t = statistics.pstdev(test_qwks) if len(test_qwks) > 1 else 0.0
        print(f"  test_qwk : median={med_t:.4f}  mean={mean_t:.4f}  std={sd_t:.4f}  "
              f"n={len(test_qwks)}")

    # Cross-comparison with a40 (from leaderboard_v2.csv if available).
    print()
    print("Cross-comparison with a40 seed sweep:")
    a40_val_std, a40_test_std = _a40_sweep_stats(rows)
    if a40_val_std is not None:
        print(f"  baseline val std = {sd_v:.4f}   |   a40 val std = {a40_val_std:.4f}")
        print(f"  baseline test std= {sd_t:.4f}   |   a40 test std = {a40_test_std:.4f}")

    # Decision rule.
    print()
    print("Decision (pre-registered in sweep_baseline_seeds.sh):")
    if not test_qwks or len(test_qwks) < 3:
        print(f"  inconclusive: only {len(test_qwks)} test_qwk rows; need >=3.")
    else:
        med_v = statistics.median(val_qwks)
        med_t = statistics.median(test_qwks)
        sd_t = statistics.pstdev(test_qwks)
        within_v = abs(med_v - LOCKED_SEED2_VAL) < 0.04
        within_t = abs(med_t - LOCKED_SEED2_TEST) < 0.04
        if sd_t < 0.04 and within_v and within_t:
            print(f"  STABLE -> test std {sd_t:.4f} < 0.04 AND medians within "
                  f"0.04 of seed=2 ({LOCKED_SEED2_VAL}/{LOCKED_SEED2_TEST}). "
                  "Single-seed locked baseline is defensible; add a "
                  "mean+/-std note in §4/§5.")
        elif sd_t >= 0.08:
            print(f"  UNSTABLE -> test std {sd_t:.4f} >= 0.08. Baseline itself "
                  "is seed-fragile on this val cohort. Switch ALL reporting to "
                  "median+/-IQR across seeds; revise the proposal's §4 baseline "
                  "section before defense.")
        else:
            print(f"  MARGINAL -> test std {sd_t:.4f} in [0.04, 0.08). Baseline "
                  "is mildly seed-sensitive; report both single-seed and "
                  "multi-seed numbers and flag this in the limitation.")
    print()
    return 0


def _a40_sweep_stats(rows: List[Dict[str, str]]) -> Tuple[Optional[float], Optional[float]]:
    """Pull a40 val/test std from the same leaderboard (best effort)."""
    needle = "_virchow2_a40_mean_over_queries_s"
    a40_val: List[float] = []
    a40_test: List[float] = []
    seen: Dict[int, str] = {}
    for r in rows:
        run = r.get("run_name", "")
        if needle not in run:
            continue
        suffix = run.split(needle, 1)[1]
        try:
            seed = int(suffix)
        except ValueError:
            continue
        ts = r.get("timestamp", "")
        if seen.get(seed, "") > ts:
            continue
        seen[seed] = ts
        try:
            a40_val.append(float(r.get("val_qwk", "nan")))
        except ValueError:
            pass
        tm = row_test_metrics(r)
        if "test_qwk" in tm:
            a40_test.append(tm["test_qwk"])
    if not a40_val or not a40_test:
        return None, None
    sv = statistics.pstdev(a40_val) if len(a40_val) > 1 else 0.0
    st = statistics.pstdev(a40_test) if len(a40_test) > 1 else 0.0
    return sv, st


if __name__ == "__main__":
    raise SystemExit(main())

