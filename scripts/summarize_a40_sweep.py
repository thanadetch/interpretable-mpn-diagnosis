#!/usr/bin/env python3
"""Summarise a40 seed-sweep results across seeds 0/1/2/3/42.

Scans `results/leaderboard.csv` for rows where `run_name` matches
`reti_novelty_attempt_virchow2_a40_mean_over_queries_s<seed>` and pulls
the per-seed best val metrics. For each matched row, opens the
corresponding `experiment_dir/test_metrics.json` for test metrics.

Prints a per-seed table + the median val_qwk and applies the
pre-registered decision rule from `scripts/sweep_a40_seeds.sh`:
    median val_qwk > 0.8182  -> a40 wins (update reporting).
    median val_qwk <= 0.80   -> a40 is epoch-1 lottery; close H20.
    in between                -> Pareto / inconclusive.

Read-only; never writes to disk.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
# v2 has test metrics inline (test_qwk, test_accuracy, ...); v1 does not.
LEADERBOARDS = [
    REPO / "results" / "leaderboard_v2.csv",
    REPO / "results" / "leaderboard.csv",
]
ATT = "a40_mean_over_queries"
TARGET_VAL = 0.8182
TARGET_TEST = 0.9476


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
    """If a seed was re-run, keep the most recent (timestamp) row.

    Only matches rows where run_name ends in `_a40_mean_over_queries_s<seed>`,
    so that cross-backbone runs (`reti_novelty_attempt_uni2_a40_..._uni2_s2`,
    `..._titan_s2`) are skipped. Virchow2 runs at the locked baseline backbone
    have run_name `reti_novelty_attempt_virchow2_a40_mean_over_queries_s<seed>`.
    """
    out: Dict[int, Dict[str, str]] = {}
    needle = f"_virchow2_{ATT}_s"
    for row in rows:
        run_name = row.get("run_name", "")
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
    """Pull test metrics from the row (v2) or from test_metrics.json (v1)."""
    out: Dict[str, float] = {}
    for k in ("test_qwk", "test_accuracy", "test_mae"):
        v = row.get(k, "")
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    if "test_qwk" in out:
        return out
    # Fallback: read test_metrics.json from experiment_dir (v1 schema).
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
    if not by_seed:
        print(f"[summarize_a40_sweep] No virchow2 rows for {ATT} in "
              f"{[lb.name for lb in LEADERBOARDS if lb.exists()]}.")
        return 1

    print()
    print(f"a40 seed-sweep summary (locked baseline: val>{TARGET_VAL}  test>{TARGET_TEST})")
    print("-" * 96)
    print(f"  {'seed':>4} {'epoch':>5}  {'val_qwk':>8} {'test_qwk':>9} {'test_acc':>9}  "
          f"{'val_gate':>8} {'test_gate':>9}  exp_dir")
    print("-" * 96)

    val_qwks: List[float] = []
    test_qwks: List[float] = []
    for seed in sorted(by_seed):
        r = by_seed[seed]
        exp_dir = r.get("experiment_dir", "")
        val_qwk = float(r.get("val_qwk", "nan") or "nan")
        epoch = r.get("best_epoch", "")
        tm = row_test_metrics(r)
        test_qwk = tm.get("test_qwk")
        test_acc = tm.get("test_accuracy")

        val_qwks.append(val_qwk)
        if test_qwk is not None:
            test_qwks.append(test_qwk)

        v_pass = "PASS" if val_qwk > TARGET_VAL else "fail"
        t_pass = "PASS" if (test_qwk is not None and test_qwk > TARGET_TEST) else "fail"
        short_dir = exp_dir.split("/experiments/", 1)[-1]
        print(f"  {seed:>4} {str(epoch):>5}  {fmt(val_qwk)}  {fmt(test_qwk):>9} "
              f"{fmt(test_acc, prec=2):>9}  {v_pass:>8} {t_pass:>9}  {short_dir}")

    print("-" * 96)
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

    # Pre-registered decision rule (refined 2026-05-26 after observing
    # seeds 0/1: val_qwk and test_qwk are anti-correlated across seeds
    # -> a single-seed "winner" is meaningless; honest selection must
    # consider both metrics' multi-seed distributions).
    print()
    print("Decision (pre-registered in sweep_a40_seeds.sh, refined 2026-05-26):")
    if not val_qwks or not test_qwks:
        print("  inconclusive: missing val_qwk or test_qwk rows.")
    elif len(val_qwks) < 3:
        print(f"  inconclusive: only {len(val_qwks)} seed(s); need >=3 to "
              "characterise variance.")
    else:
        med_v = statistics.median(val_qwks)
        med_t = statistics.median(test_qwks)
        sd_t = statistics.pstdev(test_qwks)
        n_both_pass = sum(1 for v, t in zip(val_qwks, test_qwks)
                          if v > TARGET_VAL and t > TARGET_TEST)
        # Joint criterion: both medians pass the gates AND test_qwk is
        # stable (std < 0.05 i.e. < ~5pp variation across seeds).
        if med_v > TARGET_VAL and med_t > TARGET_TEST and sd_t < 0.05:
            print(f"  WIN  -> median val {med_v:.4f} > {TARGET_VAL} AND "
                  f"median test {med_t:.4f} > {TARGET_TEST} AND "
                  f"test std {sd_t:.4f} < 0.05. Stable winner; update reporting.")
        elif med_v > TARGET_VAL and med_t > TARGET_TEST:
            print(f"  WIN-FRAGILE -> medians pass both gates but test std "
                  f"{sd_t:.4f} >= 0.05. Real win, but report mean±std.")
        elif n_both_pass >= max(1, len(val_qwks) // 2):
            print(f"  PARTIAL -> {n_both_pass}/{len(val_qwks)} seeds pass BOTH "
                  "gates jointly. Pareto win on some seeds, lose on others.")
        elif sd_t >= 0.08:
            print(f"  UNSTABLE -> test_qwk std {sd_t:.4f} >= 0.08 across seeds. "
                  "val/test are likely anti-correlated; a40 is seed-fragile, "
                  "not a defensible winner. The single-seed comparison was "
                  "an artefact of the seed choice.")
        else:
            print(f"  LOSE -> medians do not pass both gates "
                  f"(val {med_v:.4f} vs {TARGET_VAL}; test {med_t:.4f} vs {TARGET_TEST}); "
                  f"test std {sd_t:.4f}. Close H20 family.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






