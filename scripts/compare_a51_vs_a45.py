#!/usr/bin/env python3
"""Paired comparison: a51_rank_norm_mid_mass vs a45_rank_norm_lengthnorm.

Mirrors compare_a45.py but with a45 as the reference instead of baseline.
Decision rule is the H23 kill criterion documented in a51's docstring +
sweep_a51_vs_a45.sh header.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
LEADERBOARDS = [
    REPO / "results" / "leaderboard_v2.csv",
    REPO / "results" / "leaderboard.csv",
]
A45_NEEDLE = "_virchow2_a45_rank_norm_lengthnorm_s"
A51_NEEDLE = "_virchow2_a51_rank_norm_mid_mass_s"
GRADES = ("G0", "G1", "G2", "G3")


def load_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for lb in LEADERBOARDS:
        if not lb.exists():
            continue
        with lb.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def latest_by_seed(rows: List[Dict[str, str]], needle: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        name = row.get("run_name", "")
        if needle not in name:
            continue
        try:
            seed = int(name.split(needle, 1)[1])
        except ValueError:
            continue
        prev = out.get(seed)
        if prev is None or row.get("timestamp", "") > prev.get("timestamp", ""):
            out[seed] = row
    return out


def metrics_bundle(row: Optional[Dict[str, str]]) -> Optional[Dict[str, float]]:
    if not row:
        return None
    out: Dict[str, float] = {}
    for k in ("val_qwk", "test_qwk", "test_accuracy", "test_macro_recall"):
        v = row.get(k, "")
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    p = Path(row.get("experiment_dir", "")) / "test_metrics.json"
    if p.exists():
        try:
            tj = json.loads(p.read_text())
            for k in ("test_qwk", "test_accuracy", "test_macro_recall"):
                if k not in out and k in tj:
                    out[k] = float(tj[k])
            rc = tj.get("test_recall_per_class", {})
            for g in GRADES:
                if g in rc:
                    out[g] = float(rc[g])
        except Exception:
            pass
    if "val_qwk" not in out or "test_qwk" not in out:
        return None
    return out


def fmt(x: Optional[float], width: int = 7, prec: int = 4) -> str:
    if x is None:
        return "  --  ".rjust(width)
    return f"{x:>{width}.{prec}f}"


def main() -> int:
    rows = load_rows()
    a45 = latest_by_seed(rows, A45_NEEDLE)
    a51 = latest_by_seed(rows, A51_NEEDLE)
    seeds = sorted(set(a45) & set(a51))
    if not seeds:
        print("[compare_a51_vs_a45] No paired seeds found yet.")
        return 1

    print()
    print(f"Paired comparison: a51 (mid-mass) vs a45 (baseline aggregator) — N={len(seeds)}")
    print("=" * 116)
    print(f"  {'seed':>4} | "
          f"{'a45 vqwk':>9} {'a51 vqwk':>9} {'Δ':>7} | "
          f"{'a45 tqwk':>9} {'a51 tqwk':>9} {'Δ':>7} | "
          f"{'a45 mR':>7} {'a51 mR':>7} {'Δ':>6} | "
          f"{'a45 G1':>7} {'a51 G1':>7} {'Δ':>6} | "
          f"{'a45 G3':>7} {'a51 G3':>7} {'Δ':>6}")
    print("-" * 116)

    d_val: List[float] = []
    d_test: List[float] = []
    d_mr: List[float] = []
    d_g1: List[float] = []
    d_g3: List[float] = []
    n_macro_wins = 0
    n_g1_wins = 0
    for s in seeds:
        m45 = metrics_bundle(a45.get(s))
        m51 = metrics_bundle(a51.get(s))
        if not m45 or not m51:
            continue
        dv = m51["val_qwk"] - m45["val_qwk"]
        dt = m51["test_qwk"] - m45["test_qwk"]
        d_val.append(dv); d_test.append(dt)
        mr_a = m45.get("test_macro_recall"); mr_b = m51.get("test_macro_recall")
        g1_a = m45.get("G1"); g1_b = m51.get("G1")
        g3_a = m45.get("G3"); g3_b = m51.get("G3")
        dmr = (mr_b - mr_a) if (mr_a is not None and mr_b is not None) else None
        dg1 = (g1_b - g1_a) if (g1_a is not None and g1_b is not None) else None
        dg3 = (g3_b - g3_a) if (g3_a is not None and g3_b is not None) else None
        if dmr is not None:
            d_mr.append(dmr)
            if dmr > 0:
                n_macro_wins += 1
        if dg1 is not None:
            d_g1.append(dg1)
            if dg1 > 0:
                n_g1_wins += 1
        if dg3 is not None:
            d_g3.append(dg3)
        print(f"  {s:>4} | "
              f"{m45['val_qwk']:>9.4f} {m51['val_qwk']:>9.4f} {dv:>+7.4f} | "
              f"{m45['test_qwk']:>9.4f} {m51['test_qwk']:>9.4f} {dt:>+7.4f} | "
              f"{fmt(mr_a, 7, 2)} {fmt(mr_b, 7, 2)} {fmt(dmr, 6, 2)} | "
              f"{fmt(g1_a, 7, 1)} {fmt(g1_b, 7, 1)} {fmt(dg1, 6, 1)} | "
              f"{fmt(g3_a, 7, 1)} {fmt(g3_b, 7, 1)} {fmt(dg3, 6, 1)}")
    print("=" * 116)

    def summarize(name: str, ds: List[float], prec: int = 4) -> None:
        if not ds:
            return
        med = statistics.median(ds)
        mean = statistics.mean(ds)
        std = statistics.pstdev(ds) if len(ds) > 1 else 0.0
        print(f"  Δ {name:<14}: median={med:+.{prec}f}  mean={mean:+.{prec}f}  "
              f"std={std:.{prec}f}  n={len(ds)}")
    summarize("val_qwk", d_val)
    summarize("test_qwk", d_test)
    summarize("macro_recall", d_mr, 2)
    summarize("G1 recall", d_g1, 2)
    summarize("G3 recall", d_g3, 2)
    print(f"  a51 wins macro_recall : {n_macro_wins}/{len(d_mr)}")
    print(f"  a51 wins G1 recall    : {n_g1_wins}/{len(d_g1)}")
    print()

    # Decision rule (pre-registered)
    print("Decision (pre-registered H23 kill criterion):")
    if not d_mr or not d_g1 or not d_g3:
        print("  inconclusive: missing recall data.")
        return 0
    mr_med = statistics.median(d_mr)
    g1_med = statistics.median(d_g1)
    g3_med = statistics.median(d_g3)
    if n_macro_wins <= 1 and len(d_mr) >= 5:
        verdict = ("HARD STOP -> a51 >= a45 on macro recall at "
                   f"{n_macro_wins}/{len(d_mr)} seeds. Close H23, lock a45.")
    elif mr_med > 1.0 and g1_med > 2.0 and g3_med >= -2.0:
        verdict = (f"WIN -> macro_recall +{mr_med:.2f}, G1 +{g1_med:.2f}, "
                   f"G3 {g3_med:+.2f}. a51 dominates a45 — promote to thesis "
                   "and consider 10-seed extension.")
    elif mr_med <= 0 or g3_med <= -2.0:
        verdict = (f"LOSE -> macro_recall {mr_med:+.2f}, G3 {g3_med:+.2f}. "
                   "a51 fails kill criterion; discard, lock a45.")
    else:
        verdict = (f"TIE -> macro_recall {mr_med:+.2f}, G1 {g1_med:+.2f}, "
                   f"G3 {g3_med:+.2f}. Inconclusive; report both.")
    print(f"  {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

