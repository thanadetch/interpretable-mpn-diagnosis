#!/usr/bin/env python3
"""Consolidated paired comparison: a45_rank_norm_lengthnorm vs ABMIL baseline.

Single script that prints (in order):
  1. Per-seed QWK paired table + summary + pre-registered decision rule
  2. Per-seed test accuracy + macro recall paired table
  3. Per-seed per-class test recall (G0..G3) paired table + per-grade summary
  4. Aggregated recall over paired seeds (mean ± std)
  5. Aggregated recall over ALL available seeds (unpaired-friendly)
  6. Wide "all seeds" view with per-class + macro recall side-by-side

Inputs:
  - results/leaderboard_v2.csv  (preferred; has inline test_qwk/acc/macro_recall)
  - results/leaderboard.csv     (legacy fallback)
  - test_metrics.json under each row's experiment_dir (for per-class recalls)
  - Hardcoded BASELINE_LOCKED_S2 row for the seed=2 locked baseline that
    never landed in either leaderboard csv.

Read-only.
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

A45_NEEDLE = "_virchow2_a45_rank_norm_lengthnorm_s"
BASELINE_NEEDLE = "reti_simple_virchow2_simple_virchow2_regression_s"

# Locked seed=2 baseline (manual run, predates leaderboard schema; metrics
# from copilot-instructions + test_metrics.json on disk).
BASELINE_LOCKED_S2 = {
    "val_qwk": "0.8182",
    "test_qwk": "0.9476017067608328",
    "experiment_dir": str(
        REPO
        / "experiments"
        / "20260523"
        / "04_reti_simple_virchow2_regression_20260523_004342"
    ),
    "run_name": "reti_simple_virchow2_regression",
    "timestamp": "20260523_004342",
}

# Pre-registered decision gates (locked seed=2 baseline targets).
TARGET_VAL = 0.8182
TARGET_TEST = 0.9476

GRADES = ("G0", "G1", "G2", "G3")


# ────────────────────────── I/O helpers ──────────────────────────────────
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


def latest_by_seed(rows: List[Dict[str, str]], needle: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        name = row.get("run_name", "")
        if needle not in name:
            continue
        suffix = name.split(needle, 1)[1]
        try:
            seed = int(suffix)
        except ValueError:
            continue
        prev = out.get(seed)
        if prev is None or row.get("timestamp", "") > prev.get("timestamp", ""):
            out[seed] = row
    return out


def row_test_metrics(row: Dict[str, str]) -> Dict[str, float]:
    """Return all available test metrics (scalars + per-class recalls flattened
    as `test_recall_per_class.Gn`)."""
    out: Dict[str, float] = {}
    for k in ("test_qwk", "test_accuracy", "test_mae", "test_macro_recall"):
        v = row.get(k, "")
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    exp_dir = row.get("experiment_dir", "")
    p = Path(exp_dir) / "test_metrics.json"
    if p.exists():
        try:
            tj = json.loads(p.read_text())
            for k, v in tj.items():
                if k in out:
                    continue
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        try:
                            out[f"{k}.{sub_k}"] = float(sub_v)
                        except (TypeError, ValueError):
                            pass
                else:
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
    return out


def get_val_test(row: Optional[Dict[str, str]]) -> Tuple[Optional[float], Optional[float]]:
    if row is None:
        return None, None
    try:
        v = float(row.get("val_qwk", "nan"))
    except ValueError:
        v = None
    tm = row_test_metrics(row)
    t = tm.get("test_qwk")
    return v, t


def get_bundle(row: Optional[Dict[str, str]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "val_qwk": None,
        "test_qwk": None,
        "test_accuracy": None,
        "test_macro_recall": None,
        **{g: None for g in GRADES},
    }
    if row is None:
        return out
    try:
        out["val_qwk"] = float(row.get("val_qwk", "nan"))
    except ValueError:
        pass
    tm = row_test_metrics(row)
    for k in ("test_qwk", "test_accuracy", "test_macro_recall"):
        if k in tm:
            out[k] = tm[k]
    for g in GRADES:
        v = tm.get(f"test_recall_per_class.{g}")
        if v is not None:
            out[g] = v
    return out


def per_class_recall(row: Optional[Dict[str, str]]) -> Optional[Dict[str, float]]:
    """Return {G0,G1,G2,G3: recall} or None."""
    if not row:
        return None
    tm = row_test_metrics(row)
    out = {}
    for g in GRADES:
        v = tm.get(f"test_recall_per_class.{g}")
        if v is None:
            return None
        out[g] = v
    return out


def fmt(x: Optional[float], width: int = 6, prec: int = 4) -> str:
    if x is None:
        return "  --  ".rjust(width)
    return f"{x:>{width}.{prec}f}"


# ────────────────────────── Sections ─────────────────────────────────────
def section_qwk(
    base: Dict[int, Dict[str, str]], a45: Dict[int, Dict[str, str]]
) -> None:
    all_seeds = sorted(set(a45) | set(base))
    print()
    print("[1] QWK paired comparison: a45_rank_norm_lengthnorm vs ABMIL baseline")
    print("-" * 102)
    print(f"  {'seed':>4}  "
          f"{'base_val':>9} {'a45_val':>8} {'Δval':>8}  "
          f"{'base_test':>10} {'a45_test':>9} {'Δtest':>8}  "
          f"{'winner':>10}")
    print("-" * 102)

    dval: List[float] = []
    dtest: List[float] = []
    n_val_win = n_test_win = n_joint_win = n_joint_loss = 0
    n_pairs = 0
    for s in all_seeds:
        bv, bt = get_val_test(base.get(s))
        av, at = get_val_test(a45.get(s))
        if bv is None or av is None or bt is None or at is None:
            label = "missing"
        else:
            n_pairs += 1
            dv = av - bv
            dt = at - bt
            dval.append(dv)
            dtest.append(dt)
            val_win = dv > 0
            test_win = dt > 0
            if val_win:
                n_val_win += 1
            if test_win:
                n_test_win += 1
            if val_win and test_win:
                n_joint_win += 1
                label = "a45 BOTH"
            elif (not val_win) and (not test_win):
                n_joint_loss += 1
                label = "base BOTH"
            elif val_win:
                label = "a45 val"
            else:
                label = "a45 test"
        d_v_str = fmt(av - bv, prec=4) if (av is not None and bv is not None) else "   --   "
        d_t_str = fmt(at - bt, prec=4) if (at is not None and bt is not None) else "   --   "
        print(f"  {s:>4}  "
              f"{fmt(bv):>9} {fmt(av):>8} {d_v_str:>8}  "
              f"{fmt(bt):>10} {fmt(at):>9} {d_t_str:>8}  "
              f"{label:>10}")
    print("-" * 102)
    if n_pairs == 0:
        print("  no complete pairs to summarise.")
        return

    print(f"  pairs                 : {n_pairs}")
    print(f"  a45 wins val_qwk      : {n_val_win}/{n_pairs}")
    print(f"  a45 wins test_qwk     : {n_test_win}/{n_pairs}")
    print(f"  a45 wins BOTH (joint) : {n_joint_win}/{n_pairs}")
    print(f"  baseline wins BOTH    : {n_joint_loss}/{n_pairs}")
    if dval:
        print(f"  Δval_qwk  : median={statistics.median(dval):+.4f}  "
              f"mean={statistics.mean(dval):+.4f}  "
              f"std={statistics.pstdev(dval) if len(dval)>1 else 0.0:.4f}")
    if dtest:
        print(f"  Δtest_qwk : median={statistics.median(dtest):+.4f}  "
              f"mean={statistics.mean(dtest):+.4f}  "
              f"std={statistics.pstdev(dtest) if len(dtest)>1 else 0.0:.4f}")

    print()
    print("Decision (pre-registered):")
    if n_joint_win >= max(3, (n_pairs + 1) // 2 + 1):
        print(f"  WIN          -> {n_joint_win}/{n_pairs} joint wins. "
              "a45 is a defensible per-seed winner. Lock as thesis aggregator.")
    elif n_joint_win == 2 or (n_pairs >= 4 and n_joint_win == (n_pairs // 2)):
        print(f"  PARETO       -> {n_joint_win}/{n_pairs} joint wins. "
              "Inconclusive; report both single-seed and per-seed comparison.")
    elif n_joint_win <= 1:
        print(f"  INDISTINCT   -> {n_joint_win}/{n_pairs} joint wins. "
              "a45 indistinguishable from baseline under seed noise.")
    else:
        print(f"  INCONCLUSIVE -> {n_joint_win}/{n_pairs} joint wins.")
    print()


def section_acc_and_macro(
    base: Dict[int, Dict[str, str]], a45: Dict[int, Dict[str, str]]
) -> None:
    all_seeds = sorted(set(a45) | set(base))
    print("[2] Test accuracy + macro recall (paired per seed)")
    print("-" * 96)
    print(f"  {'seed':>4}  "
          f"{'base_acc':>9} {'a45_acc':>8} {'Δacc':>8}  "
          f"{'base_mR':>9} {'a45_mR':>8} {'ΔmR':>8}")
    print("-" * 96)
    dacc: List[float] = []
    dmr: List[float] = []
    for s in all_seeds:
        bb = get_bundle(base.get(s))
        aa = get_bundle(a45.get(s))
        ba, ab = bb["test_accuracy"], aa["test_accuracy"]
        bm, am = bb["test_macro_recall"], aa["test_macro_recall"]
        if ba is not None and ab is not None:
            dacc.append(ab - ba)
        if bm is not None and am is not None:
            dmr.append(am - bm)
        da = (ab - ba) if (ba is not None and ab is not None) else None
        dm = (am - bm) if (bm is not None and am is not None) else None
        print(f"  {s:>4}  "
              f"{fmt(ba, prec=2):>9} {fmt(ab, prec=2):>8} {fmt(da, prec=2):>8}  "
              f"{fmt(bm, prec=2):>9} {fmt(am, prec=2):>8} {fmt(dm, prec=2):>8}")
    print("-" * 96)
    if dacc:
        print(f"  Δtest_acc          : median={statistics.median(dacc):+.2f}  "
              f"mean={statistics.mean(dacc):+.2f}  "
              f"std={statistics.pstdev(dacc) if len(dacc)>1 else 0.0:.2f}")
    if dmr:
        print(f"  Δtest_macro_recall : median={statistics.median(dmr):+.2f}  "
              f"mean={statistics.mean(dmr):+.2f}  "
              f"std={statistics.pstdev(dmr) if len(dmr)>1 else 0.0:.2f}")
    print()


def section_per_class_recall(
    base: Dict[int, Dict[str, str]], a45: Dict[int, Dict[str, str]]
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Print Δ per-class recall per seed; return macro_deltas + grade_deltas."""
    all_seeds = sorted(set(a45) | set(base))
    print("[3] Per-class test recall — paired per seed (positive Δ = a45 better)")
    print("-" * 96)
    print(f"  {'seed':>4}  "
          f"{'base G0':>8} {'a45 G0':>8} {'Δ':>6} | "
          f"{'base G1':>8} {'a45 G1':>8} {'Δ':>6}")
    print(f"  {'    ':>4}  "
          f"{'base G2':>8} {'a45 G2':>8} {'Δ':>6} | "
          f"{'base G3':>8} {'a45 G3':>8} {'Δ':>6}")
    print("-" * 96)
    grade_deltas: Dict[str, List[float]] = {g: [] for g in GRADES}
    grade_wins: Dict[str, int] = {g: 0 for g in GRADES}
    grade_ties: Dict[str, int] = {g: 0 for g in GRADES}
    macro_deltas: List[float] = []
    macro_wins = 0
    n_pairs = 0
    for s in all_seeds:
        br = per_class_recall(base.get(s))
        ar = per_class_recall(a45.get(s))
        if br is None or ar is None:
            continue
        n_pairs += 1
        cells: Dict[str, Tuple[str, str, str]] = {}
        for g in GRADES:
            d = ar[g] - br[g]
            grade_deltas[g].append(d)
            if d > 0.5:
                grade_wins[g] += 1
            elif abs(d) <= 0.5:
                grade_ties[g] += 1
            cells[g] = (fmt(br[g], prec=1), fmt(ar[g], prec=1), fmt(d, prec=1))
        bm = sum(br.values()) / 4
        am = sum(ar.values()) / 4
        macro_deltas.append(am - bm)
        if am > bm:
            macro_wins += 1
        print(f"  {s:>4}  "
              f"{cells['G0'][0]:>8} {cells['G0'][1]:>8} {cells['G0'][2]:>6} | "
              f"{cells['G1'][0]:>8} {cells['G1'][1]:>8} {cells['G1'][2]:>6}")
        print(f"  {'    ':>4}  "
              f"{cells['G2'][0]:>8} {cells['G2'][1]:>8} {cells['G2'][2]:>6} | "
              f"{cells['G3'][0]:>8} {cells['G3'][1]:>8} {cells['G3'][2]:>6}")
        print()
    print("-" * 96)
    print("Per-grade summary across paired seeds:")
    print(f"  {'grade':>5}  {'a45 wins':>10}  {'ties':>5}  {'base wins':>10}   "
          f"{'median Δ':>10}  {'mean Δ':>9}  {'std':>6}")
    for g in GRADES:
        ds = grade_deltas[g]
        if not ds:
            continue
        nw = grade_wins[g]
        nt = grade_ties[g]
        nl = len(ds) - nw - nt
        print(f"  {g:>5}  {nw:>4}/{len(ds):<4}   {nt:>5}  {nl:>4}/{len(ds):<6} "
              f"{statistics.median(ds):>+10.2f}  {statistics.mean(ds):>+9.2f}  "
              f"{statistics.pstdev(ds) if len(ds)>1 else 0.0:>6.2f}")
    if macro_deltas:
        print()
        print(f"  Δ macro_recall  : median={statistics.median(macro_deltas):+.2f}  "
              f"mean={statistics.mean(macro_deltas):+.2f}  "
              f"a45 wins {macro_wins}/{len(macro_deltas)} seeds")
    print()
    return macro_deltas, grade_deltas


def section_aggregated_paired(
    base: Dict[int, Dict[str, str]], a45: Dict[int, Dict[str, str]]
) -> int:
    seeds = sorted(set(base) & set(a45))
    by_grade_b = {g: [] for g in GRADES}
    by_grade_a = {g: [] for g in GRADES}
    macro_b: List[float] = []
    macro_a: List[float] = []
    for s in seeds:
        br = per_class_recall(base.get(s))
        ar = per_class_recall(a45.get(s))
        if br is None or ar is None:
            continue
        for g in GRADES:
            by_grade_b[g].append(br[g])
            by_grade_a[g].append(ar[g])
        macro_b.append(sum(br.values()) / 4)
        macro_a.append(sum(ar.values()) / 4)
    n = len(macro_b)
    print(f"[4] Aggregated recall — PAIRED (n={n})")
    print("=" * 78)
    print(f"  {'grade':>6}  {'baseline':>18}  {'a45':>18}  {'Δ mean':>10}")
    print("-" * 78)
    for g in GRADES:
        bs = by_grade_b[g]
        as_ = by_grade_a[g]
        if not bs:
            continue
        bm = statistics.mean(bs)
        bsd = statistics.pstdev(bs) if len(bs) > 1 else 0.0
        am = statistics.mean(as_)
        asd = statistics.pstdev(as_) if len(as_) > 1 else 0.0
        print(f"  {g:>6}  {bm:>8.2f} ± {bsd:>6.2f}   {am:>8.2f} ± {asd:>6.2f}   {am-bm:>+10.2f}")
    print("-" * 78)
    if macro_b:
        bmM = statistics.mean(macro_b)
        bsM = statistics.pstdev(macro_b) if n > 1 else 0.0
        amM = statistics.mean(macro_a)
        asM = statistics.pstdev(macro_a) if n > 1 else 0.0
        print(f"  {'macro':>6}  {bmM:>8.2f} ± {bsM:>6.2f}   "
              f"{amM:>8.2f} ± {asM:>6.2f}   {amM-bmM:>+10.2f}")
    print("=" * 78)
    print()
    return n


def section_aggregated_unpaired(
    base: Dict[int, Dict[str, str]], a45: Dict[int, Dict[str, str]], n_paired: int
) -> Tuple[int, int]:
    by_grade_b = {g: [] for g in GRADES}
    by_grade_a = {g: [] for g in GRADES}
    macro_b: List[float] = []
    macro_a: List[float] = []
    for s in sorted(set(base) | set(a45)):
        br = per_class_recall(base.get(s))
        ar = per_class_recall(a45.get(s))
        if br is not None:
            for g in GRADES:
                by_grade_b[g].append(br[g])
            macro_b.append(sum(br.values()) / 4)
        if ar is not None:
            for g in GRADES:
                by_grade_a[g].append(ar[g])
            macro_a.append(sum(ar.values()) / 4)
    nb, na = len(macro_b), len(macro_a)
    print(f"[5] Aggregated recall — UNPAIRED (baseline n={nb}, a45 n={na})")
    print("=" * 78)
    print(f"  {'grade':>6}  {'baseline (all)':>22}  {'a45 (all)':>22}  {'Δ mean':>8}")
    print("-" * 78)
    for g in GRADES:
        bs = by_grade_b[g]
        as_ = by_grade_a[g]
        bm = statistics.mean(bs) if bs else 0
        bsd = statistics.pstdev(bs) if len(bs) > 1 else 0.0
        am = statistics.mean(as_) if as_ else 0
        asd = statistics.pstdev(as_) if len(as_) > 1 else 0.0
        print(f"  {g:>6}  {bm:>8.2f}±{bsd:>6.2f} (n={len(bs)})   "
              f"{am:>8.2f}±{asd:>6.2f} (n={len(as_)})   {am-bm:>+8.2f}")
    print("-" * 78)
    if macro_b and macro_a:
        bmM = statistics.mean(macro_b)
        bsM = statistics.pstdev(macro_b) if nb > 1 else 0.0
        amM = statistics.mean(macro_a)
        asM = statistics.pstdev(macro_a) if na > 1 else 0.0
        print(f"  {'macro':>6}  {bmM:>8.2f}±{bsM:>6.2f} (n={nb})   "
              f"{amM:>8.2f}±{asM:>6.2f} (n={na})   {amM-bmM:>+8.2f}")
    print("=" * 78)
    print()
    print(f"Coverage: paired={n_paired}  baseline-only={nb - n_paired}  "
          f"a45-only={na - n_paired}")
    print()
    return nb, na


def section_all_seeds_wide(
    base: Dict[int, Dict[str, str]], a45: Dict[int, Dict[str, str]]
) -> None:
    all_seeds = sorted(set(base) | set(a45))
    print(f"[6] Per-class test recall + macro — ALL seeds (n={len(all_seeds)})")
    print("=" * 116)
    header = f"  {'seed':>4} |"
    for g in (*GRADES, "macro"):
        header += f"  {g+' base':>8} {g+' a45':>8} {'Δ':>6}"
    print(header)
    print("-" * 116)
    for s in all_seeds:
        br = per_class_recall(base.get(s))
        ar = per_class_recall(a45.get(s))
        row = f"  {s:>4} |"
        for g in GRADES:
            b_str = f"{br[g]:.1f}" if br else "  --  "
            a_str = f"{ar[g]:.1f}" if ar else "  --  "
            d_str = f"{ar[g] - br[g]:+.1f}" if (br and ar) else "  -- "
            row += f"  {b_str:>8} {a_str:>8} {d_str:>6}"
        bm = sum(br.values()) / 4 if br else None
        am = sum(ar.values()) / 4 if ar else None
        bm_str = f"{bm:.1f}" if bm is not None else "  --  "
        am_str = f"{am:.1f}" if am is not None else "  --  "
        dm_str = f"{am-bm:+.1f}" if (bm is not None and am is not None) else "  -- "
        row += f"  {bm_str:>8} {am_str:>8} {dm_str:>6}"
        if br and ar:
            tag = ""
        elif br:
            tag = "  [base only]"
        else:
            tag = "  [a45 only]"
        print(row + tag)
    print("=" * 116)
    print()


# ────────────────────────── Main ─────────────────────────────────────────
def main() -> int:
    rows = load_rows()
    a45 = latest_by_seed(rows, A45_NEEDLE)
    base = latest_by_seed(rows, BASELINE_NEEDLE)
    # Merge in the locked seed=2 baseline (synthetic row).
    if 2 not in base:
        base[2] = dict(BASELINE_LOCKED_S2)

    if not a45 and not base:
        print("[compare_a45] No matching rows found.")
        return 1

    section_qwk(base, a45)
    section_acc_and_macro(base, a45)
    section_per_class_recall(base, a45)
    n_paired = section_aggregated_paired(base, a45)
    section_aggregated_unpaired(base, a45, n_paired)
    section_all_seeds_wide(base, a45)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

