#!/usr/bin/env python3
"""Collect the 24 thesis-ablation runs (12 standard + 12 no_patch) into clean tables.

Reads each run's test_metrics.json + training_log.csv (best val on the selection
metric) from experiments/<DATE>/ and prints markdown tables for the thesis.
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import pandas as pd

DATE = sys.argv[1] if len(sys.argv) > 1 else "20260603"
ROOT = Path(__file__).resolve().parents[1]
# Ablation runs live under experiments/<DATE>_ablation/{patch,no_patch}/ (moved there
# from the flat experiments/<DATE>/ dir). Fall back to the flat layout if needed.
ABL = ROOT / "experiments" / f"{DATE}_ablation"
EXP = ROOT / "experiments" / DATE
GROUP_SUBDIR = {"std": "patch", "no_patch": "no_patch"}

# (group, idx, prefix, backbone, model_type, formulation, main_metric, postfix)
RUNS = [
    ("std", 1, "01", "uni2", "mean_pool", "regression", "qwk", "regression"),
    ("std", 2, "02", "uni2", "simple", "regression", "qwk", "regression"),
    ("std", 3, "03", "virchow2", "mean_pool", "regression", "qwk", "regression"),
    ("std", 4, "04", "virchow2", "simple", "regression", "qwk", "regression"),
    ("std", 5, "05", "titan", "mean_pool", "regression", "qwk", "regression"),
    ("std", 6, "06", "titan", "simple", "regression", "qwk", "regression"),
    ("std", 7, "07", "uni2", "mean_pool", "classification", "macro_recall", "multiclass"),
    ("std", 8, "08", "uni2", "simple", "classification", "macro_recall", "multiclass"),
    ("std", 9, "09", "virchow2", "mean_pool", "classification", "macro_recall", "multiclass"),
    ("std", 10, "10", "virchow2", "simple", "classification", "macro_recall", "multiclass"),
    ("std", 11, "11", "titan", "mean_pool", "classification", "macro_recall", "multiclass"),
    ("std", 12, "12", "titan", "simple", "classification", "macro_recall", "multiclass"),
    ("no_patch", 13, "01", "uni2_no_patch", "mean_pool", "regression", "qwk", "regression"),
    ("no_patch", 14, "02", "uni2_no_patch", "simple", "regression", "qwk", "regression"),
    ("no_patch", 15, "03", "virchow2_no_patch", "mean_pool", "regression", "qwk", "regression"),
    ("no_patch", 16, "04", "virchow2_no_patch", "simple", "regression", "qwk", "regression"),
    ("no_patch", 17, "05", "titan_no_patch", "mean_pool", "regression", "qwk", "regression"),
    ("no_patch", 18, "06", "titan_no_patch", "simple", "regression", "qwk", "regression"),
    ("no_patch", 19, "07", "uni2_no_patch", "mean_pool", "classification", "macro_recall", "multiclass"),
    ("no_patch", 20, "08", "uni2_no_patch", "simple", "classification", "macro_recall", "multiclass"),
    ("no_patch", 21, "09", "virchow2_no_patch", "mean_pool", "classification", "macro_recall", "multiclass"),
    ("no_patch", 22, "10", "virchow2_no_patch", "simple", "classification", "macro_recall", "multiclass"),
    ("no_patch", 23, "11", "titan_no_patch", "mean_pool", "classification", "macro_recall", "multiclass"),
    ("no_patch", 24, "12", "titan_no_patch", "simple", "classification", "macro_recall", "multiclass"),
]


def find_dir(group, prefix, model_type, backbone, postfix):
    name = f"{prefix}_reti_{model_type}_{backbone}_{postfix}_*"
    # preferred: experiments/<DATE>_ablation/{patch,no_patch}/<run>
    bases = [ABL / GROUP_SUBDIR.get(group, ""), EXP]  # fall back to flat layout
    for base in bases:
        cands = sorted(glob.glob(str(base / name)), reverse=True)  # newest first
        if cands:
            return Path(cands[0])
    return None


def collect():
    rows = []
    for group, idx, prefix, bb, mt, form, metric, postfix in RUNS:
        d = find_dir(group, prefix, mt, bb, postfix)
        rec = dict(idx=idx, group=group, backbone=bb, model=mt, formulation=form)
        if d and (d / "test_metrics.json").is_file():
            tm = json.load(open(d / "test_metrics.json"))
            rec["test_qwk"] = tm.get("test_qwk")
            rec["test_acc"] = tm.get("test_accuracy")
            rec["test_macro_recall"] = tm.get("test_macro_recall")
            rec["test_mae"] = tm.get("test_mae")
            pc = tm.get("test_recall_per_class", {})
            for g in ("G0", "G1", "G2", "G3"):
                rec[f"test_{g}"] = pc.get(g)
            try:
                df = pd.read_csv(d / "training_log.csv")
                col = "val_qwk" if metric == "qwk" else f"val_{metric}"
                rec["best_val"] = df[col].max() if col in df else None
            except Exception:
                rec["best_val"] = None
            rec["dir"] = d.name
        else:
            rec["status"] = "MISSING" if d is None else "no test_metrics (incomplete)"
        rows.append(rec)
    return rows


def fmt(x, nd=4, scale=1.0):
    if x is None:
        return "—"
    try:
        return f"{x*scale:.{nd}f}"
    except Exception:
        return str(x)


def table(rows, form):
    sub = [r for r in rows if r["formulation"] == form]
    sel = "val_qwk" if form == "regression" else "val_macro_recall"
    lines = [
        f"| backbone | model | best_{sel} | test_qwk | test_acc% | test_macro_rec% | test_mae | G0 | G1 | G2 | G3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sub:
        if "test_qwk" not in r:
            lines.append(f"| {r['backbone']} | {r['model']} | {r.get('status','—')} |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {r['backbone']} | {r['model']} | {fmt(r.get('best_val'))} | "
            f"{fmt(r['test_qwk'])} | {fmt(r['test_acc'],2)} | {fmt(r['test_macro_recall'],2)} | "
            f"{fmt(r['test_mae'])} | {fmt(r['test_G0'],1)} | {fmt(r['test_G1'],1)} | "
            f"{fmt(r['test_G2'],1)} | {fmt(r['test_G3'],1)} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = collect()
    done = sum(1 for r in rows if "test_qwk" in r)
    print(f"# Thesis ablation grid ({DATE}, local Mac, seed=2) — {done}/24 runs collected\n")
    for grp, label in [("std", "Standard features (patch grid)"), ("no_patch", "Resize features (no_patch)")]:
        print(f"\n## {label}\n")
        grprows = [r for r in rows if r["group"] == grp]
        print("### Regression (SmoothL1, select on val_qwk)\n")
        print(table(grprows, "regression"))
        print("\n### Classification (CE+LS, select on val_macro_recall)\n")
        print(table(grprows, "classification"))
    missing = [r["idx"] for r in rows if "test_qwk" not in r]
    if missing:
        print(f"\n> Not yet complete: runs {missing}")
