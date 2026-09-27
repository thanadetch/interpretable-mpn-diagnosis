#!/usr/bin/env python3
"""Pool a patient-level k-fold CV sweep into ONE estimate over the whole cohort.

Why pooling and not averaging fold scores: with --n_folds every patient is predicted exactly
once, as test, in exactly one fold. Concatenating the folds' test predictions therefore gives a
single out-of-fold prediction for all 50 patients, which is the estimate to report - and unlike
averaging per-fold QWK it does not break when a fold's test set happens to contain only two or
three grades.

Both units are reported:
    ROI level      1330 out-of-fold ROI predictions
    PATIENT level  50 out-of-fold patient decisions (mean of the case's ROI scores)

A patient-level bootstrap (resampling PATIENTS, the independent unit) gives the CI on the
difference between two models.

Usage:
    python scripts/collect_cv.py --prefix_a cvas --prefix_b cvab --label_a ASGAP --label_b ABMIL
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from train_grading_reti import BACKBONE_CONFIG, patient_kfold_split  # noqa: E402

_DS: dict = {}


def qwk(a, b) -> float:
    return float(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3]))


def macro_recall(y, p) -> float:
    rs = [(p[y == g] == g).mean() * 100 for g in range(4) if (y == g).any()]
    return float(np.mean(rs)) if rs else float("nan")


def dataset(backbone: str, data_root: str) -> GradingBagDatasetFull:
    key = (backbone, data_root)
    if key not in _DS:
        _DS[key] = GradingBagDatasetFull(Path(data_root) / BACKBONE_CONFIG[backbone]["feature_dir"])
    return _DS[key]


def gather(prefix: str):
    """{backbone: (patients, y, raw)} concatenated over folds, out-of-fold."""
    out = defaultdict(lambda: ([], [], []))
    for cf in sorted(glob.glob(str(ROOT / "experiments/*/*/config.json"))):
        c = json.load(open(cf))
        if not str(c.get("prefix", "")).startswith(prefix) or not c.get("n_folds"):
            continue
        d = Path(cf).parent
        csv = d / "test_predictions.csv"
        if not csv.exists():
            continue
        ds = dataset(c["backbone"], c.get("data_root", "data"))
        _, _, te = patient_kfold_split(ds, n_folds=c["n_folds"], fold=c["fold"],
                                       seed=c.get("fold_seed", 2))
        df = pd.read_csv(csv)
        if len(df) != len(te) or list(df.label_idx) != [ds.samples[i][1] for i in te]:
            print(f"  ⚠ skipping {d.name}: predictions do not match its fold")
            continue
        P, Y, R = out[c["backbone"]]
        P.extend(ds.get_slide_path(i).parent.name for i in te)
        Y.extend(df.label_idx.tolist())
        R.extend(df.raw_output.tolist())
    return {k: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for k, v in out.items()}


def patient_level(P, Y, R):
    pl = sorted(set(P))
    pm = np.array([R[P == p].mean() for p in pl])
    py = np.array([Y[P == p][0] for p in pl])
    pp = np.clip(np.round(pm), 0, 3).astype(int)
    return np.array(pl), py, pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix_a", default="cvas")
    ap.add_argument("--prefix_b", default="cvab")
    ap.add_argument("--label_a", default="ASGAP")
    ap.add_argument("--label_b", default="ABMIL")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    A, B = gather(a.prefix_a), gather(a.prefix_b)
    rng = np.random.default_rng(0)

    print(f"\nOut-of-fold results over the WHOLE cohort (every patient predicted exactly once)\n")
    hdr = (f"{'backbone':9s} {'model':8s} {"pat":>5s} {'ROIs':>5s} | {'ROI QWK':>8s} {'ROI mRec':>9s}"
           f" | {'PAT QWK':>8s} {'PAT acc':>9s}")
    print(hdr); print("-" * len(hdr))
    store = {}
    for bb in sorted(set(A) | set(B)):
        for lbl, G in ((a.label_a, A), (a.label_b, B)):
            if bb not in G:
                continue
            P, Y, R = G[bb]
            roi = np.clip(np.round(R), 0, 3).astype(int)
            pl, py, pp = patient_level(P, Y, R)
            store[(bb, lbl)] = (P, Y, R)
            print(f"{bb:9s} {lbl:8s} {len(pl):5d} {len(Y):5d} | "
                  f"{qwk(Y, roi):8.4f} {macro_recall(Y, roi):9.1f} | "
                  f"{qwk(py, pp):8.4f} {f'{int((pp == py).sum())}/{len(pl)}':>9s}")
        print()

    print(f"Patient-level bootstrap of {a.label_a} - {a.label_b} ({a.boot:,} resamples of the cohort)\n")
    hdr2 = f"{'backbone':9s} {'metric':10s} {'Δ':>9s} {'95% CI':>22s} {'P(A>B)':>8s}"
    print(hdr2); print("-" * len(hdr2))
    for bb in sorted(set(A) & set(B)):
        Pa, Ya, Ra = store[(bb, a.label_a)]
        Pb, Yb, Rb = store[(bb, a.label_b)]
        pl = sorted(set(Pa) & set(Pb))
        ia = {p: np.where(Pa == p)[0] for p in pl}
        ib = {p: np.where(Pb == p)[0] for p in pl}
        for mname in ("ROI QWK", "ROI mRec", "PAT acc"):
            def sc(sel_a, sel_b):
                ra = np.clip(np.round(Ra[sel_a]), 0, 3).astype(int)
                rb = np.clip(np.round(Rb[sel_b]), 0, 3).astype(int)
                if mname == "ROI QWK":
                    return qwk(Ya[sel_a], ra), qwk(Yb[sel_b], rb)
                if mname == "ROI mRec":
                    return macro_recall(Ya[sel_a], ra), macro_recall(Yb[sel_b], rb)
                pa_ = np.array([np.clip(round(Ra[ia[p]].mean()), 0, 3) for p in pl])
                pb_ = np.array([np.clip(round(Rb[ib[p]].mean()), 0, 3) for p in pl])
                ty = np.array([Ya[ia[p]][0] for p in pl])
                return float((pa_ == ty).mean() * 100), float((pb_ == ty).mean() * 100)

            if mname == "PAT acc":
                ty = np.array([Ya[ia[p]][0] for p in pl])
                pa_ = np.array([np.clip(round(Ra[ia[p]].mean()), 0, 3) for p in pl])
                pb_ = np.array([np.clip(round(Rb[ib[p]].mean()), 0, 3) for p in pl])
                oa, ob = float((pa_ == ty).mean() * 100), float((pb_ == ty).mean() * 100)
                d = []
                for _ in range(a.boot):
                    j = rng.integers(0, len(pl), len(pl))
                    d.append(float((pa_[j] == ty[j]).mean() * 100) - float((pb_[j] == ty[j]).mean() * 100))
            else:
                oa, ob = sc(np.arange(len(Ya)), np.arange(len(Yb)))
                d = []
                for _ in range(a.boot):
                    j = rng.integers(0, len(pl), len(pl))
                    sa = np.concatenate([ia[pl[t]] for t in j])
                    sb = np.concatenate([ib[pl[t]] for t in j])
                    va, vb = sc(sa, sb)
                    if np.isfinite(va) and np.isfinite(vb):
                        d.append(va - vb)
            d = np.array(d)
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = " *" if (lo > 0 or hi < 0) else ""
            print(f"{bb:9s} {mname:10s} {oa - ob:+9.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22s} "
                  f"{100 * (d > 0).mean():7.1f}%{sig}")
        print()
    print("  * = 95% CI excludes zero")


if __name__ == "__main__":
    main()
