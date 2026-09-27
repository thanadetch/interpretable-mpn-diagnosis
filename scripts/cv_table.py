#!/usr/bin/env python3
"""One out-of-fold table for every model in the patient-level CV sweep.

Each `--prefix` names one model's five fold-runs (`--n_folds 5 --fold 0..4`). Their test
predictions are concatenated, giving one out-of-fold prediction per ROI and, via the case mean,
one per patient — so every row is an estimate over the WHOLE 50-patient cohort rather than over
a single 10-patient split.

    python scripts/cv_table.py cvab=ABMIL cvas=ASGAP cvmp=MeanPool --ref ABMIL

`--ref` adds a patient-level bootstrap of (model − reference) with a 95% CI; the resample unit is
the PATIENT, because ROIs of one case are not independent.
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


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3]))


def mrec(y, p):
    r = [(p[y == g] == g).mean() * 100 for g in range(4) if (y == g).any()]
    return float(np.mean(r)) if r else float("nan")


def ds_of(bb, root):
    if (bb, root) not in _DS:
        _DS[(bb, root)] = GradingBagDatasetFull(Path(root) / BACKBONE_CONFIG[bb]["feature_dir"])
    return _DS[(bb, root)]


def gather(prefix):
    out = defaultdict(lambda: ([], [], []))
    folds = defaultdict(set)
    for cf in sorted(glob.glob(str(ROOT / "experiments/*/*/config.json"))):
        c = json.load(open(cf))
        if not str(c.get("prefix", "")).startswith(prefix) or not c.get("n_folds"):
            continue
        d = Path(cf).parent
        if not (d / "test_predictions.csv").exists():
            continue
        ds = ds_of(c["backbone"], c.get("data_root", "data"))
        _, _, te = patient_kfold_split(ds, c["n_folds"], c["fold"], c.get("fold_seed", 2))
        df = pd.read_csv(d / "test_predictions.csv")
        if len(df) != len(te) or list(df.label_idx) != [ds.samples[i][1] for i in te]:
            continue
        if c["fold"] in folds[c["backbone"]]:
            continue                                   # keep one run per fold
        folds[c["backbone"]].add(c["fold"])
        P, Y, R = out[c["backbone"]]
        P.extend(ds.get_slide_path(i).parent.name for i in te)
        Y.extend(df.label_idx.tolist())
        R.extend(df.raw_output.tolist())
    return ({k: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for k, v in out.items()},
            {k: len(v) for k, v in folds.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="prefix=Label, e.g. cvab=ABMIL")
    ap.add_argument("--ref", default=None, help="label used as the bootstrap reference")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    models = {}
    nfold = {}
    for s in a.specs:
        pre, _, lbl = s.partition("=")
        models[lbl or pre], nfold[lbl or pre] = gather(pre)

    print("\nOUT-OF-FOLD over the whole 50-patient cohort (each patient predicted exactly once)\n")
    h = (f"{'backbone':9s} {'model':9s} {'folds':>5s} {'ROIs':>5s} | {'ROI QWK':>8s} {'ROI mRec':>9s}"
         f" | {'PAT QWK':>8s} {'PAT acc':>9s} {'PAT MAE':>8s}")
    print(h); print("-" * len(h))
    for bb in ("virchow2", "uni2", "titan"):
        for lbl, (G, _) in ((l, (m, None)) for l, m in models.items()):
            if bb not in G:
                continue
            P, Y, R = G[bb]
            roi = np.clip(np.round(R), 0, 3).astype(int)
            pl = sorted(set(P))
            pm = np.array([R[P == p].mean() for p in pl])
            py = np.array([Y[P == p][0] for p in pl])
            pp = np.clip(np.round(pm), 0, 3).astype(int)
            print(f"{bb:9s} {lbl:9s} {nfold[lbl].get(bb, 0):5d} {len(Y):5d} | "
                  f"{qwk(Y, roi):8.4f} {mrec(Y, roi):9.1f} | {qwk(py, pp):8.4f} "
                  f"{f'{int((pp == py).sum())}/{len(pl)}':>9s} {np.abs(pm - py).mean():8.3f}")
        print()

    if not a.ref or a.ref not in models:
        return
    rng = np.random.default_rng(0)
    print(f"Patient-level bootstrap vs {a.ref} ({a.boot:,} resamples of the 50 patients)\n")
    h2 = f"{'backbone':9s} {'model':9s} {'metric':9s} {'Δ':>9s} {'95% CI':>22s} {'P(>ref)':>8s}"
    print(h2); print("-" * len(h2))
    Gr, _ = models[a.ref], None
    for bb in ("virchow2", "uni2", "titan"):
        if bb not in models[a.ref]:
            continue
        Pr, Yr, Rr = models[a.ref][bb]
        plr = sorted(set(Pr))
        ir = {p: np.where(Pr == p)[0] for p in plr}
        for lbl, G in models.items():
            if lbl == a.ref or bb not in G:
                continue
            Pm, Ym, Rm = G[bb]
            im = {p: np.where(Pm == p)[0] for p in plr}
            ty = np.array([Yr[ir[p]][0] for p in plr])
            pa = np.array([np.clip(round(Rm[im[p]].mean()), 0, 3) for p in plr])
            pb = np.array([np.clip(round(Rr[ir[p]].mean()), 0, 3) for p in plr])
            res = {"ROI QWK": [], "ROI mRec": [], "PAT acc": []}
            for _ in range(a.boot):
                j = rng.integers(0, len(plr), len(plr))
                sa = np.concatenate([im[plr[t]] for t in j])
                sb = np.concatenate([ir[plr[t]] for t in j])
                ra = np.clip(np.round(Rm[sa]), 0, 3).astype(int)
                rb = np.clip(np.round(Rr[sb]), 0, 3).astype(int)
                res["ROI QWK"].append(qwk(Ym[sa], ra) - qwk(Yr[sb], rb))
                va, vb = mrec(Ym[sa], ra), mrec(Yr[sb], rb)
                if np.isfinite(va) and np.isfinite(vb):
                    res["ROI mRec"].append(va - vb)
                res["PAT acc"].append((pa[j] == ty[j]).mean() * 100 - (pb[j] == ty[j]).mean() * 100)
            obs = {
                "ROI QWK": qwk(Ym, np.clip(np.round(Rm), 0, 3).astype(int))
                           - qwk(Yr, np.clip(np.round(Rr), 0, 3).astype(int)),
                "ROI mRec": mrec(Ym, np.clip(np.round(Rm), 0, 3).astype(int))
                            - mrec(Yr, np.clip(np.round(Rr), 0, 3).astype(int)),
                "PAT acc": (pa == ty).mean() * 100 - (pb == ty).mean() * 100,
            }
            for k, v in res.items():
                v = np.array(v)
                lo, hi = np.percentile(v, [2.5, 97.5])
                star = " *" if (lo > 0 or hi < 0) else ""
                print(f"{bb:9s} {lbl:9s} {k:9s} {obs[k]:+9.4f} "
                      f"{f'[{lo:+.4f}, {hi:+.4f}]':>22s} {100 * (v > 0).mean():7.1f}%{star}")
        print()
    print("  * = 95% CI excludes zero")


if __name__ == "__main__":
    main()
