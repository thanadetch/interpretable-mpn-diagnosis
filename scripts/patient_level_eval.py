#!/usr/bin/env python3
"""Re-score finished runs at the PATIENT level, the unit the clinical decision is made on.

Why this exists
---------------
Every run in this project is ranked by ROI-level QWK, but ROI labels are constant within a
patient. Any mechanism that makes a patient's ROIs agree with each other therefore raises
ROI-level QWK **without changing a single patient-level decision**. Measured on 2026-08-08:
`a352_patient_query` reached the highest test ROI-QWK in the study (.9706 virchow2, .9733 titan)
while scoring exactly the same 9/10 test patients as the ASGAP baseline, and while *dropping*
patient-level accuracy on val. An ROI-only gain is a consistency claim, not an accuracy claim.

No checkpoint is needed: the saved `{split}_predictions.csv` rows are written in dataset order
for that split, so the patient of each row is recoverable from `patient_split(seed)`. This is
verified per run — if the recomputed ROI-QWK does not match the saved metrics the run is flagged
STALE (it was produced against a different feature set or split) and skipped.

Usage
-----
    python scripts/patient_level_eval.py experiments/20260808/fc*          # explicit dirs
    python scripts/patient_level_eval.py --all --min_test_qwk 0.94         # sweep everything
    python scripts/patient_level_eval.py experiments/20260808/fc* --patients   # per-patient dump
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from train_grading_reti import BACKBONE_CONFIG, patient_split  # noqa: E402

_CACHE: dict = {}
AGG = "mean"   # case-level readout: mean | max | q90 (set by --agg)


def qwk(a, b) -> float:
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def split_info(backbone: str, seed: int, data_root: str):
    """(dataset, {split: [patient id per row]}) — cached, since building it walks 1330 files."""
    key = (backbone, seed, data_root)
    if key in _CACHE:
        return _CACHE[key]
    feat_dir = Path(data_root) / BACKBONE_CONFIG[backbone]["feature_dir"]
    ds = GradingBagDatasetFull(feat_dir)
    tr, va, te = patient_split(ds, seed=seed)
    pat = {
        "val": [ds.get_slide_path(i).parent.name for i in va],
        "test": [ds.get_slide_path(i).parent.name for i in te],
    }
    lab = {
        "val": [ds.samples[i][1] for i in va],
        "test": [ds.samples[i][1] for i in te],
    }
    _CACHE[key] = (pat, lab)
    return _CACHE[key]


def score(run: Path, want_patients: bool = False):
    cfg_path = run / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.load(open(cfg_path))
    bb = cfg.get("backbone")
    if bb not in BACKBONE_CONFIG:
        return None
    try:
        pat, lab = split_info(bb, int(cfg.get("seed", 2)), cfg.get("data_root", "data"))
    except Exception:
        return None

    out = {
        "run": run.name,
        "backbone": bb,
        "model": cfg.get("novelty_id") or cfg.get("model_type"),
        "aug": cfg.get("augmentation") or "-",
    }
    per_patient = {}
    for split in ("val", "test"):
        csv = run / f"{split}_predictions.csv"
        mj = run / f"{split}_metrics.json"
        if not csv.exists() or not mj.exists():
            return None
        df = pd.read_csv(csv)
        P, Y = np.array(pat[split]), np.array(lab[split])
        if len(df) != len(P) or list(df.label_idx) != list(Y):
            out[f"{split}_stale"] = True
            continue

        roi_pred = df.pred_idx.to_numpy()
        saved = json.load(open(mj))[f"{split}_qwk"]
        recomputed = qwk(Y, roi_pred)
        if abs(recomputed - saved) > 1e-4:
            out[f"{split}_stale"] = True
            continue

        raw = df.raw_output.to_numpy() if "raw_output" in df.columns else roi_pred.astype(float)
        pl = sorted(set(P))
        # How a case's fields are combined into one grade. "mean" is the default; "max" and
        # "q90" match the WHO/EUMNET rule of grading from the most representative (worst) area,
        # and are the correct readout for a model trained with the M1 asymmetric loss.
        agg = {"mean": np.mean, "max": np.max,
               "q90": lambda x: np.percentile(x, 90)}[AGG]
        pm = np.array([agg(raw[P == p]) for p in pl])
        py = np.array([Y[P == p][0] for p in pl])
        pp = np.clip(np.round(pm), 0, 3).astype(int)

        out[f"{split}_roi_qwk"] = recomputed
        out[f"{split}_pat_qwk"] = qwk(py, pp)
        out[f"{split}_pat_ok"] = f"{int((pp == py).sum())}/{len(pl)}"
        out[f"{split}_pat_mae"] = float(np.abs(pm - py).mean())
        out[f"{split}_within_sd"] = float(np.mean([raw[P == p].std() for p in pl]))
        if want_patients:
            per_patient[split] = [
                (p, int(t), round(float(m), 3), int(q), "ok" if q == t else "WRONG")
                for p, t, m, q in zip(pl, py, pm, pp)
            ]
    out["_patients"] = per_patient
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run directories (globs already expanded by the shell)")
    ap.add_argument("--all", action="store_true", help="scan every run under experiments/")
    ap.add_argument("--min_test_qwk", type=float, default=None, help="only show runs above this ROI test QWK")
    ap.add_argument("--patients", action="store_true", help="dump the per-patient table for each run")
    ap.add_argument("--agg", default="mean", choices=["mean", "max", "q90"],
                    help="how a case's ROI scores are combined into one grade (default mean)")
    a = ap.parse_args()
    global AGG
    AGG = a.agg

    dirs = [Path(d) for d in a.runs]
    if a.all:
        dirs += [Path(p).parent for p in glob.glob(str(ROOT / "experiments/*/*/config.json"))]
    dirs = sorted({d for d in dirs if d.is_dir()})
    if not dirs:
        print("no run directories given (pass paths, or --all)")
        return

    rows, stale = [], 0
    for d in dirs:
        r = score(d, want_patients=a.patients)
        if r is None:
            continue
        if r.get("test_stale") or r.get("val_stale"):
            stale += 1
            continue
        if a.min_test_qwk is not None and r.get("test_roi_qwk", 0) < a.min_test_qwk:
            continue
        rows.append(r)

    if not rows:
        print(f"nothing scoreable ({stale} runs skipped as stale)")
        return
    rows.sort(key=lambda r: -r.get("test_roi_qwk", 0))

    print(f"{'backbone':9s} {'model':28s} {'aug':18s} | "
          f"{'valROI':>7s} {'valPAT':>7s} {'ok':>6s} | {'testROI':>7s} {'testPAT':>7s} {'ok':>6s} {'wSD':>6s}")
    print("-" * 118)
    for r in rows:
        print(f"{r['backbone']:9s} {str(r['model'])[:28]:28s} {str(r['aug'])[:18]:18s} | "
              f"{r.get('val_roi_qwk', float('nan')):7.4f} {r.get('val_pat_qwk', float('nan')):7.4f} "
              f"{r.get('val_pat_ok', '-'):>6s} | "
              f"{r.get('test_roi_qwk', float('nan')):7.4f} {r.get('test_pat_qwk', float('nan')):7.4f} "
              f"{r.get('test_pat_ok', '-'):>6s} {r.get('test_within_sd', float('nan')):6.3f}")
        if a.patients:
            for split, tab in r["_patients"].items():
                bad = [t for t in tab if t[4] == "WRONG"]
                print(f"    {split}: wrong = " + (", ".join(f"{p}(true {t}, pred {m})" for p, t, m, _, _ in bad) or "none"))
    print(f"\n{len(rows)} runs scored, {stale} skipped as stale (different split/features).")


if __name__ == "__main__":
    main()
