"""First experiment for the 'ordinal decision-geometry' direction (offline, no training).

For the ABMIL + Virchow2 + regression baseline, the logged raw scalar
(raw_output in *_predictions.csv) is decoded to a grade by the trainer with a
FIXED rule: round(clip(raw,0,3)) == cut-points at 0.5/1.5/2.5.

This script asks: if we instead LEARN monotone cut-points c1<c2<c3 on the VAL
raw scalars (maximising val QWK only) and decode TEST once with them, how much
test QWK lift do we get, and how far do the learned cuts move off 0.5/1.5/2.5?
Repeated across every available seed-fold to test stability (NOT a single fold).

Leakage-safe: cuts are fit on val only; test is decoded exactly once per seed,
matching the locked 'selection on val only' protocol. No training, no trainer
edit, no split change.
"""
from __future__ import annotations
import csv, glob, json, os
from itertools import combinations
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def qwk(y_true, y_pred, N=4):
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    O = np.zeros((N, N), float)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    w = np.zeros((N, N), float)
    for i in range(N):
        for j in range(N):
            w[i, j] = (i - j) ** 2 / (N - 1) ** 2
    act = O.sum(1); pred = O.sum(0)
    E = np.outer(act, pred) / max(O.sum(), 1)
    denom = (w * E).sum()
    return 1.0 - (w * O).sum() / denom if denom > 0 else 0.0


def load(csv_path):
    raw, lab = [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            raw.append(float(r["raw_output"])); lab.append(int(r["label_idx"]))
    return np.array(raw), np.array(lab)


def decode(raw, cuts):
    return np.digitize(raw, cuts)  # cuts=[c1,c2,c3] -> classes 0..3


def default_decode(raw):
    return np.clip(np.round(raw), 0, 3).astype(int)


def best_cuts(val_raw, val_lab, grid):
    best, best_q = None, -2.0
    for c in combinations(grid, 3):
        q = qwk(val_lab, decode(val_raw, list(c)))
        if q > best_q:
            best_q, best = q, c
    return list(best), best_q


def find_runs():
    runs = {}
    for tp in glob.glob(os.path.join(ROOT, "experiments", "*", "*", "test_predictions.csv")):
        d = os.path.dirname(tp)
        if "simple" not in d or "virchow2" not in d or "regression" not in d:
            continue
        vp = os.path.join(d, "val_predictions.csv")
        cfg = os.path.join(d, "config.json")
        if not (os.path.exists(vp) and os.path.exists(cfg)):
            continue
        try:
            s = json.load(open(cfg)).get("seed")
        except Exception:
            continue
        if s is None or s in runs:   # one per seed (first found)
            continue
        runs[s] = (vp, tp)
    return dict(sorted(runs.items()))


def main():
    grid = [round(x, 2) for x in np.arange(0.0, 3.01, 0.1)]
    runs = find_runs()
    print(f"seeds found: {list(runs)}\n")
    hdr = f"{'seed':>4} | {'val_def':>7} {'val_lrn':>7} | {'test_def':>8} {'test_lrn':>8} {'Δtest':>7} | learned cuts"
    print(hdr); print("-" * len(hdr))
    dts = []
    for s, (vp, tp) in runs.items():
        vr, vl = load(vp); tr, tl = load(tp)
        vqd = qwk(vl, default_decode(vr)); tqd = qwk(tl, default_decode(tr))
        cuts, vql = best_cuts(vr, vl, grid)
        tql = qwk(tl, decode(tr, cuts))
        dt = tql - tqd; dts.append(dt)
        print(f"{s:>4} | {vqd:7.3f} {vql:7.3f} | {tqd:8.3f} {tql:8.3f} {dt:+7.3f} | {cuts}")
    dts = np.array(dts)
    print("-" * len(hdr))
    print(f"\nΔtest_qwk (learned − default round()):  mean {dts.mean():+.4f}  median {np.median(dts):+.4f}  "
          f"min {dts.min():+.3f}  max {dts.max():+.3f}  (n={len(dts)} seed-folds)")
    print(f"folds where learned >= default on TEST: {(dts >= -1e-9).sum()}/{len(dts)}")
    print("\nREADING: if Δtest≈0 and cuts≈[0.5,1.5,2.5] -> round() is already optimal -> the lift")
    print("is in the AXIS not the decode (mechanism finding). If cuts move a lot but Δtest is")
    print("unstable across seeds -> decode is a per-fold lottery (variance finding). Either is a clean result.")


if __name__ == "__main__":
    main()
