#!/usr/bin/env python3
"""Is the grading model reading FIBROSIS, or is it reading MAGNIFICATION?

The reticulin ROIs were acquired by hand, so their scale bars differ (20/50/100/200 um) and the
magnification is itself correlated with grade -- a pathologist zooms in further on fibrotic
marrow. Predicting the grade from the scale bar ALONE already scores QWK 0.29, so magnification
is an available shortcut and nothing in this repo has ever checked whether the models take it.

Three questions, in increasing sharpness:

  A  overall out-of-fold ROI QWK                       the reference number
  B  QWK computed WITHIN one magnification             a drop is suspicious, but QWK also falls
                                                       when a stratum holds fewer grades, so B
                                                       alone cannot convict
  C  within each TRUE grade, does the predicted score   the decisive test: true grade is held
     move with magnification?                           fixed, so any remaining dependence on
                                                        magnification is pure shortcut

Reads the out-of-fold predictions the CV sweep already wrote; no training, no inference.

    python scripts/scale_audit.py cvas=ASGAP cvab=ABMIL cvmp=MeanPool cvtm=TransMIL
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

from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from train_grading_reti import BACKBONE_CONFIG, patient_kfold_split  # noqa: E402

# scale bar in um -> tissue width covered by one 224 px patch is proportional to it, so the
# ordering below is "how much tissue one patch sees", low = zoomed in.
SCALE_ORDER = ["20", "50", "100", "200"]
_DS: dict = {}


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3]))


def ds_of(bb, root):
    if (bb, root) not in _DS:
        _DS[(bb, root)] = GradingBagDatasetFull(Path(root) / BACKBONE_CONFIG[bb]["feature_dir"])
    return _DS[(bb, root)]


def load_scalebar() -> dict:
    df = pd.read_csv(ROOT / "results/scalebar_results.csv")
    df["stem"] = df.filename.str.replace(r"\.tif$", "", regex=True)
    return {(r.patient, r.stem): str(r.scalebar_micron) for r in df.itertuples()}


def gather(prefix: str):
    """{backbone: DataFrame(patient, stem, y, raw)} pooled out-of-fold, one run per fold."""
    rows = defaultdict(list)
    seen = defaultdict(set)
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
        if c["fold"] in seen[c["backbone"]]:
            continue
        seen[c["backbone"]].add(c["fold"])
        for i, (_, r) in zip(te, df.iterrows()):
            p = ds.get_slide_path(i)
            rows[c["backbone"]].append((p.parent.name, p.stem, int(r.label_idx), float(r.raw_output)))
    return {k: pd.DataFrame(v, columns=["patient", "stem", "y", "raw"]) for k, v in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="prefix=Label")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    sb = load_scalebar()
    models = {}
    for s in a.specs:
        pre, _, lbl = s.partition("=")
        models[lbl or pre] = gather(pre)

    # ---- reference: what magnification alone can do, on the same ROIs -------------------------
    ref = next(iter(models.values()))
    ref = next(iter(ref.values()))
    ref = ref.assign(scale=[sb.get((p, s), "unknown") for p, s in zip(ref.patient, ref.stem)])
    known = ref[ref.scale != "unknown"]
    maj = known.groupby("scale").y.agg(lambda x: x.value_counts().idxmax())
    print(f"\nMagnification-only predictor on the {len(known)} ROIs with a readable scale bar: "
          f"QWK {qwk(known.y, known.scale.map(maj)):.4f}\n")

    print("A/B  out-of-fold ROI QWK overall, and WITHIN one magnification\n")
    h = (f"{'backbone':9s} {'model':9s} {'overall':>8s} |" +
         "".join(f"{s + 'um':>12s}" for s in SCALE_ORDER) + f"{'unknown':>12s}")
    print(h); print("-" * len(h))
    for bb in ("virchow2", "uni2", "titan"):
        for lbl, G in models.items():
            if bb not in G:
                continue
            d = G[bb].assign(scale=[sb.get((p, s), "unknown")
                                    for p, s in zip(G[bb].patient, G[bb].stem)])
            d["pred"] = np.clip(np.round(d.raw), 0, 3).astype(int)
            cells = []
            for s in SCALE_ORDER + ["unknown"]:
                q = d[d.scale == s]
                cells.append("     -      " if len(q) < 20 or q.y.nunique() < 2
                             else f"{qwk(q.y, q.pred):7.4f}({q.y.nunique()}g)")
            print(f"{bb:9s} {lbl:9s} {qwk(d.y, d.pred):8.4f} |" + "".join(cells))
        print()
    print("  (Ng) = number of distinct true grades present in that stratum; QWK is not comparable\n"
          "  across strata that hold different numbers of grades -- that is what C is for.\n")

    # ---- C: the decisive test ----------------------------------------------------------------
    print("C  WITHIN each true grade, does the predicted score move with magnification?\n"
         f"   rho = Spearman(predicted score, tissue-per-patch), pooled over grades after\n"
         f"   centring each grade; CI from {a.boot:,} bootstrap resamples of PATIENTS.\n")
    rng = np.random.default_rng(0)
    h2 = f"{'backbone':9s} {'model':9s} {'n':>5s} {'rho':>8s} {'95% CI':>20s}   {'mean pred by 20/50/100/200um'}"
    print(h2); print("-" * len(h2))
    for bb in ("virchow2", "uni2", "titan"):
        for lbl, G in models.items():
            if bb not in G:
                continue
            d = G[bb].assign(scale=[sb.get((p, s), "unknown")
                                    for p, s in zip(G[bb].patient, G[bb].stem)])
            d = d[d.scale.isin(SCALE_ORDER)].copy()
            d["mag"] = d.scale.astype(float)                 # um per scale bar ~ tissue per patch
            d["resid"] = d.raw - d.groupby("y").raw.transform("mean")   # hold TRUE grade fixed
            obs = spearmanr(d.resid, d.mag).statistic
            pl = sorted(d.patient.unique())
            idx = {p: np.where(d.patient.values == p)[0] for p in pl}
            bs = []
            for _ in range(a.boot):
                j = rng.integers(0, len(pl), len(pl))
                sel = np.concatenate([idx[pl[t]] for t in j])
                if d.mag.values[sel].std() > 0:
                    bs.append(spearmanr(d.resid.values[sel], d.mag.values[sel]).statistic)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = " *" if (lo > 0 or hi < 0) else ""
            means = "  ".join(f"{d[d.scale == s].raw.mean():.2f}" if (d.scale == s).sum() >= 20
                              else "  - " for s in SCALE_ORDER)
            print(f"{bb:9s} {lbl:9s} {len(d):5d} {obs:+8.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20s}   {means}{star}")
        print()
    print("  * = 95% CI excludes zero, i.e. the prediction depends on magnification even after the\n"
          "  true grade is held fixed -> the model is partly reading acquisition, not tissue.")


if __name__ == "__main__":
    main()
