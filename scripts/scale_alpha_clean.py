#!/usr/bin/env python3
"""Leak-free evaluation of the scale-conditioned entmax order.

The earlier a407 result was contaminated: the alpha for each scale band was read off the CV
out-of-fold TEST predictions and then scored on those same predictions. This script removes that
by fitting the rule inside each fold, on the VALIDATION split only:

    for every CV fold k:
        1. take the ASGAP checkpoint trained on fold k's TRAIN split
        2. on fold k's VAL bags, sweep alpha per scale band and keep the band's best alpha
           (bands with fewer than MIN_N val bags fall back to 1.5, the training value)
        3. apply that fold-specific rule to fold k's TEST bags

Test data never touches the rule, so the pooled out-of-fold numbers are honest. No training is
needed: the pooling operator is swapped at inference on checkpoints that already exist.

Three arms, all on identical bags:
    fixed 1.5        the operator the model was trained with -- the reference
    scale (val-fit)  alpha from the ROI's measured um/px, rule fitted on val
    shuffled ctrl    same rule, but the scales are permuted among the test bags, so the alpha
                     marginal is preserved and only the ROI<->scale correspondence is destroyed

If the scale arm does not beat BOTH the fixed arm and the shuffled arm, the measurement carries
no usable signal.

    python scripts/scale_alpha_clean.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from acr_alpha_sweep import ALPHAS, logits_of, pool  # noqa: E402
from cardinality_invariance import build, ds_of, runs_for  # noqa: E402

from train_grading_reti import BACKBONE_CONFIG  # noqa: E402

EDGES = [0.0, 0.45, 0.70, 3.01]          # um/px band edges; index 3 = "unmeasurable"
UNKNOWN = 3
FALLBACK = 1.5
MIN_N = 15                               # a band needs this many val bags to fit its own alpha


def band_of(um) -> int:
    if um is None or not np.isfinite(um) or not (0.2 <= um <= 3.0):
        return UNKNOWN
    return int(np.searchsorted(EDGES[1:], um, side="right"))


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cvas")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cpu")

    sc = pd.read_csv(ROOT / "results/scale_um_per_px.csv")
    scale = {(r.patient, r.stem): r.um_per_px for r in sc.itertuples()}

    arms = defaultdict(lambda: defaultdict(lambda: ([], [])))   # arm -> bb -> (y, pred)
    chosen = defaultdict(list)
    seen = set()

    for cfg, ckpt in runs_for(a.prefix):
        key = (cfg["backbone"], cfg["fold"])
        if key in seen:
            continue
        seen.add(key)
        bb = cfg["backbone"]
        ds = ds_of(bb, cfg.get("data_root", "data"))
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model = build(cfg, BACKBONE_CONFIG[bb]["dim"]).to(dev)
        model.load_state_dict(st["model_state_dict"])
        model.eval()

        def scan(idx):
            """[(band, y, {alpha: pred})] for the given dataset indices."""
            rows = []
            with torch.no_grad():
                for i in idx:
                    p = ds.get_slide_path(i)
                    x, y = ds[i][0].squeeze(0).to(dev), ds.samples[i][1]
                    h, e = logits_of(model, x)
                    preds = {al: float(model.classifier(torch.mv(h.t(), pool(e, al))).view(-1)[0])
                             for al in ALPHAS}
                    rows.append((band_of(scale.get((p.parent.name, p.stem))), y, preds))
            return rows

        # ---- fit the rule on VAL only -------------------------------------------------------
        val = scan(st["val_idx"])
        rule = {}
        for b in range(4):
            sub = [r for r in val if r[0] == b]
            if len(sub) < MIN_N:
                rule[b] = FALLBACK
                continue
            err = {al: np.mean([abs(r[2][al] - r[1]) for r in sub]) for al in ALPHAS}
            rule[b] = min(err, key=err.get)
        chosen[bb].append(tuple(rule[b] for b in range(4)))

        # ---- apply to TEST ------------------------------------------------------------------
        test = scan(st["test_idx"])
        rng = np.random.default_rng(a.seed + cfg["fold"])
        shuf = rng.permutation([r[0] for r in test])
        for j, (b, y, preds) in enumerate(test):
            arms["fixed 1.5"][bb][0].append(y)
            arms["fixed 1.5"][bb][1].append(preds[FALLBACK])
            arms["scale (val-fit)"][bb][0].append(y)
            arms["scale (val-fit)"][bb][1].append(preds[rule[b]])
            arms["shuffled ctrl"][bb][0].append(y)
            arms["shuffled ctrl"][bb][1].append(preds[rule[int(shuf[j])]])
        print(f"  {bb} fold {cfg['fold']}: rule={rule}", file=sys.stderr)

    print("\nLEAK-FREE: alpha per scale band fitted on each fold's VAL, applied to its TEST\n")
    h = (f"{'backbone':9s} {'arm':18s} {'ROI ถูก':>11s} {'acc':>7s} {'QWK':>8s} {'MAE':>6s}")
    print(h); print("-" * len(h))
    for bb in ("virchow2", "uni2", "titan"):
        for arm in ("fixed 1.5", "scale (val-fit)", "shuffled ctrl"):
            if bb not in arms[arm]:
                continue
            y = np.array(arms[arm][bb][0]); r = np.array(arms[arm][bb][1])
            p = np.clip(np.round(r), 0, 3).astype(int)
            print(f"{bb:9s} {arm:18s} {f'{(p == y).sum()}/{len(y)}':>11s} "
                  f"{(p == y).mean() * 100:6.1f}% {qwk(y, p):8.4f} {np.abs(r - y).mean():6.3f}")
        print()
    print("alpha ที่ val เลือกในแต่ละ fold  (band: <0.45 | 0.45-0.70 | >=0.70 | unknown)")
    for bb, v in chosen.items():
        print(f"  {bb:9s} " + "  ".join(str(t) for t in v))


if __name__ == "__main__":
    main()
