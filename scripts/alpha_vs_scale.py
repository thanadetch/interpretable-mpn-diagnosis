#!/usr/bin/env python3
"""Does the best entmax order depend on the ROI's PHYSICAL scale?

The idea under test: a 224 px patch covers 61 um of tissue at 0.27 um/px but 605 um at 2.70, so a
patch means something different at each magnification. At high magnification one patch is a small
local sample and many are needed to estimate a diffuse fibre density, which argues for DENSE
pooling (low alpha); at low magnification a single patch already summarises most of the field,
which argues for SPARSER pooling (high alpha). If that is right, the alpha minimising the error
should shift monotonically with um/px, and conditioning alpha on the measured scale would have
something to exploit.

This is a free pre-test: it reuses the trained ASGAP checkpoints and only swaps the pooling
operator at inference (same procedure as `acr_alpha_sweep.py`), so no training is needed. If the
best alpha is flat across scale groups, the idea is dead before any module is written.

KILL CONDITION, stated before looking: the best alpha must move in a consistent direction across
scale bins on at least 2 of 3 backbones, by more than one alpha step. Anything else = no lever.

    python scripts/alpha_vs_scale.py
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

BINS = [(0.20, 0.45, "0.20-0.45 (~25-40x)"), (0.45, 0.70, "0.45-0.70 (~20x)"),
        (0.70, 1.20, "0.70-1.20 (~10-12x)"), (1.20, 3.00, "1.20-3.00 (~4-8x)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cvas")
    a = ap.parse_args()
    dev = torch.device("cpu")

    sc = pd.read_csv(ROOT / "results/scale_um_per_px.csv")
    sc = sc[sc.um_per_px.notna() & sc.um_per_px.between(0.2, 3.0)]
    scale = {(r.patient, r.stem): r.um_per_px for r in sc.itertuples()}

    # {(backbone, alpha): [(um_per_px, y, yhat)]}
    out = defaultdict(list)
    seen = set()
    for cfg, ckpt in runs_for(a.prefix):
        key = (cfg["backbone"], cfg["fold"])
        if key in seen:
            continue
        seen.add(key)
        bb = cfg["backbone"]
        ds = ds_of(bb, cfg.get("data_root", "data"))
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        model = build(cfg, BACKBONE_CONFIG[bb]["dim"]).to(dev)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        with torch.no_grad():
            for i in state["test_idx"]:
                p = ds.get_slide_path(i)
                um = scale.get((p.parent.name, p.stem))
                if um is None:
                    continue
                x, y = ds[i][0].squeeze(0).to(dev), ds.samples[i][1]
                h, e = logits_of(model, x)
                for al in ALPHAS:
                    yhat = float(model.classifier(torch.mv(h.t(), pool(e, al))).view(-1)[0])
                    out[(bb, al)].append((um, y, yhat))
        print(f"  done {bb} fold {cfg['fold']}", file=sys.stderr)

    print("\nBEST ALPHA PER PHYSICAL-SCALE BIN — trained ASGAP scorer, pooling swept at inference\n"
          "cells are mean |predicted - true| (lower is better); ** marks the best alpha in the row\n")
    hdr = (f"{'backbone':9s} {'um/px bin':22s} {'n':>5s} "
           + "".join(f"{'a=' + f'{al:.2f}':>10s}" for al in ALPHAS) + f"  {'best':>6s}")
    print(hdr); print("-" * len(hdr))
    for bb in ("virchow2", "uni2", "titan"):
        if (bb, ALPHAS[0]) not in out:
            continue
        for lo, hi, name in BINS:
            errs, n = [], 0
            for al in ALPHAS:
                arr = np.array(out[(bb, al)])
                m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
                n = int(m.sum())
                errs.append(np.abs(arr[m, 2] - arr[m, 1]).mean() if n else np.nan)
            if n < 30:
                continue
            k = int(np.nanargmin(errs))
            cells = "".join((f"{e:9.3f}**" if j == k else f"{e:10.3f}") for j, e in enumerate(errs))
            print(f"{bb:9s} {name:22s} {n:5d} {cells}  {ALPHAS[k]:6.2f}")
        print()
    print("KILL CONDITION: the best alpha must shift consistently with um/px on >=2 of 3 backbones,\n"
          "by more than one alpha step. A flat 'best' column means conditioning alpha on the\n"
          "measured scale has nothing to exploit.")


if __name__ == "__main__":
    main()
