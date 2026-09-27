#!/usr/bin/env python3
"""How concentrated is the attention, really?

Counting non-zero weights is the wrong ruler. alpha-entmax at 1.5 zeroes only the patches whose
logit falls below tau, which on this data is a handful -- yet the heat maps look obviously
different from ABMIL's. The difference lives in how the mass is DISTRIBUTED among the survivors,
not in how many survive, so this script reports the rulers that measure that:

    support        # of weights > 1e-8                      (what the previous script measured)
    eff N (IPR)    1 / sum a_i^2   -- inverse participation ratio; the number of patches the bag
                   vector effectively averages over. 1 = one patch decides, N = uniform.
    eff N (exp H)  exp(-sum a_i log a_i) -- the same idea read off the entropy
    top5 mass      fraction of the total weight held by the five largest patches
    gini           0 = uniform, 1 = one patch takes everything

Run on the CV checkpoints; no retraining.

    python scripts/attention_concentration.py cvas=ASGAP cvab=ABMIL cvtm=TransMIL
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from cardinality_invariance import build, ds_of, runs_for  # noqa: E402

from train_grading_reti import BACKBONE_CONFIG, patient_kfold_split  # noqa: E402


def stats(a: np.ndarray) -> dict:
    a = a / max(a.sum(), 1e-12)
    nz = a[a > 1e-8]
    h = -(nz * np.log(nz)).sum()
    s = np.sort(a)[::-1]
    n = len(a)
    idx = np.arange(1, n + 1)
    return dict(
        N=n,
        support=float((a > 1e-8).sum()),
        ipr=float(1.0 / (a ** 2).sum()),
        expH=float(np.exp(h)),
        top5=float(s[:5].sum()),
        gini=float((2 * (idx * np.sort(a)).sum()) / (n * a.sum()) - (n + 1) / n),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    dev = torch.device("cpu")
    acc = defaultdict(list)

    for spec in a.specs:
        pre, _, lbl = spec.partition("=")
        lbl = lbl or pre
        seen = set()
        for cfg, ckpt in runs_for(pre):
            key = (cfg["backbone"], cfg["fold"])
            if key in seen:
                continue
            seen.add(key)
            ds = ds_of(cfg["backbone"], cfg.get("data_root", "data"))
            state = torch.load(ckpt, map_location=dev, weights_only=False)
            te = state.get("test_idx")
            if te is None:
                _, _, te = patient_kfold_split(ds, cfg["n_folds"], cfg["fold"],
                                               cfg.get("fold_seed", 2))
            model = build(cfg, BACKBONE_CONFIG[cfg["backbone"]]["dim"]).to(dev)
            model.load_state_dict(state.get("model_state_dict", state))
            model.eval()
            with torch.no_grad():
                for i in te:
                    x = ds[i][0].to(dev)
                    try:
                        out = model(x, return_attention=True)
                    except TypeError:
                        continue
                    if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
                        continue
                    acc[(lbl, cfg["backbone"])].append(stats(out[1].reshape(-1).abs().numpy()))
        print(f"  done {lbl}", file=sys.stderr)

    labels = [s.partition("=")[2] or s.partition("=")[0] for s in a.specs]
    print("\nATTENTION CONCENTRATION over the out-of-fold ROIs (mean +- sd; bag size N ~ 44)\n")
    h = (f"{'backbone':9s} {'model':9s} {'ROIs':>5s} {'support':>13s} {'eff N (IPR)':>14s}"
         f" {'eff N (expH)':>14s} {'top-5 mass':>13s} {'gini':>13s}")
    print(h); print("-" * len(h))
    for bb in ("virchow2", "uni2", "titan"):
        for lbl in labels:
            v = acc.get((lbl, bb))
            if not v:
                continue
            g = lambda k: np.array([r[k] for r in v])  # noqa: E731
            f = lambda k, p=1: f"{g(k).mean():.{p}f} +-{g(k).std():.{p}f}"  # noqa: E731
            print(f"{bb:9s} {lbl:9s} {len(v):5d} {f('support'):>13s} {f('ipr'):>14s} "
                  f"{f('expH'):>14s} {f('top5', 3):>13s} {f('gini', 3):>13s}")
        print()
    print("  support counts survivors; eff N says how many of them actually carry the decision.\n"
          "  A large gap between the two is exactly what makes two heat maps look different while\n"
          "  their non-zero counts agree.")


if __name__ == "__main__":
    main()
