#!/usr/bin/env python3
"""Does the predicted grade depend on how much tissue the pathologist happened to crop?

The ROIs here hold 13 to 112 patches -- an 8.6x range that reflects the operator, not the
disease. A grading model should be invariant to it. Whether it is follows from the pooling
operator alone:

    softmax (ABMIL, TransMIL)   every patch keeps a strictly positive weight, so the normaliser
                                grows with N and the weight on any fixed informative patch decays
                                as O(1/N). Appending uninformative patches ALWAYS moves the bag
                                vector.
    alpha-entmax (ASGAP)        patches whose logit falls below the threshold tau receive exactly
                                zero. Appending such patches leaves the pooled vector EXACTLY
                                unchanged -- an invariance, not a tendency.
    mean pooling                the worst case: every appended patch gets weight 1/N by fiat.

This script measures that on the trained CV checkpoints; no retraining.

  DILUTE   append k patches sampled from OTHER patients' ROIs (what a wider crop would add) and
           record how far the prediction moves, and what happens to out-of-fold QWK.
  SUBSET   keep a random 75 / 50 / 25 % of each bag -- the opposite perturbation.
  SUPPORT  for the entmax models, the number of patches with non-zero attention as a function of
           N. The invariance claim requires this to stay flat; if the support grows with N the
           theorem does not bind and the claim must be dropped.

    python scripts/cardinality_invariance.py cvas=ASGAP cvab=ABMIL cvmp=MeanPool cvtm=TransMIL
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from models.simple_mil import SimpleGatedMIL  # noqa: E402
from models.mean_pool_mil import MeanPoolMIL  # noqa: E402
from train_grading_reti import BACKBONE_CONFIG, patient_kfold_split  # noqa: E402

DILUTE = [0.0, 0.25, 0.5, 1.0, 2.0]      # k = frac * N appended distractor patches
KEEP = [1.0, 0.75, 0.5, 0.25]
_DS: dict = {}


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3]))


def ds_of(bb, root):
    if (bb, root) not in _DS:
        _DS[(bb, root)] = GradingBagDatasetFull(Path(root) / BACKBONE_CONFIG[bb]["feature_dir"])
    return _DS[(bb, root)]


def build(cfg, input_dim):
    mt = cfg["model_type"]
    if mt == "simple":
        return SimpleGatedMIL(input_dim=input_dim, num_classes=1, topk=cfg.get("topk", 0))
    if mt == "mean_pool":
        return MeanPoolMIL(vision_dim=input_dim, num_classes=1)
    if mt == "novelty_attempt":
        mod = importlib.import_module(f"models.novelty_attempts.{cfg['novelty_id']}")
        kw = dict(getattr(mod, "KWARGS", {}))
        kw["input_dim"], kw["num_classes"] = input_dim, 1
        return mod.Model(**kw)
    raise ValueError(mt)


@torch.no_grad()
def predict(model, x):
    out = model(x)                       # the trainer feeds [N, D]; no batch dimension
    y = out[0] if isinstance(out, tuple) else out
    return float(y.reshape(-1)[0])


@torch.no_grad()
def support_size(model, x):
    """Number of patches with non-zero attention, or nan if the model exposes none."""
    try:
        out = model(x, return_attention=True)
    except TypeError:                     # mean pooling has no attention to report
        return float("nan")
    if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
        return float("nan")
    a = out[1].reshape(-1).abs()
    return float((a > 1e-8).sum())


def runs_for(prefix):
    for cf in sorted(glob.glob(str(ROOT / "experiments/*/*/config.json"))):
        c = json.load(open(cf))
        if str(c.get("prefix", "")).startswith(prefix) and c.get("n_folds"):
            d = Path(cf).parent
            ck = list(d.glob("best_*.pth"))
            if ck:
                yield c, ck[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cpu")

    dilute = defaultdict(lambda: defaultdict(lambda: ([], [], [])))   # (drift, y, pred)
    subset = defaultdict(lambda: defaultdict(lambda: ([], [], [])))
    supp = defaultdict(list)

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
            # the checkpoint stores the exact indices it was trained on - prefer them over
            # recomputing the split, so a mismatch can never go unnoticed
            tr = state.get("train_idx")
            te = state.get("test_idx")
            if tr is None or te is None:
                tr, _, te = patient_kfold_split(ds, cfg["n_folds"], cfg["fold"],
                                                cfg.get("fold_seed", 2))
            model = build(cfg, BACKBONE_CONFIG[cfg["backbone"]]["dim"]).to(dev)
            model.load_state_dict(state.get("model_state_dict", state))
            model.eval()

            rng = np.random.default_rng(a.seed + cfg["fold"])
            pool = [ds[i][0].squeeze(0) for i in rng.choice(tr, size=min(40, len(tr)), replace=False)]
            pool = torch.cat(pool, 0)                     # distractor patches from OTHER patients

            for i in te:
                x, y = ds[i][0].squeeze(0).to(dev), ds.samples[i][1]
                base = predict(model, x)
                s = support_size(model, x)
                if np.isfinite(s):
                    supp[(lbl, cfg["backbone"])].append((len(x), s))
                for f in DILUTE:
                    k = int(round(f * len(x)))
                    xx = x if k == 0 else torch.cat(
                        [x, pool[rng.choice(len(pool), size=k, replace=False)]], 0)
                    p = predict(model, xx)
                    D, Y, P = dilute[(lbl, cfg["backbone"])][f]
                    D.append(abs(p - base)); Y.append(y); P.append(p)
                for f in KEEP:
                    k = max(1, int(round(f * len(x))))
                    xx = x if k >= len(x) else x[rng.choice(len(x), size=k, replace=False)]
                    p = predict(model, xx)
                    D, Y, P = subset[(lbl, cfg["backbone"])][f]
                    D.append(abs(p - base)); Y.append(y); P.append(p)
        print(f"  done {lbl}", file=sys.stderr)

    labels = [s.partition("=")[2] or s.partition("=")[0] for s in a.specs]

    def block(title, store, grid, colfmt):
        print(f"\n{title}\n")
        h = f"{'backbone':9s} {'model':9s} " + "".join(f"{colfmt(f):>16s}" for f in grid)
        print(h); print("-" * len(h))
        for bb in ("virchow2", "uni2", "titan"):
            for lbl in labels:
                if (lbl, bb) not in store:
                    continue
                cells = []
                for f in grid:
                    D, Y, P = store[(lbl, bb)][f]
                    q = qwk(np.array(Y), np.clip(np.round(P), 0, 3).astype(int))
                    cells.append(f"{q:.4f}/{np.mean(D):.3f}".rjust(16))
                print(f"{bb:9s} {lbl:9s} " + "".join(cells))
            print()

    block("DILUTE — append k = f x N patches from OTHER patients.   QWK / mean |drift|",
          dilute, DILUTE, lambda f: f"+{int(f * 100)}%")
    block("SUBSET — keep a random fraction of each bag.             QWK / mean |drift|",
          subset, KEEP, lambda f: f"keep {int(f * 100)}%")

    print("\nSUPPORT — patches with non-zero attention vs bag size N "
          "(the invariance claim needs this flat)\n")
    h = f"{'backbone':9s} {'model':9s} {'N range':>12s} {'support':>16s} {'slope dS/dN':>12s} {'corr':>7s}"
    print(h); print("-" * len(h))
    for bb in ("virchow2", "uni2", "titan"):
        for lbl in labels:
            v = supp.get((lbl, bb))
            if not v:
                continue
            n, s = np.array(v).T
            sl = np.polyfit(n, s, 1)[0]
            print(f"{bb:9s} {lbl:9s} {f'{int(n.min())}-{int(n.max())}':>12s} "
                  f"{f'{s.mean():.1f} +- {s.std():.1f}':>16s} {sl:12.3f} "
                  f"{np.corrcoef(n, s)[0, 1]:7.3f}")
        print()
    print("  slope 0 = the pooled vector ignores extra patches entirely (dilution-free);\n"
          "  slope 1 = every added patch enters the pool, as softmax and mean pooling must.")


if __name__ == "__main__":
    main()
