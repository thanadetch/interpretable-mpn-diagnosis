#!/usr/bin/env python3
"""The ACR (Attention Concentration-Robustness) profile as a function of the entmax order alpha.

Earlier scripts measured three scattered points (MeanPool, ABMIL, ASGAP) and found the trade-off:
more concentrated attention = better heat maps, worse robustness to foreign patches, same
accuracy. This script turns those points into ONE curve with a knob, using the trained ASGAP
checkpoints and INFERENCE ONLY: the bag's attention logits e come from the trained scorer, and
the pooling operator applied to them is swept over

    alpha = 1.0 (softmax) / 1.25 / 1.5 (training value) / 1.75 / 2.0 (sparsemax)

HONESTY CONSTRAINT: the scorer and classifier were trained at alpha ~1.5, so every other alpha is
a deliberate train/test mismatch. This isolates the POOLING OPERATOR's causal contribution while
holding the learned scorer fixed -- it does not say what training at that alpha would give. The
alpha=1.5 column must therefore reproduce the CV numbers exactly, which doubles as the check that
the manual forward below matches the model's own.

Per alpha x backbone, over all 1330 out-of-fold ROIs (paired: the same diluted bags reused):

    QWK clean      out-of-fold accuracy
    QWK +25/+100   accuracy after appending 25% / 100% foreign patches (from train patients)
    gini / eff N   concentration of the attention on the clean bag

    python scripts/acr_alpha_sweep.py
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

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from cardinality_invariance import build, ds_of, runs_for  # noqa: E402
from models.novelty_attempts.a215_adaptive_sparse_gated_attention_pooling import (  # noqa: E402
    entmax_bisect,
)
from train_grading_reti import BACKBONE_CONFIG  # noqa: E402

ALPHAS = [1.0, 1.25, 1.5, 1.75, 2.0]
DILUTE = [0.25, 1.0]


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3]))


def pool(e: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.softmax(e, dim=0) if alpha == 1.0 else entmax_bisect(e, alpha)


@torch.no_grad()
def logits_of(model, x):
    """The trained scorer's (h, e) for one bag -- everything upstream of the pooling operator."""
    h = model.bottleneck(x)
    e = model.attention_W(model.attention_V(h) * model.attention_U(h)).squeeze(-1)
    return h, e


def stats(a: np.ndarray) -> tuple[float, float]:
    a = a / max(a.sum(), 1e-12)
    n = len(a)
    idx = np.arange(1, n + 1)
    gini = (2 * (idx * np.sort(a)).sum()) / (n * a.sum()) - (n + 1) / n
    return float(gini), float(1.0 / (a ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cvas")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cpu")

    # {(bb, alpha, variant): (ys, preds)} ; {(bb, alpha): [(gini, effN)]}
    acc = defaultdict(lambda: ([], []))
    conc = defaultdict(list)
    seen = set()

    for cfg, ckpt in runs_for(a.prefix):
        key = (cfg["backbone"], cfg["fold"])
        if key in seen:
            continue
        seen.add(key)
        bb = cfg["backbone"]
        ds = ds_of(bb, cfg.get("data_root", "data"))
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        tr, te = state["train_idx"], state["test_idx"]
        model = build(cfg, BACKBONE_CONFIG[bb]["dim"]).to(dev)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        rng = np.random.default_rng(a.seed + cfg["fold"])
        donor = torch.cat([ds[i][0].squeeze(0)
                           for i in rng.choice(tr, size=min(40, len(tr)), replace=False)], 0)

        for i in te:
            x, y = ds[i][0].squeeze(0).to(dev), ds.samples[i][1]
            bags = {"clean": x}
            for f in DILUTE:
                k = int(round(f * len(x)))
                bags[f"+{int(f * 100)}%"] = torch.cat(
                    [x, donor[rng.choice(len(donor), size=k, replace=False)]], 0)
            for tag, xx in bags.items():
                h, e = logits_of(model, xx)
                for al in ALPHAS:
                    w = pool(e, al)
                    yhat = float(model.classifier(torch.mv(h.t(), w)).view(-1)[0])
                    Y, P = acc[(bb, al, tag)]
                    Y.append(y); P.append(yhat)
                    if tag == "clean":
                        conc[(bb, al)].append(stats(w.numpy()))
        print(f"  done {bb} fold {cfg['fold']}", file=sys.stderr)

    print("\nACR PROFILE vs entmax order alpha — trained ASGAP scorer, pooling swept at inference"
          "\n(alpha 1.5 = the training value; its clean column must reproduce the CV numbers)\n")
    h = (f"{'backbone':9s} {'alpha':>6s} {'QWK clean':>10s} {'QWK +25%':>9s} {'QWK +100%':>10s}"
         f" {'gini':>7s} {'eff N':>7s}")
    print(h); print("-" * len(h))
    for bb in ("virchow2", "uni2", "titan"):
        for al in ALPHAS:
            if (bb, al, "clean") not in acc:
                continue
            cells = []
            for tag in ("clean", "+25%", "+100%"):
                Y, P = acc[(bb, al, tag)]
                cells.append(qwk(np.array(Y), np.clip(np.round(P), 0, 3).astype(int)))
            g = np.array(conc[(bb, al)])
            name = {1.0: "1.00*", 2.0: "2.00+"}.get(al, f"{al:.2f}")
            print(f"{bb:9s} {name:>6s} {cells[0]:10.4f} {cells[1]:9.4f} {cells[2]:10.4f} "
                  f"{g[:, 0].mean():7.3f} {g[:, 1].mean():7.1f}")
        print()
    print("  * = softmax (the ABMIL operator)   + = sparsemax (the MINN-SA operator)")


if __name__ == "__main__":
    main()
