"""Test the hypothesis: does a215 put MORE attention on actual fibrosis than the baseline?

We must not assume "concentrated attention" == "attends fibrosis". To check it honestly we use a
DATA-DERIVED fibrosis axis (NOT zero-shot concept prompts, which are prohibited):
    v = mean(raw patch feats in G2/G3 bags) - mean(raw patch feats in G0/G1 bags)   [derived on TRAIN]
The diagnostic established this direction tracks fibrosis grade (held-out coverage Spearman ~+0.84).
For every TEST bag we project each patch onto v_hat to get a per-patch fibrosis score, then compare
baseline vs a215 attention by:
  (1) within-bag Spearman(attention, fibrosis-score)  -- does attention align with fibre density?
  (2) attention-weighted mean fibrosis-score MINUS the unweighted (uniform) mean -- the "lift": how much
      more fibrotic are the patches a model up-weights vs reading every patch equally.
Higher = the model steers attention toward the more-fibrotic sub-regions. Reported overall and per grade.
Analysis only; CPU.
"""
from __future__ import annotations
import argparse, sys, glob
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL
from models.novelty_attempts.a215_adaptive_sparse_gated_attention_pooling import Model as A215

BB = "virchow2"; DIM = BACKBONE_CONFIG[BB]["dim"]


def load(model, pattern):
    hits = sorted(glob.glob(str(ROOT / pattern)))
    if not hits:
        raise FileNotFoundError(f"no checkpoint matches {pattern!r} - pass --abmil_ckpt/--asgap_ckpt")
    ck = hits[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd); return model.eval()


def spear(a, b):
    if len(a) < 3: return np.nan
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9); rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def parse_args():
    # The checkpoints used to be hard-coded and went stale when those runs were deleted.
    # Defaults point at the seed-2, no-augmentation, 50-epoch runs of each aggregator; pass
    # explicit globs to compare any other pair.
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--abmil_ckpt", default="experiments/*/r15_baseline_virchow2_s2_reti_simple_virchow2_*/best_*.pth")
    p.add_argument("--asgap_ckpt", default="experiments/*/as_v2_off_reti_novelty_attempt_virchow2_regression*/best_*.pth")
    p.add_argument("--seed", type=int, default=2, help="Split seed; must match the checkpoints' split.")
    return p.parse_args()


def main():
    args = parse_args()
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    tr, va, te = patient_split(ds, seed=args.seed)

    # ---- derive fibrosis axis v on TRAIN raw features (patch grade = its bag grade) ----
    hi, lo = [], []   # G2/G3 vs G0/G1
    for i in tr:
        f, lab, _ = ds[i]; f = f.float().numpy(); g = int(lab)
        (hi if g >= 2 else lo).append(f.mean(0))   # bag-mean keeps every ROI equal weight
    v = np.mean(hi, 0) - np.mean(lo, 0)
    v = v / (np.linalg.norm(v) + 1e-9)
    print(f"fibrosis axis derived on {len(tr)} train bags (||v||=1, dim={len(v)})")

    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0), args.abmil_ckpt)
    a215 = load(A215(input_dim=DIM, num_classes=1), args.asgap_ckpt)

    rows = []
    with torch.no_grad():
        for i in te:
            f, lab, _ = ds[i]; ff = f.float()
            proj = (f.float().numpy() @ v)                 # per-patch fibrosis score
            if len(proj) < 3: continue
            _, wb, _ = base(ff, return_attention=True); _, wa, _ = a215(ff, return_attention=True)
            wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
            unif = proj.mean()
            rows.append(dict(
                g=int(lab),
                sp_b=spear(wb, proj), sp_a=spear(wa, proj),
                lift_b=float((wb * proj).sum() - unif), lift_a=float((wa * proj).sum() - unif),
                std=float(proj.std())))
    rows = [r for r in rows if not np.isnan(r["sp_b"])]
    A = lambda k: np.array([r[k] for r in rows])
    print(f"\nn_test_bags={len(rows)}   (per-patch fibrosis-score std within bag ~ {A('std').mean():.3f})")
    print("=== within-bag Spearman(attention, fibrosis-score)  [+ = attention follows fibre density] ===")
    print(f"  baseline: {A('sp_b').mean():+.4f}      a215: {A('sp_a').mean():+.4f}      Δ(a215-base)={A('sp_a').mean()-A('sp_b').mean():+.4f}")
    print("=== attention-weighted fibrosis-score LIFT over uniform  [+ = up-weights fibrotic patches] ===")
    print(f"  baseline: {A('lift_b').mean():+.4f}      a215: {A('lift_a').mean():+.4f}      Δ(a215-base)={A('lift_a').mean()-A('lift_b').mean():+.4f}")
    print("\n--- per grade (lift over uniform) ---")
    for g in range(4):
        gr = [r for r in rows if r["g"] == g]
        if not gr: continue
        lb = np.mean([r["lift_b"] for r in gr]); la = np.mean([r["lift_a"] for r in gr])
        sb = np.mean([r["sp_b"] for r in gr]); sa = np.mean([r["sp_a"] for r in gr])
        print(f"  G{g} (n={len(gr):3d}): lift base {lb:+.4f} -> a215 {la:+.4f} (Δ{la-lb:+.4f}) | spearman base {sb:+.3f} -> a215 {sa:+.3f}")
    # how often a215 is MORE fibrosis-aligned than baseline
    more = np.mean(A('lift_a') > A('lift_b'))
    print(f"\nfraction of bags where a215 up-weights fibrosis MORE than baseline: {more*100:.1f}%")


if __name__ == "__main__":
    main()
