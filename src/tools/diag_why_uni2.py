"""Diagnose WHY uni2 underperforms (and why sparsity mechanisms like a215 hurt it most).

For each backbone (titan, virchow2, uni2) on the SAME seed-2 split, measure two independent things:

  (1) INTRINSIC feature-space quality for fibrosis (no trained head):
      - fibrosis axis v = mean(bag-mean | G2/G3) - mean(bag-mean | G0/G1), derived on TRAIN.
      - test Spearman( bag-mean projection onto v , grade )  -> how linearly grade-separable the bag is.
      - nearest-grade-centroid QWK on test (centroids = train bag-means per grade) -> head-free accuracy.

  (2) ATTENTION concentration:
      - baseline ABMIL attention entropy on test (1=diffuse, 0=peaked).
      - a215 learned alpha + a215 attention entropy, and the entropy DROP a215 induces.

If uni2 has (1) lower separability AND (2) already-low attention entropy, that explains both why its
baseline is weaker and why adding sparsity (a215) over-concentrates an already-peaked read and collapses it.
Analysis only; CPU.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL
from models.novelty_attempts.a215_adaptive_sparse_gated_attention_pooling import Model as A215

BASE = {
    "titan": "experiments/20260520_grading_full_data_ablation_seed_2_G1_3_3/06_reti_simple_titan_regression_*/best_*.pth",
    "virchow2": "experiments/20260520_grading_full_data_ablation_seed_2_G1_3_3/04_reti_simple_virchow2_regression_*/best_*.pth",
    "uni2": "experiments/20260520_grading_full_data_ablation_seed_2_G1_3_3/02_reti_simple_uni2_regression_*/best_*.pth",
}
A215CK = {bb: f"experiments/*/casc_a215_{bb}_*/best_*.pth" for bb in BASE}


def load(model, pattern):
    ck = sorted(glob.glob(str(ROOT / pattern)))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd); return model.eval()


def entropy(w):
    w = np.clip(w, 1e-12, None)
    return float(-(w * np.log(w)).sum() / np.log(len(w))) if len(w) > 1 else 1.0


def spear(a, b):
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9); rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def qwk(p, t, K=4):
    p = np.clip(np.round(p), 0, K - 1).astype(int); t = t.astype(int)
    O = np.zeros((K, K))
    for pi, ti in zip(p, t): O[ti, pi] += 1
    w = np.array([[(i - j) ** 2 / (K - 1) ** 2 for j in range(K)] for i in range(K)])
    act = O.sum(1); pre = O.sum(0); E = np.outer(act, pre) / O.sum()
    return 1 - (w * O).sum() / max((w * E).sum(), 1e-9)


def main():
    print(f"{'backbone':9s} {'dim':>5s} | {'sep-Spearman':>12s} {'centroid-QWK':>12s} | "
          f"{'base-entropy':>12s} {'a215-α':>7s} {'a215-entropy':>12s} {'Δentropy':>9s}")
    print("-" * 92)
    for bb in ["titan", "virchow2", "uni2"]:
        dim = BACKBONE_CONFIG[bb]["dim"]
        ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[bb]["feature_dir"])
        tr, va, te = patient_split(ds, seed=2)

        # ---- (1) intrinsic feature separability (head-free) ----
        trm = [(ds[i][0].float().numpy().mean(0), int(ds[i][1])) for i in tr]
        tem = [(ds[i][0].float().numpy().mean(0), int(ds[i][1])) for i in te]
        hi = np.mean([m for m, g in trm if g >= 2], 0); lo = np.mean([m for m, g in trm if g < 2], 0)
        v = (hi - lo); v /= (np.linalg.norm(v) + 1e-9)
        proj = np.array([m @ v for m, _ in tem]); gt = np.array([g for _, g in tem])
        sep = spear(proj, gt)
        cents = np.stack([np.mean([m for m, g in trm if g == k], 0) for k in range(4)])
        pred = np.array([int(np.argmin(((m - cents) ** 2).sum(1))) for m, _ in tem])
        cqwk = qwk(pred.astype(float), gt)

        # ---- (2) attention concentration ----
        base = load(SimpleGatedMIL(input_dim=dim, num_classes=1, topk=0), BASE[bb])
        a215 = load(A215(input_dim=dim, num_classes=1), A215CK[bb])
        alpha = float((1.0 + torch.sigmoid(a215.alpha_raw)).item())
        eb, ea = [], []
        with torch.no_grad():
            for i in te:
                f = ds[i][0].float()
                _, wb, _ = base(f, return_attention=True); _, wa, _ = a215(f, return_attention=True)
                eb.append(entropy(wb.flatten().numpy())); ea.append(entropy(wa.flatten().numpy()))
        eb, ea = np.mean(eb), np.mean(ea)
        print(f"{bb:9s} {dim:5d} | {sep:12.3f} {cqwk:12.4f} | {eb:12.4f} {alpha:7.3f} {ea:12.4f} {ea-eb:+9.4f}")


if __name__ == "__main__":
    main()
