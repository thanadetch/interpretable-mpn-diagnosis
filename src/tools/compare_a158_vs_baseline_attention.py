"""Compare a158 (diffuse-temperature) vs baseline ABMIL attention weights on the SAME test bags.

Question: does a158 assign DIFFERENT patch weights than the baseline, or is it effectively the same
read (T learned ~0.98 ~ 1 => standard softmax)? We quantify, on virchow2 test bags:
  (1) attention ENTROPY (diffuseness) per bag: baseline vs a158
  (2) per-bag SPEARMAN correlation of the two weight vectors (ranking agreement)
  (3) top-10% patch JACCARD overlap (do they spotlight the same patches)
  (4) PURE-T effect: on a158's OWN energies, entropy(softmax(e)) vs entropy(softmax(e/T)) -- isolates
      what the temperature alone does, holding the scorer fixed.
Saves a 4-panel figure + prints summary. Analysis only (CPU); does not touch training.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL
from models.novelty_attempts.a158_diffuse_temperature_gated import Model as A158

BB = "virchow2"
DIM = BACKBONE_CONFIG[BB]["dim"]


def load(model, pattern):
    ck = sorted(glob.glob(str(ROOT / pattern)))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    return model.eval(), ck


def entropy(w):
    w = w.clamp(min=1e-12)
    return float(-(w * w.log()).sum() / np.log(len(w)))


def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-8); rb = (rb - rb.mean()) / (rb.std() + 1e-8)
    return float((ra * rb).mean())


def main():
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    test_idx = patient_split(ds, seed=2)[2]
    base, bck = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0),
                     "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
    a158, ack = load(A158(input_dim=DIM, num_classes=1),
                     "experiments/*/r48_a158_virchow2_s2_*/best_*.pth")
    T = float(F.softplus(a158.T_raw))
    print(f"baseline: {Path(bck).parent.name}")
    print(f"a158:     {Path(ack).parent.name}   learned T={T:.4f}")

    ent_b, ent_a, sp, jac, ent_T0, ent_T = [], [], [], [], [], []
    with torch.no_grad():
        for i in test_idx:
            feat, _, _ = ds[i]; feat = feat.float()
            _, wb, _ = base(feat, return_attention=True)
            _, wa, _ = a158(feat, return_attention=True)
            wb = wb.flatten(); wa = wa.flatten()
            ent_b.append(entropy(wb)); ent_a.append(entropy(wa))
            sp.append(spearman(wb, wa))
            k = max(1, int(0.10 * len(wb)))
            tb = set(wb.topk(k).indices.tolist()); ta = set(wa.topk(k).indices.tolist())
            jac.append(len(tb & ta) / len(tb | ta))
            # pure-T: a158 energies
            h = a158.bottleneck(feat)
            e = a158.attention_W(a158.attention_V(h) * a158.attention_U(h)).squeeze(-1)
            ent_T0.append(entropy(F.softmax(e, dim=0)))
            ent_T.append(entropy(F.softmax(e / T, dim=0)))

    ent_b, ent_a = np.array(ent_b), np.array(ent_a)
    print(f"\nn_test_bags={len(ent_b)}")
    print(f"attention entropy (1=uniform/diffuse, 0=peaked):")
    print(f"  baseline mean={ent_b.mean():.4f}  a158 mean={ent_a.mean():.4f}  (Δ={ent_a.mean()-ent_b.mean():+.4f})")
    print(f"per-bag Spearman(weights base vs a158): mean={np.mean(sp):.4f}  (1=identical ranking)")
    print(f"top-10% patch Jaccard overlap:          mean={np.mean(jac):.4f}  (1=same patches)")
    print(f"PURE-T effect (a158 energies): entropy softmax(e)={np.mean(ent_T0):.4f} -> softmax(e/T)={np.mean(ent_T):.4f} (Δ={np.mean(ent_T)-np.mean(ent_T0):+.4f})")

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].scatter(ent_b, ent_a, s=10, alpha=0.5)
    lo = min(ent_b.min(), ent_a.min()); hi = max(ent_b.max(), ent_a.max())
    ax[0, 0].plot([lo, hi], [lo, hi], "r--", lw=1)
    ax[0, 0].set_xlabel("baseline attention entropy"); ax[0, 0].set_ylabel("a158 attention entropy")
    ax[0, 0].set_title(f"Diffuseness per bag (on=identical)\nbase {ent_b.mean():.3f} vs a158 {ent_a.mean():.3f}")
    ax[0, 1].hist(sp, bins=30, color="steelblue"); ax[0, 1].axvline(np.mean(sp), color="r", ls="--")
    ax[0, 1].set_xlabel("per-bag Spearman(weights)"); ax[0, 1].set_title(f"Weight-ranking agreement (mean {np.mean(sp):.3f})")
    ax[1, 0].hist(jac, bins=30, color="seagreen"); ax[1, 0].axvline(np.mean(jac), color="r", ls="--")
    ax[1, 0].set_xlabel("top-10% patch Jaccard"); ax[1, 0].set_title(f"Same salient patches? (mean {np.mean(jac):.3f})")
    # example: sorted weight profiles of the largest bag
    biggest = max(test_idx, key=lambda i: ds[i][0].shape[0])
    feat, _, _ = ds[biggest]; feat = feat.float()
    with torch.no_grad():
        _, wb, _ = base(feat, return_attention=True); _, wa, _ = a158(feat, return_attention=True)
    ax[1, 1].plot(np.sort(wb.flatten().numpy())[::-1], label="baseline", lw=2)
    ax[1, 1].plot(np.sort(wa.flatten().numpy())[::-1], label="a158", lw=2, ls="--")
    ax[1, 1].set_xlabel("patch rank"); ax[1, 1].set_ylabel("attention weight")
    ax[1, 1].set_title(f"Sorted weight profile (largest bag, N={feat.shape[0]})"); ax[1, 1].legend()
    fig.suptitle(f"a158 (learned T={T:.3f}) vs baseline ABMIL attention — virchow2 test", fontsize=13)
    out = ROOT / "results" / "a158_vs_baseline_attention.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    main()
