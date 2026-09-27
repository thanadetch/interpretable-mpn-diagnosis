"""Visualize a215 (learnable-alpha entmax) attention vs the ABMIL baseline ONLY.

Compares ONLY a215 against the simple-gate baseline (no other novelty). On virchow2 test bags it
produces two figures:
  (A) spatial overlay grid: [ROI | baseline (diffuse softmax) | a215 (entmax, learned alpha)] per grade,
      attention glowing on the reconstructed tissue -> SEE where each model looks.
  (B) quantitative 4-panel: attention entropy (diffuse<->sparse) baseline vs a215; fraction of patches
      entmax zeroes per bag (the sparsity a215 actually uses); per-bag weight-rank agreement; sorted
      weight profile of the largest bag (shows the exact-zero tail). Analysis only, CPU.
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL
from models.novelty_attempts.a215_adaptive_sparse_gated_attention_pooling import Model as A215

BB = "virchow2"; DIM = BACKBONE_CONFIG[BB]["dim"]; P = 224; S = 112
ZERO = 1e-6  # weight below this counts as "zeroed by entmax"


def load(model, pattern):
    ck = sorted(glob.glob(str(ROOT / pattern)))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd); return model.eval()


def entropy(w):
    w = np.clip(w, 1e-12, None)
    return float(-(w * np.log(w)).sum() / np.log(len(w)))


def stitch(bag):
    rc = bag["rc"]; rc = rc.numpy() if hasattr(rc, "numpy") else np.array(rc)
    R, C = int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1
    H, W = (R - 1) * S + P, (C - 1) * S + P
    canvas = np.ones((H, W, 3), np.float32) * 0.95
    for (r, c), pp in zip(rc, bag["patch_paths"]):
        fp = pp if os.path.isabs(str(pp)) else str(ROOT / str(pp))
        if not os.path.exists(fp): continue
        img = np.asarray(Image.open(fp).convert("RGB"), np.float32) / 255.0
        canvas[r * S:r * S + P, c * S:c * S + P] = img
    return canvas, rc, R, C, H, W


def weight_map(rc, w, H, W):
    R, C = int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1
    g = np.zeros((R, C), np.float32)
    for (r, c), a in zip(rc, w): g[int(r), int(c)] = a
    return np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR), np.float32)


def overlay(canvas, wm, gamma=0.7):
    n = wm / (wm.max() + 1e-9)
    rgba = cm.get_cmap("inferno")(n)[..., :3].astype(np.float32)
    alpha = (n ** gamma)[..., None] * 0.85
    gray = canvas.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 0.2
    return (1 - alpha) * gray + alpha * rgba


def square(arr, SQ=360):
    a = np.clip(arr, 0, 1)
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((SQ, SQ), Image.BILINEAR), np.float32) / 255.0


def main():
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    te = patient_split(ds, seed=2)[2]
    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0),
                "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
    a215 = load(A215(input_dim=DIM, num_classes=1),
                "experiments/a215_ablation/regression_seed2/casc_a215_virchow2_s2_*/best_*.pth")
    alpha = float(1.0 + torch.sigmoid(a215.alpha_raw))
    print(f"a215 learned alpha = {alpha:.4f}  (1=softmax/dense, 2=sparsemax/sparse)")
    out = ROOT / "results" / "a215_vs_baseline"; out.mkdir(parents=True, exist_ok=True)

    # ---------- collect per-bag stats over ALL test bags ----------
    ent_b, ent_a, frac0, sp = [], [], [], []
    with torch.no_grad():
        for i in te:
            feat, _, _ = ds[i]; feat = feat.float()
            _, wb, _ = base(feat, return_attention=True)
            _, wa, _ = a215(feat, return_attention=True)
            wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
            ent_b.append(entropy(wb)); ent_a.append(entropy(wa))
            frac0.append(float((wa < ZERO).mean()))
            ra = wb.argsort().argsort().astype(float); rb = wa.argsort().argsort().astype(float)
            ra = (ra - ra.mean()) / (ra.std() + 1e-9); rb = (rb - rb.mean()) / (rb.std() + 1e-9)
            sp.append(float((ra * rb).mean()))
    ent_b, ent_a, frac0, sp = map(np.array, (ent_b, ent_a, frac0, sp))
    print(f"n_test_bags={len(ent_b)}")
    print(f"entropy(1=diffuse,0=peaked): baseline {ent_b.mean():.4f}  a215 {ent_a.mean():.4f}  (Δ={ent_a.mean()-ent_b.mean():+.4f})")
    print(f"a215 fraction of patches zeroed: mean={frac0.mean():.4f}  max={frac0.max():.4f}  (bags w/ any zero: {(frac0>0).mean()*100:.1f}%)")
    print(f"per-bag weight-rank agreement: mean={sp.mean():.4f}")

    # ---------- Figure A: spatial overlay grid ----------
    allg = {g: [] for g in range(4)}
    for i in te: allg[int(ds.samples[i][1])].append(i)
    fig, ax = plt.subplots(4, 3, figsize=(11, 14.5))
    cols = ["ROI (original)", "baseline — DIFFUSE (softmax)", "ASGAP (1.5-entmax)"]
    for c in range(3): ax[0, c].set_title(cols[c], fontsize=12)
    for g in range(4):
        i = allg[g][len(allg[g]) // 2]
        pt, label = ds.samples[i]
        bag = torch.load(pt, map_location="cpu", weights_only=False)
        feats = bag["feats"].float()
        with torch.no_grad():
            _, wb, _ = base(feats, return_attention=True)
            _, wa, _ = a215(feats, return_attention=True)
        wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
        nz = int((wa < ZERO).sum())
        canvas, rc, R, C, H, W = stitch(bag)
        cells = [square(canvas), square(overlay(canvas, weight_map(rc, wb, H, W))),
                 square(overlay(canvas, weight_map(rc, wa, H, W)))]
        for c in range(3):
            ax[g, c].imshow(cells[c]); ax[g, c].set_xticks([]); ax[g, c].set_yticks([])
        ax[g, 0].set_ylabel(f"G{label}\n(N={len(wb)}, {R}x{C})", fontsize=12, rotation=0, labelpad=38, va="center")
        ax[g, 2].text(0.5, -0.06, f"{len(wa)-nz}/{len(wa)} patches kept ({nz} zeroed)",
                      transform=ax[g, 2].transAxes, ha="center", fontsize=9, color="dimgray")
        print(f"row G{label}: N={len(wb)} base_maxw={wb.max():.3f} a215_maxw={wa.max():.3f} zeroed={nz}")
    fig.suptitle("ASGAP (α = 1.5 (fixed) entmax) vs ABMIL baseline\n"
                 "attention glows where weight is high; brightness ∝ patch weight", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpA = out / "overlay_grid.png"; fig.savefig(fpA, dpi=135); plt.close(fig)
    print(f"saved -> {fpA}")

    # ---------- Figure B: quantitative 4-panel ----------
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].scatter(ent_b, ent_a, s=10, alpha=0.5)
    lo = min(ent_b.min(), ent_a.min()); hi = max(ent_b.max(), ent_a.max())
    ax[0, 0].plot([lo, hi], [lo, hi], "r--", lw=1)
    ax[0, 0].set_xlabel("baseline attention entropy"); ax[0, 0].set_ylabel("ASGAP attention entropy")
    ax[0, 0].set_title(f"Diffuseness per bag (diag=identical)\nbase {ent_b.mean():.3f} vs ASGAP {ent_a.mean():.3f}  (below diag = ASGAP sharper)")
    ax[0, 1].hist(frac0 * 100, bins=30, color="indianred")
    ax[0, 1].axvline(frac0.mean() * 100, color="k", ls="--")
    ax[0, 1].set_xlabel("% of patches entmax zeroes (weight=0)")
    ax[0, 1].set_title(f"How sparse ASGAP actually is (mean {frac0.mean()*100:.1f}% zeroed)\nα = 1.5 (fixed); α=1 would zero 0%")
    ax[1, 0].hist(sp, bins=30, color="steelblue"); ax[1, 0].axvline(sp.mean(), color="r", ls="--")
    ax[1, 0].set_xlabel("per-bag weight-rank agreement"); ax[1, 0].set_title(f"Same patch ranking as baseline? (mean {sp.mean():.3f})")
    biggest = max(te, key=lambda i: ds[i][0].shape[0])
    feat, _, _ = ds[biggest]; feat = feat.float()
    with torch.no_grad():
        _, wb, _ = base(feat, return_attention=True); _, wa, _ = a215(feat, return_attention=True)
    ax[1, 1].plot(np.sort(wb.flatten().numpy())[::-1], label="baseline (softmax)", lw=2)
    ax[1, 1].plot(np.sort(wa.flatten().numpy())[::-1], label="ASGAP (1.5-entmax)", lw=2, ls="--")
    ax[1, 1].set_xlabel("patch rank"); ax[1, 1].set_ylabel("attention weight")
    ax[1, 1].set_title(f"Sorted weight profile (largest bag, N={feat.shape[0]})\nentmax tail can hit exactly 0"); ax[1, 1].legend()
    fig.suptitle(f"ASGAP (α = 1.5 (fixed) entmax) vs ABMIL  (n={len(ent_b)} bags)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpB = out / "quantitative.png"; fig.savefig(fpB, dpi=130); plt.close(fig)
    print(f"saved -> {fpB}")


if __name__ == "__main__":
    main()
