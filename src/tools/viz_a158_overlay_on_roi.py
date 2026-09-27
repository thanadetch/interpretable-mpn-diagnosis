"""Overlay baseline vs a158 attention ON the reconstructed ROI image.

Stitches the ROI back from its patches (224px, stride 112) and overlays each model's attention as a
glow whose opacity scales with the patch weight -> you SEE which tissue regions each model attends.
Panels per ROI: [reconstructed ROI | + baseline attention | + a158 attention]. Analysis only, CPU.
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
from models.novelty_attempts.a158_diffuse_temperature_gated import Model as A158

BB = "virchow2"; DIM = BACKBONE_CONFIG[BB]["dim"]; P = 224; S = 112


def load(model, pattern):
    ck = sorted(glob.glob(str(ROOT / pattern)))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd); return model.eval()


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
    up = np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR), np.float32)
    return up


def overlay(canvas, wm, gamma=0.7):
    n = wm / (wm.max() + 1e-9)
    rgba = cm.get_cmap("inferno")(n)[..., :3].astype(np.float32)
    alpha = (n ** gamma)[..., None] * 0.85
    gray = canvas.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 0.2   # dim background for contrast
    return (1 - alpha) * gray + alpha * rgba


def main():
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    te = patient_split(ds, seed=2)[2]
    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0),
                "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
    a158 = load(A158(input_dim=DIM, num_classes=1), "experiments/*/r48_a158_virchow2_s2_*/best_*.pth")
    T = float(F.softplus(a158.T_raw))
    out = ROOT / "results" / "a158_overlay_on_roi"; out.mkdir(parents=True, exist_ok=True)
    allg = {g: [] for g in range(4)}
    for i in te: allg[int(ds.samples[i][1])].append(i)

    SQ = 360  # uniform square cell so the grid is easy to scan
    def square(arr):
        a = np.clip(arr, 0, 1)
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((SQ, SQ), Image.BILINEAR), np.float32) / 255.0

    fig, ax = plt.subplots(4, 3, figsize=(11, 14.5))
    cols = ["ROI (original)", "baseline attention", f"a158 attention (T={T:.3f})"]
    for c in range(3): ax[0, c].set_title(cols[c], fontsize=13)
    for g in range(4):
        i = allg[g][len(allg[g]) // 2]
        pt, label = ds.samples[i]
        bag = torch.load(pt, map_location="cpu", weights_only=False)
        feats = bag["feats"].float()
        with torch.no_grad():
            _, wb, _ = base(feats, return_attention=True)
            _, wa, _ = a158(feats, return_attention=True)
        wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
        canvas, rc, R, C, H, W = stitch(bag)
        cells = [square(canvas), square(overlay(canvas, weight_map(rc, wb, H, W))),
                 square(overlay(canvas, weight_map(rc, wa, H, W)))]
        for c in range(3):
            ax[g, c].imshow(cells[c]); ax[g, c].set_xticks([]); ax[g, c].set_yticks([])
        ax[g, 0].set_ylabel(f"G{label}\n(N={len(wb)}, {R}x{C})", fontsize=12, rotation=0, labelpad=38, va="center")
        print(f"row G{label}: N={len(wb)} base_maxw={wb.max():.3f} a158_maxw={wa.max():.3f}")
    fig.suptitle("Attention on the tissue (virchow2 test) — baseline vs a158, per grade\n"
                 "attention glows where weight is high; brightness ∝ patch weight", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = out / "overlay_grid.png"; fig.savefig(fp, dpi=135); plt.close(fig)
    print(f"\nsaved GRID -> {fp}")


if __name__ == "__main__":
    main()
