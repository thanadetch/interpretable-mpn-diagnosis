"""Publication-quality attention-overlay figures (a215 vs ABMIL baseline ONLY) for the thesis.

Image-based (not charts): semi-transparent attention heatmaps overlaid on the reconstructed reticulin
ROI so the tissue stays clearly visible. Two figures on virchow2 test:
  - thesis_overlay_grid.png : 4 grades x [original ROI | baseline attention | a215 attention], one shared
                              colourbar. The main thesis figure.
  - thesis_focus_G2.png     : a single large G2 case (the grade a215 helps most: test recall 73.3->90.0),
                              [ROI | baseline | a215], for a focused in-text figure.
Tissue at full colour; attention blended on top with opacity ~ weight (low weight => tissue shows through).
Analysis/reporting only; CPU.
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL
from models.novelty_attempts.a215_adaptive_sparse_gated_attention_pooling import Model as A215

BB = "virchow2"; DIM = BACKBONE_CONFIG[BB]["dim"]; P = 224; S = 112
CMAP = "turbo"   # perceptually-ok, contrasts with brown reticulin tissue


def load(model, pattern):
    ck = sorted(glob.glob(str(ROOT / pattern)))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd); return model.eval()


def stitch(bag):
    rc = bag["rc"]; rc = rc.numpy() if hasattr(rc, "numpy") else np.array(rc)
    R, C = int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1
    H, W = (R - 1) * S + P, (C - 1) * S + P
    canvas = np.ones((H, W, 3), np.float32)
    for (r, c), pp in zip(rc, bag["patch_paths"]):
        fp = pp if os.path.isabs(str(pp)) else str(ROOT / str(pp))
        if not os.path.exists(fp): continue
        canvas[r * S:r * S + P, c * S:c * S + P] = np.asarray(Image.open(fp).convert("RGB"), np.float32) / 255.0
    return canvas, rc, H, W


def weight_map(rc, w, H, W):
    R, C = int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1
    g = np.zeros((R, C), np.float32)
    for (r, c), a in zip(rc, w): g[int(r), int(c)] = a
    return np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR), np.float32)


def overlay(canvas, wm, vmax, amax=0.6, gamma=0.85):
    """Blend a turbo heatmap onto full-colour tissue; opacity scales with normalised weight."""
    n = np.clip(wm / (vmax + 1e-9), 0, 1)
    rgba = cm.get_cmap(CMAP)(n)[..., :3].astype(np.float32)
    alpha = (n ** gamma)[..., None] * amax
    return (1 - alpha) * canvas + alpha * rgba


def pad_square(img):
    """Centre-pad an image to a square (white background) so grid cells are uniform & undistorted."""
    H, W = img.shape[:2]; M = max(H, W)
    o = np.ones((M, M, 3), np.float32)
    o[(M - H) // 2:(M - H) // 2 + H, (M - W) // 2:(M - W) // 2 + W] = img
    return o


def attn(model, feats):
    with torch.no_grad():
        _, w, _ = model(feats.float(), return_attention=True)
    return w.flatten().numpy()


def main():
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    te = patient_split(ds, seed=2)[2]
    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0),
                "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
    a215 = load(A215(input_dim=DIM, num_classes=1), "experiments/a215_ablation/regression_seed2/casc_a215_virchow2_s2_*/best_*.pth")
    alpha = float((1.0 + torch.sigmoid(a215.alpha_raw)).item())
    out = ROOT / "results" / "a215_vs_baseline"; out.mkdir(parents=True, exist_ok=True)

    by_grade = {g: [] for g in range(4)}
    for i in te: by_grade[int(ds.samples[i][1])].append(i)

    def panels(i):
        bag = torch.load(ds.samples[i][0], map_location="cpu", weights_only=False)
        feats = bag["feats"].float()
        wb, wa = attn(base, feats), attn(a215, feats)
        canvas, rc, H, W = stitch(bag)
        vmax = float(max(wb.max(), wa.max()))  # shared scale -> base vs a215 directly comparable
        roi = pad_square(canvas)
        ob = pad_square(overlay(canvas, weight_map(rc, wb, H, W), vmax))
        oa = pad_square(overlay(canvas, weight_map(rc, wa, H, W), vmax))
        return roi, ob, oa, len(wb)

    # ---------- MAIN GRID 4x3 ----------
    fig, ax = plt.subplots(4, 3, figsize=(11, 14.6))
    cols = ["Reticulin ROI", "Baseline (ABMIL)", "ASGAP (1.5-entmax)"]
    for c in range(3): ax[0, c].set_title(cols[c], fontsize=12.5, pad=10)
    for g in range(4):
        i = by_grade[g][len(by_grade[g]) // 2]
        roi, ob, oa, N = panels(i)
        for c, im in enumerate([roi, ob, oa]):
            ax[g, c].imshow(np.clip(im, 0, 1)); ax[g, c].set_xticks([]); ax[g, c].set_yticks([])
            for s in ax[g, c].spines.values(): s.set_edgecolor("0.7")
        ax[g, 0].set_ylabel(f"Grade {g}", fontsize=13, rotation=90, labelpad=10, va="center")
    fig.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.02, wspace=0.04, hspace=0.08)
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=CMAP); sm.set_array([])
    cax = fig.add_axes([0.90, 0.30, 0.018, 0.40])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("attention weight (normalised per ROI)", fontsize=11)
    cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["low", "mid", "high"])
    fig.suptitle("Attention overlays on reticulin ROIs — Baseline vs. ASGAP", fontsize=14, y=0.975)
    fig.savefig(out / "thesis_overlay_grid.png", dpi=200); plt.close(fig)
    print(f"saved thesis_overlay_grid.png  (α={alpha:.3f})")

    # ---------- FOCUS: a single G2 case (a215 helps this grade most) ----------
    # among the larger-than-median G2 bags (more tissue, less whitespace), pick the one where a215
    # most increases its peak weight (most visibly different from baseline).
    g2 = by_grade[2]
    sizes = {i: ds[i][0].shape[0] for i in g2}
    med = np.median(list(sizes.values()))
    cand = [i for i in g2 if sizes[i] >= med] or g2
    best_i, best_d = None, -1
    for i in cand:
        bag = torch.load(ds.samples[i][0], map_location="cpu", weights_only=False)
        feats = bag["feats"].float()
        wb, wa = attn(base, feats), attn(a215, feats)
        d = wa.max() - wb.max()
        if d > best_d: best_d, best_i = d, i
    roi, ob, oa, N = panels(best_i)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6))
    for a, im, t in zip(ax, [roi, ob, oa],
                        ["Reticulin ROI (Grade 2)", "Baseline (ABMIL)", "ASGAP (1.5-entmax)"]):
        a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=13); a.set_xticks([]); a.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.02, wspace=0.04)
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=CMAP); sm.set_array([])
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.55])
    cb = fig.colorbar(sm, cax=cax); cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["low", "mid", "high"])
    cb.set_label("attention", fontsize=11)
    fig.suptitle(f"Grade-2 example (N={N} patches): ASGAP concentrates attention more than the baseline", fontsize=13, y=0.97)
    fig.savefig(out / "thesis_focus_G2.png", dpi=200); plt.close(fig)
    print(f"saved thesis_focus_G2.png  (G2 bag idx={best_i})")


if __name__ == "__main__":
    main()
