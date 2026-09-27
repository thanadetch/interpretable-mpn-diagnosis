"""Thesis figure: reticulin ROIs where a215 CORRECTS a baseline grading error (virchow2 test).

Image-based (attention overlaid on real tissue), same clean style as thesis_overlay_grid. Selects test
ROIs where the baseline mis-grades the ROI but a215 (learnable-alpha entmax) grades it correctly, chosen
to show a spread: baseline over-predicting (pulled down) and under-predicting (pushed up). Each row:
[ROI | baseline attention (+ its wrong prediction) | a215 attention (+ its correct prediction)].
These are illustrative qualitative examples on the backbone/split where a215 helps (Virchow2 test);
a215 is NOT gate-robust across backbones. Analysis/reporting only; CPU.
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
CMAP = "turbo"


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
    n = np.clip(wm / (vmax + 1e-9), 0, 1)
    rgba = cm.get_cmap(CMAP)(n)[..., :3].astype(np.float32)
    alpha = (n ** gamma)[..., None] * amax
    return (1 - alpha) * canvas + alpha * rgba


def pad_square(img):
    H, W = img.shape[:2]; M = max(H, W)
    o = np.ones((M, M, 3), np.float32)
    o[(M - H) // 2:(M - H) // 2 + H, (M - W) // 2:(M - W) // 2 + W] = img
    return o


def main():
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    te = patient_split(ds, seed=2)[2]
    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0),
                "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
    a215 = load(A215(input_dim=DIM, num_classes=1), "experiments/a215_ablation/regression_seed2/casc_a215_virchow2_s2_*/best_*.pth")
    alpha = float((1.0 + torch.sigmoid(a215.alpha_raw)).item())
    out = ROOT / "results" / "a215_vs_baseline"; out.mkdir(parents=True, exist_ok=True)

    # gather predictions
    rec = []
    with torch.no_grad():
        for i in te:
            f, lab, _ = ds[i]; f = f.float(); g = int(lab)
            yb = float(base(f)[0].item()); ya = float(a215(f)[0].item())
            pb = int(np.clip(round(yb), 0, 3)); pa = int(np.clip(round(ya), 0, 3))
            rec.append(dict(i=i, g=g, yb=yb, ya=ya, pb=pb, pa=pa, n=f.shape[0]))
    fixes = [r for r in rec if r["pb"] != r["g"] and r["pa"] == r["g"]]

    def best(pool, key, n=1):
        return sorted(pool, key=key, reverse=True)[:n]

    # curated spread of corrections (prefer big error reduction & enough tissue)
    picks = []
    g1 = best([r for r in fixes if r["g"] == 1 and r["n"] >= 24],
              lambda r: abs(r["yb"] - r["g"]) - abs(r["ya"] - r["g"]))
    g2_over = best([r for r in fixes if r["g"] == 2 and r["pb"] > 2 and r["n"] >= 30],
                   lambda r: abs(r["yb"] - r["g"]) - abs(r["ya"] - r["g"]))
    g2_under = best([r for r in fixes if r["g"] == 2 and r["pb"] < 2 and r["n"] >= 25],
                    lambda r: abs(r["yb"] - r["g"]) - abs(r["ya"] - r["g"]))
    g3 = best([r for r in fixes if r["g"] == 3 and r["n"] >= 30],
              lambda r: abs(r["yb"] - r["g"]) - abs(r["ya"] - r["g"]))
    for grp in (g1, g2_over, g2_under, g3):
        if grp: picks.append(grp[0])
    print("picked corrections:")
    for r in picks: print(f"  idx={r['i']} N={r['n']} trueG{r['g']} base ŷ{r['yb']:.2f}→G{r['pb']} | a215 ŷ{r['ya']:.2f}→G{r['pa']}")

    nrow = len(picks)
    fig, ax = plt.subplots(nrow, 3, figsize=(11, 3.55 * nrow))
    if nrow == 1: ax = ax[None, :]
    titles = ["Reticulin ROI", "Baseline (ABMIL)", "ASGAP (1.5-entmax)"]
    for c in range(3): ax[0, c].set_title(titles[c], fontsize=12.5, pad=10)
    for row, r in enumerate(picks):
        bag = torch.load(ds.samples[r["i"]][0], map_location="cpu", weights_only=False)
        feats = bag["feats"].float()
        with torch.no_grad():
            _, wb, _ = base(feats, return_attention=True)
            _, wa, _ = a215(feats, return_attention=True)
        wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
        canvas, rc, H, W = stitch(bag)
        vmax = float(max(wb.max(), wa.max()))
        cells = [pad_square(canvas),
                 pad_square(overlay(canvas, weight_map(rc, wb, H, W), vmax)),
                 pad_square(overlay(canvas, weight_map(rc, wa, H, W), vmax))]
        for c in range(3):
            ax[row, c].imshow(np.clip(cells[c], 0, 1)); ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
            for s in ax[row, c].spines.values(): s.set_edgecolor("0.7")
        ax[row, 0].set_ylabel(f"True Grade {r['g']}", fontsize=12.5, labelpad=10)
        ax[row, 1].text(0.5, -0.07, f"predicts ŷ={r['yb']:.2f} → Grade {r['pb']}  ✗",
                        transform=ax[row, 1].transAxes, ha="center", fontsize=11, color="#b22222", weight="bold")
        ax[row, 2].text(0.5, -0.07, f"predicts ŷ={r['ya']:.2f} → Grade {r['pa']}  ✓",
                        transform=ax[row, 2].transAxes, ha="center", fontsize=11, color="#1a7a1a", weight="bold")

    fig.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.03, wspace=0.04, hspace=0.22)
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=CMAP); sm.set_array([])
    cax = fig.add_axes([0.90, 0.30, 0.018, 0.40]); cb = fig.colorbar(sm, cax=cax)
    cb.set_label("attention weight (normalised per ROI)", fontsize=11)
    cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["low", "mid", "high"])
    fig.suptitle("ROIs where ASGAP corrects a baseline grading error", fontsize=14, y=0.975)
    fig.savefig(out / "thesis_corrections.png", dpi=200); plt.close(fig)
    print(f"saved thesis_corrections.png  (n={nrow} cases, α={alpha:.3f})")


if __name__ == "__main__":
    main()
