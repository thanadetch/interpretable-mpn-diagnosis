"""Side-by-side SPATIAL attention heatmaps: baseline ABMIL vs a158, on the same ROI grid.

For example test ROIs (per grade), lay each model's attention weights on the (row,col) patch grid and
show: [baseline heatmap | a158 heatmap | difference a158-baseline]. Shared colour scale for the two
heatmaps; diverging map for the difference. Answers visually "do they weight different regions?"
(expectation: near-identical, since learned T~0.98 ~ standard softmax). Analysis only, CPU.
"""
from __future__ import annotations
import sys, glob, math, os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.simple_mil import SimpleGatedMIL
from models.novelty_attempts.a158_diffuse_temperature_gated import Model as A158
import torch.nn.functional as F

BB = "virchow2"; DIM = BACKBONE_CONFIG[BB]["dim"]; PER_GRADE = 2


def load(model, pattern):
    ck = sorted(glob.glob(str(ROOT / pattern)))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd); return model.eval()


def heat_grid(rc, attn):
    rc = rc.numpy() if hasattr(rc, "numpy") else np.array(rc)
    R, C = int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1
    h = np.full((R, C), np.nan)
    for (r, c), a in zip(rc, attn):
        h[int(r), int(c)] = a
    return h, R, C


def main():
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    te = patient_split(ds, seed=2)[2]
    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0),
                "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
    a158 = load(A158(input_dim=DIM, num_classes=1), "experiments/*/r48_a158_virchow2_s2_*/best_*.pth")
    T = float(F.softplus(a158.T_raw))
    out = ROOT / "results" / "a158_vs_baseline_heatmaps"; out.mkdir(parents=True, exist_ok=True)

    allg = {g: [] for g in range(4)}
    for i in te: allg[int(ds.samples[i][1])].append(i)
    for g in range(4):
        lst = allg[g]
        step = max(1, len(lst) // PER_GRADE)
        picks = lst[::step][:PER_GRADE]
        for n, i in enumerate(picks):
            pt, label = ds.samples[i]
            bag = torch.load(pt, map_location="cpu", weights_only=False)
            feats = bag["feats"].float(); rc = bag["rc"]
            with torch.no_grad():
                yb, wb, _ = base(feats, return_attention=True)
                ya, wa, _ = a158(feats, return_attention=True)
            wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
            hb, R, C = heat_grid(rc, wb); ha, _, _ = heat_grid(rc, wa)
            diff = ha - hb
            vmax = np.nanmax([np.nanmax(hb), np.nanmax(ha)]); vmin = 0.0
            dmax = np.nanmax(np.abs(diff))
            sp = np.corrcoef(wb.argsort().argsort(), wa.argsort().argsort())[0, 1]

            fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
            im0 = ax[0].imshow(hb, cmap="inferno", vmin=vmin, vmax=vmax)
            ax[0].set_title(f"baseline ABMIL\nmax_w={np.nanmax(hb):.3f}"); fig.colorbar(im0, ax=ax[0], fraction=0.046)
            im1 = ax[1].imshow(ha, cmap="inferno", vmin=vmin, vmax=vmax)
            ax[1].set_title(f"a158 (T={T:.3f})\nmax_w={np.nanmax(ha):.3f}"); fig.colorbar(im1, ax=ax[1], fraction=0.046)
            im2 = ax[2].imshow(diff, cmap="bwr", vmin=-dmax, vmax=dmax)
            ax[2].set_title(f"difference (a158 - base)\nmax|Δw|={dmax:.4f}"); fig.colorbar(im2, ax=ax[2], fraction=0.046)
            for a in ax: a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f"virchow2 test G{label}  (grid {R}x{C}, N={len(wb)})  |  weight-rank corr={sp:.3f}", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fp = out / f"hm_G{label}_{n}.png"; fig.savefig(fp, dpi=120); plt.close(fig)
            print(f"saved {fp.name}  G{label} N={len(wb)} rankcorr={sp:.3f} max|dW|={dmax:.4f} (vs max_w {vmax:.3f})")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
