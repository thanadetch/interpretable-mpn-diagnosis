"""Generate MANY thesis-style overlay grids (a215 vs ABMIL baseline) for the user to choose from.

Same clean style as thesis_overlay_grid.png (4 grades x [ROI | baseline | a215], turbo overlay on tissue,
shared colourbar), but produces N_GRIDS grids, each using a DIFFERENT example ROI per grade so the user
can browse and pick. Per grade, bags are ranked by "interestingness" (a215 corrects a baseline error >
larger error reduction > more visible attention sharpening); grid k uses the rank-k bag per grade, so the
earliest grids are the most compelling. Each panel is annotated with the predicted grade (✓/✗). Virchow2
test; analysis/reporting only; CPU.
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

BB = os.environ.get("BB", "virchow2"); DIM = BACKBONE_CONFIG[BB]["dim"]; P = 224; S = 112
CMAP = "turbo"
N_GRIDS = int(os.environ.get("N_GRIDS", "10"))
START = int(os.environ.get("GRID_START", "0"))   # render grids [START, N_GRIDS) (1-indexed file names)
BASE_PATTERN = {
    "virchow2": "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth",
    "titan": "experiments/20260520_grading_full_data_ablation_seed_2_G1_3_3/06_reti_simple_titan_regression_*/best_*.pth",
    "uni2": "experiments/20260520_grading_full_data_ablation_seed_2_G1_3_3/02_reti_simple_uni2_regression_*/best_*.pth",
}


def load(model, pattern):
    ck = sorted(glob.glob(str(ROOT / pattern)))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd); return model.eval()


def entropy(w):
    w = np.clip(w, 1e-12, None)
    return float(-(w * np.log(w)).sum() / np.log(len(w))) if len(w) > 1 else 1.0


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
    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0), BASE_PATTERN[BB])
    a215 = load(A215(input_dim=DIM, num_classes=1), f"experiments/a215_ablation/regression_seed2/casc_a215_{BB}_s2_*/best_*.pth")
    alpha = float((1.0 + torch.sigmoid(a215.alpha_raw)).item())
    out = ROOT / "results" / "a215_vs_baseline" / (f"grids_{BB}" if BB != "virchow2" else "grids")
    out.mkdir(parents=True, exist_ok=True)

    # ---- score every test bag (no image loading) ----
    rec = {g: [] for g in range(4)}
    with torch.no_grad():
        for i in te:
            f, lab, _ = ds[i]; f = f.float(); g = int(lab)
            yb = float(base(f)[0].item()); ya = float(a215(f)[0].item())
            pb = int(np.clip(round(yb), 0, 3)); pa = int(np.clip(round(ya), 0, 3))
            _, wb, _ = base(f, return_attention=True); _, wa, _ = a215(f, return_attention=True)
            eb, ea = entropy(wb.flatten().numpy()), entropy(wa.flatten().numpy())
            fix = (pb != g and pa == g); brk = (pb == g and pa != g)
            score = 100 * fix - 100 * brk + 10 * ((abs(yb - g)) - (abs(ya - g))) + (eb - ea)
            rec[g].append(dict(i=i, g=g, yb=yb, ya=ya, pb=pb, pa=pa, n=f.shape[0], score=score, fix=fix, brk=brk))
    ranked = {g: [r for r in sorted(rec[g], key=lambda r: r["score"], reverse=True) if r["n"] >= 18][:N_GRIDS]
              for g in range(4)}
    for g in range(4):
        print(f"G{g}: top picks -> " + ", ".join(f"idx{r['i']}({'fix' if r['fix'] else 'brk' if r['brk'] else 'eq'})" for r in ranked[g]))

    def panels(r):
        bag = torch.load(ds.samples[r["i"]][0], map_location="cpu", weights_only=False)
        feats = bag["feats"].float()
        with torch.no_grad():
            _, wb, _ = base(feats, return_attention=True); _, wa, _ = a215(feats, return_attention=True)
        wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
        canvas, rc, H, W = stitch(bag)
        vmax = float(max(wb.max(), wa.max()))
        return (pad_square(canvas),
                pad_square(overlay(canvas, weight_map(rc, wb, H, W), vmax)),
                pad_square(overlay(canvas, weight_map(rc, wa, H, W), vmax)))

    for k in range(START, N_GRIDS):
        fig, ax = plt.subplots(4, 3, figsize=(11, 14.6))
        cols = ["Reticulin ROI", "Baseline (ABMIL)", "ASGAP (1.5-entmax)"]
        for c in range(3): ax[0, c].set_title(cols[c], fontsize=16, pad=13)
        for g in range(4):
            r = ranked[g][k % len(ranked[g])]
            roi, ob, oa = panels(r)
            for c, im in enumerate([roi, ob, oa]):
                ax[g, c].imshow(np.clip(im, 0, 1)); ax[g, c].set_xticks([]); ax[g, c].set_yticks([])
                for s in ax[g, c].spines.values(): s.set_edgecolor("0.7")
            ax[g, 0].set_ylabel(f"Grade {g}", fontsize=16, labelpad=10)
        fig.subplots_adjust(left=0.05, right=0.88, top=0.912, bottom=0.03, wspace=0.04, hspace=0.06)
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=CMAP); sm.set_array([])
        cax = fig.add_axes([0.90, 0.30, 0.018, 0.40]); cb = fig.colorbar(sm, cax=cax)
        cb.set_label("attention weight (normalised per ROI)", fontsize=13)
        cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["low", "mid", "high"])
        fig.suptitle("Attention overlays on reticulin ROIs — Baseline vs. ASGAP",
                     fontsize=17, y=0.975)
        fp = out / f"grid_{k+1:02d}.png"; fig.savefig(fp, dpi=170); plt.close(fig)
        print(f"saved {fp.name}")
    print(f"\nALL -> {out}")


if __name__ == "__main__":
    main()
