"""More a215-vs-ABMIL-baseline visualizations (a215 only; no other novelty).

Produces 5 extra figures on virchow2 test, all comparing ONLY a215 against the simple-gate baseline:
  1. entmax_family.png      - what the learnable alpha does: entmax of one real energy vector at
                              alpha in {1.0,1.25,1.5,1.75,2.0}, learned alpha highlighted (mechanism view).
  2. per_grade_breakdown.png- entropy (base vs a215) per grade; % patches a215 zeroes per grade;
                              bag-size vs % zeroed scatter (does entmax drop more on big/which-grade bags?).
  3. diff_heatmap_grid.png  - per grade [ROI | baseline | a215 | (a215 - baseline) weight diff] heatmaps,
                              so you SEE where a215 moves weight relative to the baseline.
  4. outcome_compare.png    - the actual grading effect: test confusion matrices (base vs a215),
                              per-class recall bars, continuous prediction scatter vs true grade.
  5. top_patch_gallery.png  - the literal top-weighted patch images baseline vs a215 spotlight (fibrotic grades).
Analysis/reporting only (a215 is an already-characterized failed candidate); CPU.
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
from models.novelty_attempts.a215_adaptive_sparse_gated_attention_pooling import Model as A215, entmax_bisect

BB = "virchow2"; DIM = BACKBONE_CONFIG[BB]["dim"]; P = 224; S = 112
ZERO = 1e-6
GCOL = ["#4477aa", "#66ccee", "#ee6677", "#aa3377"]


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
        canvas[r * S:r * S + P, c * S:c * S + P] = np.asarray(Image.open(fp).convert("RGB"), np.float32) / 255.0
    return canvas, rc, R, C, H, W


def heat_grid(rc, w):
    rc = rc.numpy() if hasattr(rc, "numpy") else np.array(rc)
    R, C = int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1
    h = np.full((R, C), np.nan)
    for (r, c), a in zip(rc, w): h[int(r), int(c)] = a
    return h, R, C


def sq(arr, SQ=320):
    a = np.clip(arr, 0, 1)
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((SQ, SQ), Image.BILINEAR), np.float32) / 255.0


def predict(model, ds, idxs):
    ys, ts = [], []
    with torch.no_grad():
        for i in idxs:
            feat, lab, _ = ds[i]
            y = float(model(feat.float())[0].item())
            ys.append(y); ts.append(int(lab))
    return np.array(ys), np.array(ts)


def main():
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[BB]["feature_dir"])
    te = patient_split(ds, seed=2)[2]
    base = load(SimpleGatedMIL(input_dim=DIM, num_classes=1, topk=0),
                "experiments/20260603_ablation/patch/*simple_virchow2_regression*/best_*.pth")
    a215 = load(A215(input_dim=DIM, num_classes=1), "experiments/a215_ablation/regression_seed2/casc_a215_virchow2_s2_*/best_*.pth")
    alpha = float((1.0 + torch.sigmoid(a215.alpha_raw)).item())
    out = ROOT / "results" / "a215_vs_baseline"; out.mkdir(parents=True, exist_ok=True)
    print(f"a215 alpha={alpha:.4f}")

    labels = np.array([int(ds.samples[i][1]) for i in te])
    by_grade = {g: [te[k] for k in range(len(te)) if labels[k] == g] for g in range(4)}

    # ---- gather per-bag weights/stats once ----
    Wb, Wa, RC, ENb, ENa, FR0, NPB = {}, {}, {}, [], [], [], []
    with torch.no_grad():
        for i in te:
            feat, _, _ = ds[i]; feat = feat.float()
            _, wb, _ = base(feat, return_attention=True)
            _, wa, _ = a215(feat, return_attention=True)
            wb = wb.flatten().numpy(); wa = wa.flatten().numpy()
            Wb[i], Wa[i] = wb, wa
            ENb.append(entropy(wb)); ENa.append(entropy(wa)); FR0.append((wa < ZERO).mean()); NPB.append(len(wa))
    ENb, ENa, FR0, NPB = map(np.array, (ENb, ENa, FR0, NPB))

    # ============ FIG 1: entmax family (mechanism) ============
    pick = by_grade[3][len(by_grade[3]) // 2]
    feat, _, _ = ds[pick]; feat = feat.float()
    with torch.no_grad():
        h = a215.bottleneck(feat)
        e = a215.attention_W(a215.attention_V(h) * a215.attention_U(h)).squeeze(-1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    alphas = [1.0, 1.25, 1.5, 1.75, 2.0]
    for al in alphas:
        if al <= 1.0: w = F.softmax(e, dim=0)
        else: w = entmax_bisect(e, torch.tensor(al))
        w = w.detach().numpy(); nz = int((w > ZERO).sum())
        lbl = f"α={al:.2f}" + (" softmax" if al == 1 else " sparsemax" if al == 2 else "") + f"  (nonzero={nz}/{len(w)})"
        lw = 3.2 if abs(al - alpha) < 0.13 else 1.6
        ax[0].plot(np.sort(w)[::-1], lw=lw, label=lbl)
    ax[0].set_title(f"entmax of ONE real energy vector at varying α\n(bold = α = 1.5 (fixed)); N={len(e)} patches, G3 bag")
    ax[0].set_xlabel("patch rank"); ax[0].set_ylabel("attention weight"); ax[0].legend(fontsize=8)
    # nonzero support size vs alpha across all test bags
    sweep = np.linspace(1.01, 2.0, 12); supp = []
    with torch.no_grad():
        for al in sweep:
            frac = []
            for i in te:
                feat, _, _ = ds[i]; feat = feat.float()
                hh = a215.bottleneck(feat)
                ee = a215.attention_W(a215.attention_V(hh) * a215.attention_U(hh)).squeeze(-1)
                w = entmax_bisect(ee, torch.tensor(float(al))).numpy()
                frac.append((w > ZERO).mean())
            supp.append(np.mean(frac))
    ax[1].plot(sweep, np.array(supp) * 100, "-o", color="indianred")
    ax[1].axvline(alpha, color="k", ls="--", label="α = 1.5 (fixed)")
    ax[1].set_xlabel("entmax α"); ax[1].set_ylabel("% patches kept (nonzero), test mean")
    ax[1].set_title("Sparsity vs α (mean over 259 test bags)\nα=1: 100% kept → α=2: hard-sparse"); ax[1].legend()
    fig.suptitle("ASGAP mechanism: α-entmax with α = 1.5 (fixed)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out / "entmax_family.png", dpi=130); plt.close(fig)
    print("saved entmax_family.png")

    # ============ FIG 2: per-grade breakdown ============
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    gi = [np.array([k for k in range(len(te)) if labels[k] == g]) for g in range(4)]
    bp1 = ax[0].boxplot([ENb[gi[g]] for g in range(4)], positions=np.arange(4) - 0.17, widths=0.3, patch_artist=True)
    bp2 = ax[0].boxplot([ENa[gi[g]] for g in range(4)], positions=np.arange(4) + 0.17, widths=0.3, patch_artist=True)
    for b in bp1["boxes"]: b.set_facecolor("#bbbbbb")
    for b in bp2["boxes"]: b.set_facecolor("#ee6677")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f"G{g}" for g in range(4)])
    ax[0].set_ylabel("attention entropy (1=diffuse)")
    ax[0].set_title("Diffuseness per grade\ngray=baseline, red=ASGAP (ASGAP always sharper)")
    ax[0].legend([bp1["boxes"][0], bp2["boxes"][0]], ["baseline", "ASGAP"], loc="lower left")
    ax[1].bar(range(4), [FR0[gi[g]].mean() * 100 for g in range(4)], color=GCOL)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([f"G{g}" for g in range(4)])
    ax[1].set_ylabel("% patches zeroed (mean)"); ax[1].set_title("How much ASGAP sparsifies, per grade")
    for g in range(4):
        ax[2].scatter(NPB[gi[g]], FR0[gi[g]] * 100, s=18, alpha=0.6, color=GCOL[g], label=f"G{g}")
    ax[2].set_xlabel("bag size (N patches)"); ax[2].set_ylabel("% patches zeroed")
    ax[2].set_title("Sparsity vs bag size"); ax[2].legend()
    fig.suptitle("ASGAP per-grade sparsity breakdown", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out / "per_grade_breakdown.png", dpi=130); plt.close(fig)
    print("saved per_grade_breakdown.png")

    # ============ FIG 3: difference heatmap grid ============
    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    cols = ["ROI (original)", "baseline weight", "ASGAP weight", "ASGAP − baseline"]
    for c in range(4): ax[0, c].set_title(cols[c], fontsize=12)
    for g in range(4):
        i = by_grade[g][len(by_grade[g]) // 2]
        bag = torch.load(ds.samples[i][0], map_location="cpu", weights_only=False)
        wb, wa = Wb[i], Wa[i]
        canvas, rc, R, C, H, W = stitch(bag)
        hb, _, _ = heat_grid(rc, wb); ha, _, _ = heat_grid(rc, wa); diff = ha - hb
        vmax = np.nanmax([np.nanmax(hb), np.nanmax(ha)]); dmax = np.nanmax(np.abs(diff))
        ax[g, 0].imshow(sq(canvas))
        ax[g, 1].imshow(hb, cmap="inferno", vmin=0, vmax=vmax)
        ax[g, 2].imshow(ha, cmap="inferno", vmin=0, vmax=vmax)
        im = ax[g, 3].imshow(diff, cmap="bwr", vmin=-dmax, vmax=dmax)
        fig.colorbar(im, ax=ax[g, 3], fraction=0.046)
        for c in range(4): ax[g, c].set_xticks([]); ax[g, c].set_yticks([])
        ax[g, 0].set_ylabel(f"G{int(ds.samples[i][1])}\nN={len(wb)}", fontsize=12, rotation=0, labelpad=30, va="center")
    fig.suptitle("Where ASGAP moves attention vs baseline\nred = ASGAP emphasizes more, blue = baseline emphasizes more", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out / "diff_heatmap_grid.png", dpi=120); plt.close(fig)
    print("saved diff_heatmap_grid.png")

    # ============ FIG 4: grading outcome ============
    yb, tb = predict(base, ds, te); ya, ta = predict(a215, ds, te)
    pb = np.clip(np.round(yb), 0, 3).astype(int); pa = np.clip(np.round(ya), 0, 3).astype(int)
    def cm4(p, t):
        m = np.zeros((4, 4), int)
        for pp, tt in zip(p, t): m[tt, pp] += 1
        return m
    Cb, Ca = cm4(pb, tb), cm4(pa, ta)
    recb = np.array([Cb[g, g] / max(1, Cb[g].sum()) for g in range(4)])
    reca = np.array([Ca[g, g] / max(1, Ca[g].sum()) for g in range(4)])
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    for a, Cm, ttl in [(ax[0, 0], Cb, "baseline"), (ax[0, 1], Ca, "ASGAP")]:
        im = a.imshow(Cm, cmap="Blues"); a.set_title(f"{ttl} confusion (test)")
        a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xlabel("pred"); a.set_ylabel("true")
        a.set_xticklabels([f"G{g}" for g in range(4)]); a.set_yticklabels([f"G{g}" for g in range(4)])
        for r in range(4):
            for c in range(4): a.text(c, r, Cm[r, c], ha="center", va="center", color="k", fontsize=11)
    x = np.arange(4)
    ax[1, 0].bar(x - 0.2, recb * 100, 0.4, label="baseline", color="#888888")
    ax[1, 0].bar(x + 0.2, reca * 100, 0.4, label="ASGAP", color="#ee6677")
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels([f"G{g}" for g in range(4)])
    ax[1, 0].set_ylabel("recall %"); ax[1, 0].set_title("Per-class recall: baseline vs ASGAP"); ax[1, 0].legend()
    for g in range(4):
        ax[1, 1].scatter(tb[tb == g] + np.linspace(-0.12, 0.12, (tb == g).sum()), yb[tb == g], s=14, alpha=0.4, color="#888888")
        ax[1, 1].scatter(ta[ta == g] + np.linspace(-0.12, 0.12, (ta == g).sum()), ya[ta == g], s=14, alpha=0.5, color="#ee6677")
    ax[1, 1].plot([-.5, 3.5], [-.5, 3.5], "k--", lw=1)
    ax[1, 1].set_xlabel("true grade"); ax[1, 1].set_ylabel("continuous prediction ŷ")
    ax[1, 1].set_title("Continuous predictions (gray=baseline, red=ASGAP)")
    fig.suptitle("ASGAP vs baseline — effect on grading outcome", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out / "outcome_compare.png", dpi=125); plt.close(fig)
    print(f"saved outcome_compare.png  base recall {np.round(recb*100,1)}  a215 recall {np.round(reca*100,1)}")

    # ============ FIG 5: top-patch gallery (fibrotic grades) ============
    K = 6
    fig, ax = plt.subplots(4, 2 * K, figsize=(2 * K * 1.5, 4 * 1.7))
    for g in range(4):
        i = by_grade[g][len(by_grade[g]) // 2]
        bag = torch.load(ds.samples[i][0], map_location="cpu", weights_only=False)
        pp = [p if os.path.isabs(str(p)) else str(ROOT / str(p)) for p in bag["patch_paths"]]
        for col, (w, tag) in enumerate([(Wb[i], "base"), (Wa[i], "ASGAP")]):
            top = np.argsort(w)[::-1][:K]
            for k, idx in enumerate(top):
                axx = ax[g, col * K + k]
                if os.path.exists(pp[idx]):
                    axx.imshow(np.asarray(Image.open(pp[idx]).convert("RGB")))
                axx.set_xticks([]); axx.set_yticks([])
                axx.set_title(f"{w[idx]:.3f}", fontsize=7)
                if k == 0: axx.set_ylabel(f"G{g}\n{tag}", fontsize=9, rotation=0, labelpad=20, va="center")
    fig.suptitle("Top-6 spotlighted patches — baseline (left 6) vs ASGAP (right 6) per grade\nnumbers = attention weight", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out / "top_patch_gallery.png", dpi=115); plt.close(fig)
    print("saved top_patch_gallery.png")


if __name__ == "__main__":
    main()
