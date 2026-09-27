"""Visualize a158 (diffuse-temperature gated attention) attention on example ROIs.

Self-contained: uses the precomputed features (.pt) + the patch PNGs referenced in each bag — no
backbone re-extraction needed. For one example test ROI per grade it saves a figure with:
  (a) the attention heatmap laid out on the patch (row,col) grid,
  (b) the top-k highest-attention patch images with their weights,
  (c) title: true grade, a158 predicted grade, max-attention & attention entropy (how peaked).

Usage:
  PYTHONPATH=src python src/tools/visualize_a158_attention.py \
      --checkpoint experiments/.../best_*.pth --backbone virchow2 --seed 2
"""
from __future__ import annotations
import argparse, glob, os, sys, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.novelty_attempts.a158_diffuse_temperature_gated import Model


def load_model(ckpt_path, input_dim, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    m = Model(input_dim=input_dim, num_classes=1)
    m.load_state_dict(sd)
    m.eval().to(device)
    T = torch.nn.functional.softplus(m.T_raw).item()
    return m, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", default="virchow2")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--per_grade", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default="results/a158_attention_viz")
    args = ap.parse_args()

    device = "cpu"
    root = Path(__file__).resolve().parents[2]
    feats_dir = root / "data" / BACKBONE_CONFIG[args.backbone]["feature_dir"]
    ds = GradingBagDatasetFull(feats_dir)
    tr, va, te = patient_split(ds, seed=args.seed)
    idx = {"train": tr, "val": va, "test": te}[args.split]
    model, T = load_model(args.checkpoint, BACKBONE_CONFIG[args.backbone]["dim"], device)
    print(f"a158 {args.backbone} | learned T = {T:.3f} (T<=~1 => selective/sharp, not diffuse)")

    # pick `per_grade` example bags per grade, EVENLY SPACED across the split (variety, not just first-N)
    allg = {g: [] for g in range(4)}
    for i in idx:
        allg[int(ds.samples[i][1])].append(i)
    by_grade = {}
    for g in range(4):
        lst = allg[g]
        if len(lst) <= args.per_grade:
            by_grade[g] = lst
        else:
            step = len(lst) / args.per_grade
            by_grade[g] = [lst[int(k * step)] for k in range(args.per_grade)]
    os.makedirs(root / args.out, exist_ok=True)

    for g in range(4):
        for n, i in enumerate(by_grade[g]):
            pt_path, label = ds.samples[i]
            bag = torch.load(pt_path, map_location=device, weights_only=False)
            feats = bag["feats"].float()
            rc = bag["rc"]
            paths = bag["patch_paths"]
            with torch.no_grad():
                y, attn, _ = model(feats, return_attention=True)
            attn = attn.numpy()
            pred = int(max(0, min(3, round(float(y.item())))))
            ent = float(-(attn * np.log(attn + 1e-12)).sum())
            ent_norm = ent / math.log(len(attn))  # 1=uniform(diffuse), 0=one-hot(peaked)

            rc_np = rc.numpy() if hasattr(rc, "numpy") else np.array(rc)
            R, C = int(rc_np[:, 0].max()) + 1, int(rc_np[:, 1].max()) + 1
            heat = np.full((R, C), np.nan)
            for (r, c), a in zip(rc_np, attn):
                heat[int(r), int(c)] = a

            order = np.argsort(-attn)[:args.topk]
            ncols_p = max(1, math.ceil(args.topk / 2))   # patches laid out in 2 rows
            fig = plt.figure(figsize=(4 + ncols_p * 1.6, 4.2))
            gs = fig.add_gridspec(2, 2 + ncols_p)
            axh = fig.add_subplot(gs[:, 0:2])
            im = axh.imshow(heat, cmap="inferno")
            axh.set_title(f"attention heatmap\n(grid {R}x{C}, N={len(attn)})", fontsize=9)
            axh.set_xticks([]); axh.set_yticks([])
            fig.colorbar(im, ax=axh, fraction=0.046)
            for j, pi in enumerate(order):
                axp = fig.add_subplot(gs[j // ncols_p, 2 + (j % ncols_p)])
                pp = paths[pi]
                pp_full = pp if os.path.isabs(str(pp)) else str(root / str(pp))
                try:
                    img = Image.open(pp_full).convert("RGB")
                    axp.imshow(img)
                except Exception:
                    axp.text(0.5, 0.5, "patch\nmissing", ha="center", va="center", fontsize=7)
                axp.set_title(f"w={attn[pi]:.3f}", fontsize=7)
                axp.set_xticks([]); axp.set_yticks([])
            fig.suptitle(
                f"a158 ({args.backbone}, {args.split}) — true G{label}  pred G{pred}  | "
                f"T={T:.2f}  max_attn={attn.max():.3f}  attn_entropy(norm)={ent_norm:.2f} "
                f"({'diffuse' if ent_norm>0.8 else 'selective'})",
                fontsize=10)
            out_path = root / args.out / f"a158_{args.backbone}_{args.split}_G{label}_{n}.png"
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_path.name}  true=G{label} pred=G{pred} max_attn={attn.max():.3f} "
                  f"top-{args.topk} mass={attn[order].sum():.2f} entropy_norm={ent_norm:.2f}")


if __name__ == "__main__":
    main()
