#!/usr/bin/env python3
"""Premise diagnostic: does Virchow2 patch feature-norm ||h|| track fibrosis
grade, or is it grade-irrelevant (consistent with the advisor claim
"high-norm patches = dark bone, which grading avoids")?

This grounds the decision to rule out a45 (rank-of-||h|| salience). If high
norm turns out to track grade (e.g. dense fibrosis is also high-norm), the
"avoid norm weighting" rule would be WRONG and must flip.

Read-only: no training, no model checkpoint, no git/test writes. Loads only
the frozen Virchow2 feature bags + their source patch PNGs. Outputs go to
results/diag/.

Run:  python scripts/diag_norm_vs_grade.py
"""
from pathlib import Path
import torch
import numpy as np
from scipy.stats import spearmanr
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features_virchow2_reti"
OUT = ROOT / "results" / "diag"
OUT.mkdir(parents=True, exist_ok=True)
GRADE_MAP = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}


def grade_of(folder_name: str):
    for g, lbl in GRADE_MAP.items():
        if g in folder_name:
            return lbl
    return None


def load_bags():
    bags = []
    for pt in sorted(FEAT.rglob("*.pt")):
        g = grade_of(pt.parent.name)
        if g is None:
            continue
        d = torch.load(pt, map_location="cpu", weights_only=False)
        feats = (d["feats"] if isinstance(d, dict) else d).float()
        bags.append(dict(
            path=pt, patient=pt.parent.name, grade=g, n=feats.shape[0],
            feats=feats, norms=feats.norm(dim=1).numpy(),
            patch_paths=d.get("patch_paths") if isinstance(d, dict) else None,
            rc=d.get("rc") if isinstance(d, dict) else None,
        ))
    return bags


def montage(items, title, save_path, cols=8):
    """items: list of (png_path, norm_value). Save a labelled grid PNG."""
    items = [(p, v) for p, v in items if p and Path(p).exists()]
    if not items:
        return None
    rows = (len(items) + cols - 1) // cols
    cell = 224
    pad = 22
    W, H = cols * cell, rows * (cell + pad)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (p, v) in enumerate(items):
        r, c = divmod(i, cols)
        im = Image.open(p).convert("RGB").resize((cell, cell))
        y = r * (cell + pad)
        canvas.paste(im, (c * cell, y + pad))
        draw.text((c * cell + 3, y + 5), f"||h||={v:.2f}", fill=(0, 0, 0))
    draw.text((3, 0), title, fill=(180, 0, 0))
    canvas.save(save_path)
    return save_path


def spatial_heatmap(bag, save_path):
    rc = bag["rc"]
    if rc is None:
        return None
    rc = rc.numpy() if hasattr(rc, "numpy") else np.asarray(rc)
    R, C = rc[:, 0].max() + 1, rc[:, 1].max() + 1
    grid = np.full((R, C), np.nan)
    for (r, c), nv in zip(rc, bag["norms"]):
        grid[r, c] = nv
    plt.figure(figsize=(C * 0.4 + 1, R * 0.4 + 1))
    plt.imshow(grid, cmap="magma")
    plt.colorbar(label="||h||")
    plt.title(f"{bag['patient']} / {bag['path'].stem}  (G{bag['grade']})  norm map")
    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close()
    return save_path


def main():
    bags = load_bags()
    grades = np.array([b["grade"] for b in bags])
    print(f"[load] {len(bags)} bags | grade counts: "
          + ", ".join(f"G{g}:{int((grades==g).sum())}" for g in range(4)))

    # ---- Part 1: per-bag norm statistics vs grade ----
    stats = {
        "mean_norm": np.array([b["norms"].mean() for b in bags]),
        "max_norm":  np.array([b["norms"].max() for b in bags]),
        "p90_norm":  np.array([np.percentile(b["norms"], 90) for b in bags]),
        "std_norm":  np.array([b["norms"].std() for b in bags]),
        "cv_norm":   np.array([b["norms"].std() / b["norms"].mean() for b in bags]),
    }
    lines = ["# Diagnostic: feature-norm ||h|| vs fibrosis grade (Virchow2)\n",
             f"Bags: {len(bags)} | "
             + ", ".join(f"G{g}:{int((grades==g).sum())}" for g in range(4)) + "\n",
             "## Part 1 — does per-bag norm track grade?\n",
             "| stat | Spearman r vs grade | p | G0 mean | G1 | G2 | G3 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, arr in stats.items():
        r, p = spearmanr(arr, grades)
        per = [arr[grades == g].mean() for g in range(4)]
        lines.append(f"| {name} | {r:+.3f} | {p:.1e} | "
                     + " | ".join(f"{x:.3f}" for x in per) + " |")
        print(f"[part1] {name:10s} spearman={r:+.3f} (p={p:.1e}) "
              + "per-grade=" + " ".join(f"{x:.2f}" for x in per))

    # within-bag spread (does norm-weighting even do much?)
    cv = stats["cv_norm"]
    lines += ["",
              f"Within-bag norm spread (cv = std/mean): "
              f"median {np.median(cv):.4f}, p90 {np.percentile(cv,90):.4f}. "
              f"Small cv => norm-weighting barely re-ranks patches.\n"]
    print(f"[part1] within-bag cv: median={np.median(cv):.4f} p90={np.percentile(cv,90):.4f}")

    # ---- Part 2: what do high/low-norm patches look like? ----
    lines.append("## Part 2 — visual: highest vs lowest-norm patches\n")
    for g in (0, 1, 2, 3):
        cand = [b for b in bags if b["grade"] == g and b["patch_paths"]]
        if not cand:
            continue
        bag = max(cand, key=lambda b: b["n"])
        order = np.argsort(bag["norms"])
        pp = bag["patch_paths"]
        lo = [(pp[i], bag["norms"][i]) for i in order[:8]]
        hi = [(pp[i], bag["norms"][i]) for i in order[::-1][:8]]
        tag = f"G{g}_{bag['patient'].replace(' ','_')}_{bag['path'].stem}"
        montage(hi, f"G{g} HIGHEST-norm patches  ({bag['patient']}/{bag['path'].stem}, N={bag['n']})",
                OUT / f"hi_{tag}.png")
        montage(lo, f"G{g} LOWEST-norm patches  ({bag['patient']}/{bag['path'].stem}, N={bag['n']})",
                OUT / f"lo_{tag}.png")
        spatial_heatmap(bag, OUT / f"map_{tag}.png")
        lines.append(f"- G{g}: `hi_{tag}.png` / `lo_{tag}.png` / `map_{tag}.png` "
                     f"(norm range {bag['norms'].min():.2f}-{bag['norms'].max():.2f})")
        print(f"[part2] G{g}: montages for {bag['patient']}/{bag['path'].stem} "
              f"(N={bag['n']}, norm {bag['norms'].min():.2f}-{bag['norms'].max():.2f})")

    # ---- Part 3: does a diffuse fibrosis-direction beat raw norm? (patient-disjoint) ----
    patients = sorted(set(b["patient"] for b in bags))
    A = set(patients[::2])  # fit group (even-indexed patients)
    fit = [b for b in bags if b["patient"] in A]
    evl = [b for b in bags if b["patient"] not in A]

    def mean_patch_feat(sub):
        s = torch.zeros(1280); c = 0
        for b in sub:
            s += b["feats"].sum(0); c += b["n"]
        return s / max(c, 1)

    v = mean_patch_feat([b for b in fit if b["grade"] >= 2]) - \
        mean_patch_feat([b for b in fit if b["grade"] <= 1])
    v = v / v.norm()
    gE = np.array([b["grade"] for b in evl])
    proj_mean = np.array([float((b["feats"] @ v).mean()) for b in evl])
    norm_mean = np.array([b["norms"].mean() for b in evl])
    # coverage: fraction of patches whose projection exceeds the global-fit median
    thr = float(np.median(np.concatenate([(b["feats"] @ v).numpy() for b in fit])))
    coverage = np.array([float(((b["feats"] @ v).numpy() > thr).mean()) for b in evl])

    r_proj, _ = spearmanr(proj_mean, gE)
    r_cov, _ = spearmanr(coverage, gE)
    r_norm, _ = spearmanr(norm_mean, gE)
    lines += ["", "## Part 3 — diffuse fibrosis-direction vs raw norm (held-out patients)\n",
              "Direction v fit on even-indexed patients; evaluated on the rest "
              "(patient-disjoint). Supervised illustration, not a model.\n",
              f"- Spearman(mean ||h||, grade)            = {r_norm:+.3f}",
              f"- Spearman(mean <h,v_fibrosis>, grade)   = {r_proj:+.3f}",
              f"- Spearman(coverage above thr, grade)    = {r_cov:+.3f}\n"]
    print(f"[part3] held-out spearman: norm={r_norm:+.3f}  proj={r_proj:+.3f}  coverage={r_cov:+.3f}")

    (OUT / "norm_vs_grade.md").write_text("\n".join(lines))
    print(f"[done] wrote {OUT/'norm_vs_grade.md'} and montages to {OUT}/")


if __name__ == "__main__":
    main()
