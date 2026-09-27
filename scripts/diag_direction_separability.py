#!/usr/bin/env python3
"""Pre-flight diagnostic 2: does coverage/projection along a fibrosis
direction separate the *adjacent* grade boundaries (where the val gate is
actually decided — G0|G1 ~50% of errors, G2|G3 ~30%), and does the fibrosis
direction skip bone / high-norm patches?

Read-only. Patient-disjoint: direction(s) fit on even-indexed patients,
evaluated on the rest. Supervised illustration, not a model. Outputs
results/diag/direction_separability.md + a top-projection vs top-norm montage.

Run:  python scripts/diag_direction_separability.py
"""
from pathlib import Path
import torch
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features_virchow2_reti"
OUT = ROOT / "results" / "diag"
OUT.mkdir(parents=True, exist_ok=True)
GMAP = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}


def grade_of(name):
    for g, l in GMAP.items():
        if g in name:
            return l
    return None


def load_bags():
    bags = []
    for pt in sorted(FEAT.rglob("*.pt")):
        g = grade_of(pt.parent.name)
        if g is None:
            continue
        d = torch.load(pt, map_location="cpu", weights_only=False)
        f = (d["feats"] if isinstance(d, dict) else d).float()
        bags.append(dict(patient=pt.parent.name, grade=g, n=f.shape[0], feats=f,
                         norms=f.norm(dim=1).numpy(),
                         patch_paths=d.get("patch_paths") if isinstance(d, dict) else None,
                         path=pt))
    return bags


def mean_patch_feat(sub):
    s = torch.zeros(1280); c = 0
    for b in sub:
        s += b["feats"].sum(0); c += b["n"]
    return s / max(c, 1)


def auc(x, y, lo, hi):
    """P(signal(hi) > signal(lo)) via Mann-Whitney U; 0.5 = random."""
    a, b = x[y == lo], x[y == hi]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    U = mannwhitneyu(b, a, alternative="two-sided").statistic
    return U / (len(a) * len(b))


def montage(items, title, save, cols=8):
    items = [(p, v) for p, v in items if p and Path(p).exists()]
    if not items:
        return
    cell, pad = 224, 22
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * (cell + pad)), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (p, val) in enumerate(items):
        r, c = divmod(i, cols)
        y = r * (cell + pad)
        canvas.paste(Image.open(p).convert("RGB").resize((cell, cell)), (c * cell, y + pad))
        d.text((c * cell + 3, y + 5), f"{val:.2f}", fill=(0, 0, 0))
    d.text((3, 0), title, fill=(180, 0, 0))
    canvas.save(save)


def main():
    bags = load_bags()
    patients = sorted(set(b["patient"] for b in bags))
    A = set(patients[::2])
    fit = [b for b in bags if b["patient"] in A]
    evl = [b for b in bags if b["patient"] not in A]
    print(f"[load] {len(bags)} bags | fit {len(fit)} / eval {len(evl)}")

    # axes (fit on group A)
    v_hilo = mean_patch_feat([b for b in fit if b["grade"] >= 2]) - \
        mean_patch_feat([b for b in fit if b["grade"] <= 1])
    v_hilo = v_hilo / v_hilo.norm()
    ord_axes = {}
    for k in (1, 2, 3):  # boundary below grade k: G>=k vs G<k
        vk = mean_patch_feat([b for b in fit if b["grade"] >= k]) - \
            mean_patch_feat([b for b in fit if b["grade"] < k])
        ord_axes[k] = vk / vk.norm()

    thr_hilo = float(np.median(np.concatenate([(b["feats"] @ v_hilo).numpy() for b in fit])))
    gE = np.array([b["grade"] for b in evl])

    def pm(b, v):
        return float((b["feats"] @ v).mean())

    sig = {
        "mean_norm":  np.array([b["norms"].mean() for b in evl]),
        "proj_hilo":  np.array([pm(b, v_hilo) for b in evl]),
        "cov_hilo":   np.array([float(((b["feats"] @ v_hilo).numpy() > thr_hilo).mean()) for b in evl]),
        "proj_ord1":  np.array([pm(b, ord_axes[1]) for b in evl]),  # G0 | G1+
        "proj_ord2":  np.array([pm(b, ord_axes[2]) for b in evl]),  # G0-1 | G2+
        "proj_ord3":  np.array([pm(b, ord_axes[3]) for b in evl]),  # G0-2 | G3
    }

    lines = ["# Pre-flight 2: direction separability per boundary + bone check\n",
             f"Eval bags (held-out patients): {len(evl)} | "
             + ", ".join(f"G{g}:{int((gE == g).sum())}" for g in range(4)) + "\n",
             "## Per-boundary AUC (separating lower vs upper grade, held-out bags)\n",
             "AUC 0.5 = random, →1 = perfect. The val gate is decided mostly at "
             "**G0|G1** (~50% of errors) and **G2|G3** (~30%).\n",
             "| signal | G0\\|G1 | G1\\|G2 | G2\\|G3 | all (Spearman) |",
             "|---|---:|---:|---:|---:|"]
    for name, arr in sig.items():
        r, _ = spearmanr(arr, gE)
        row = [auc(arr, gE, 0, 1), auc(arr, gE, 1, 2), auc(arr, gE, 2, 3)]
        lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in row) + f" | {r:+.3f} |")
        print(f"[bnd] {name:10s} G0|G1={row[0]:.3f} G1|G2={row[1]:.3f} G2|G3={row[2]:.3f} sp={r:+.3f}")

    # per-patch: norm vs fibrosis projection
    alln = np.concatenate([b["norms"] for b in evl])
    allp = np.concatenate([(b["feats"] @ v_hilo).numpy() for b in evl])
    r_np, _ = spearmanr(alln, allp)
    lines += ["", "## Bone / high-norm vs fibrosis-projection\n",
              f"Per-patch Spearman(‖h‖, ⟨h,v_hilo⟩) over {len(alln)} held-out patches = **{r_np:+.3f}**.",
              "Near 0 ⇒ high-norm (tissue-content / bone) patches are NOT the "
              "high-fibrosis-projection patches ⇒ coverage-along-v is not just a "
              "tissue/bone re-ranking.\n"]
    print(f"[bone] per-patch spearman(norm, proj_hilo) = {r_np:+.3f}")

    # montage: top fibrosis-projection vs top norm, same G3 ROI
    cand = [b for b in bags if b["grade"] == 3 and b["patch_paths"]]
    bag = max(cand, key=lambda b: b["n"])
    proj = (bag["feats"] @ v_hilo).numpy()
    pp = bag["patch_paths"]
    ov = np.argsort(proj)[::-1][:8]
    on = np.argsort(bag["norms"])[::-1][:8]
    montage([(pp[i], proj[i]) for i in ov],
            f"G3 top fibrosis-PROJECTION ({bag['patient']}/{bag['path'].stem})",
            OUT / "topproj_G3.png")
    montage([(pp[i], bag["norms"][i]) for i in on],
            "G3 top NORM (same ROI)", OUT / "topnorm_G3.png")
    overlap = len(set(ov.tolist()) & set(on.tolist()))
    lines.append(f"Top-8 patch overlap (projection vs norm) on G3 ROI "
                 f"{bag['patient']}/{bag['path'].stem}: **{overlap}/8**. "
                 f"Montages: `topproj_G3.png` vs `topnorm_G3.png`.")
    print(f"[montage] top-8 proj-vs-norm overlap: {overlap}/8")

    (OUT / "direction_separability.md").write_text("\n".join(lines))
    print(f"[done] wrote {OUT/'direction_separability.md'}")


if __name__ == "__main__":
    main()
