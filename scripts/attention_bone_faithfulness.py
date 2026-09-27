"""Faithfulness test: does the baseline gated-attention already AVOID bone?

Loads the locked baseline (SimpleGatedMIL + Virchow2 + regression, seed=2),
gets per-patch attention weights on the seed=2 TEST bags, and correlates them
with the per-patch semantic bone & fibrosis scores (data/patch_concept_scores_reti.pt,
TITAN zero-shot concepts, patch-index aligned to the Virchow2 bags).

Within each bag: Spearman(attention, bone) and Spearman(attention, fibrosis).
- If Spearman(attn,bone) << 0  -> baseline already DOWN-weights bone (FAITHFUL).
  => an explicit bone-suppressor cannot add much (QWK tie expected); the result is
     an interpretability/faithfulness contribution.
- If Spearman(attn,bone) ~ 0 or > 0 -> baseline does NOT avoid bone (a GAP).
  => an explicit bone-suppressor may genuinely help AND there is a faithfulness gap.

Read-only. No training.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data.bag_dataset import GradingBagDatasetFull           # noqa: E402
from train_grading_reti import patient_split                 # noqa: E402
from models.simple_mil import SimpleGatedMIL                 # noqa: E402

CKPT = ROOT / "experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342/best_04_reti_simple_virchow2_regression.pth"
SCORES = ROOT / "data" / "patch_concept_scores_reti.pt"
FEATS = ROOT / "data" / "features_virchow2_reti"


def rankdata(x):
    x = np.asarray(x, float); order = x.argsort(); ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    # simple average-tie via groupby on sorted
    sx = x[order]; i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j+1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j+1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = SimpleGatedMIL(input_dim=1280, num_classes=1)
    model.load_state_dict(state); model.eval()
    scores = torch.load(SCORES, map_location="cpu", weights_only=False)

    ds = GradingBagDatasetFull(FEATS)
    _, _, test_idx = patient_split(ds, seed=2)
    print(f"[faithfulness] seed=2 TEST bags: {len(test_idx)}")

    sb, sf = [], []   # per-bag spearman(attn,bone), spearman(attn,fibrosis)
    by_grade = {}
    miss = 0
    for idx in test_idx:
        pt_path, label = ds.samples[idx]
        rel = "/".join(Path(pt_path).parts[-3:])
        if rel not in scores:
            miss += 1; continue
        feats, _, _ = ds[idx]
        feats = feats.float() if torch.is_tensor(feats) else torch.as_tensor(feats).float()
        with torch.no_grad():
            out = model(feats, return_attention=True)
        attn = out[1] if isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]) else None
        if attn is None:
            raise RuntimeError("model did not return attention")
        attn = attn.view(-1).numpy()
        bone = scores[rel]["bone"].numpy(); fib = scores[rel]["fibrosis"].numpy()
        if len(attn) != len(bone):
            miss += 1; continue
        cb, cf = spearman(attn, bone), spearman(attn, fib)
        if not np.isnan(cb): sb.append(cb)
        if not np.isnan(cf): sf.append(cf)
        by_grade.setdefault(label, []).append((cb, cf))

    sb, sf = np.array(sb), np.array(sf)
    print(f"  matched bags: {len(sb)}  (missing/mismatch: {miss})\n")
    print(f"Spearman(attention, BONE)     per-bag: mean {np.nanmean(sb):+.3f}  median {np.nanmedian(sb):+.3f}  "
          f"frac<0 {np.mean(sb<0):.2f}  (n={len(sb)})")
    print(f"Spearman(attention, FIBROSIS) per-bag: mean {np.nanmean(sf):+.3f}  median {np.nanmedian(sf):+.3f}  "
          f"frac>0 {np.mean(sf>0):.2f}  (n={len(sf)})")
    print("\nper grade  mean Spearman(attn,bone) / (attn,fibrosis):")
    for g in sorted(by_grade):
        arr = np.array(by_grade[g], float)
        print(f"  G{g}: bone {np.nanmean(arr[:,0]):+.3f}   fibrosis {np.nanmean(arr[:,1]):+.3f}   (n={len(arr)})")
    print("\nVERDICT:")
    mb = np.nanmean(sb)
    if mb < -0.15:
        print(f"  attn vs bone = {mb:+.3f} (clearly negative) -> baseline ALREADY avoids bone = FAITHFUL.")
        print("  => explicit bone-suppressor unlikely to lift QWK; the contribution is the faithfulness result itself.")
    elif mb > 0.05:
        print(f"  attn vs bone = {mb:+.3f} (positive) -> baseline FOCUSES on bone (faithfulness GAP).")
        print("  => an explicit bone-suppressor has a real chance to help QWK AND fixes a faithfulness gap.")
    else:
        print(f"  attn vs bone = {mb:+.3f} (near zero) -> baseline is INDIFFERENT to bone (mild gap).")
        print("  => bone-suppression is worth a direct trial; modest upside.")


if __name__ == "__main__":
    main()
