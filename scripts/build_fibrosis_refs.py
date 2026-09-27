"""Build per-grade reference quantile vectors R_g from seed=2 TRAIN patches ONLY.

Used by `src/models/novelty_attempts/a74_fibrosis_dist_match.py`
(Hn_fibrosis_dist_match). The fibrosis grade of a bag is inferred by matching
the bag's per-patch score distribution (along the prototype fibrosis axis) to a
per-grade REFERENCE distribution. Those references are defined OFFLINE here so
the novelty module loads frozen [4, Q] reference quantiles at __init__ and never
touches training-set labels — or any val/test patch — at training time.

*** LEAKAGE RULE (critical) ***
The references R_g are computed from the seed=2 TRAIN split ONLY:
  1. Load the full dataset the SAME way the trainer does
     (GradingBagDatasetFull over data/features_virchow2_reti).
  2. Call the LOCKED patient_split(full_dataset, seed=2) to get
     (train_idx, val_idx, test_idx).
  3. Use ONLY train_idx bags. val_idx and test_idx patches are NEVER read,
     NEVER projected, and NEVER contribute to any R_g — asserted below.
  4. Project every TRAIN patch onto the fibrosis axis v = blob['axis'] (itself
     built from seed=2 TRAIN patients by compute_grade_prototypes.py).
  5. Per grade g, take fixed empirical quantiles R_g[q] at levels p_1..p_Q.
  6. Save {'levels':..., 'R': {0:..,1:..,2:..,3:..}, 'seed':2, ...} to
     data/fibrosis_refs_seed2.pt.

The saved file is seed=2-SPECIFIC by construction (the split, the axis, and the
'seed' field all reference seed=2). Re-run with --seed S to build a different
split's references; the model's default refs_path points at seed=2.

Output: `data/fibrosis_refs_seed{seed}.pt`, a dict
    {
        "levels": tuple(float) quantile levels p_1..p_Q,
        "R": {0:[Q], 1:[Q], 2:[Q], 3:[Q]} per-grade reference quantiles,
        "R_mean": {0:float,..,3:float} per-grade mean projection (a75 ablation),
        "axis_path": str, "seed": int, "backbone": str, "features_dir": str,
        "train_bags": int, "patch_counts": {g:int}, "split": "train-only",
    }

Usage:
    python scripts/build_fibrosis_refs.py        # default: virchow2, seed=2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make `src/` importable so we reuse the LOCKED patient_split + dataset.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from train_grading_reti import patient_split  # noqa: E402

# MUST match a74_fibrosis_dist_match._DEFAULT_LEVELS exactly.
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _load_axis(backbone: str, seed: int, input_dim: int) -> torch.Tensor:
    """Unit fibrosis axis from the seed-train prototype cache (TRAIN-only)."""
    proto_path = ROOT / "data" / f"prototypes_{backbone}_reti_train_seed{seed}.pt"
    assert proto_path.is_file(), (
        f"Prototype cache not found: {proto_path}\n"
        f"Run `python scripts/compute_grade_prototypes.py --seed {seed}` first."
    )
    blob = torch.load(proto_path, map_location="cpu", weights_only=False)
    axis = blob["axis"].float()
    axis = axis / axis.norm().clamp(min=1e-8)
    assert axis.shape[0] == input_dim, f"axis dim {axis.shape[0]} != {input_dim}"
    return axis, str(proto_path)


def build(backbone: str = "virchow2", seed: int = 2) -> Path:
    features_dir = ROOT / "data" / f"features_{backbone}_reti"
    assert features_dir.is_dir(), f"missing features dir: {features_dir}"

    ds = GradingBagDatasetFull(features_dir)
    # LOCKED split — identical to the trainer. We take ONLY train_idx.
    train_idx, val_idx, test_idx = patient_split(ds, seed=seed)
    train_set = set(train_idx)

    # ---- LEAKAGE GUARD: assert no val/test index enters the reference build ----
    forbidden = set(val_idx) | set(test_idx)
    assert train_set.isdisjoint(forbidden), "train_idx overlaps val/test — split bug!"
    print(f"[build_fibrosis_refs] features_dir={features_dir}")
    print(
        f"[build_fibrosis_refs] split seed={seed}  train={len(train_idx)}  "
        f"val={len(val_idx)}  test={len(test_idx)}  (using TRAIN ONLY)"
    )

    # Infer input_dim from the first train bag, then load the matching axis.
    feats0, _, _ = ds[train_idx[0]]
    input_dim = int(feats0.shape[1])
    axis, axis_path = _load_axis(backbone, seed, input_dim)

    # Collect every TRAIN patch projection per grade.
    scores_per_grade: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: [], 3: []}
    patch_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    for i, idx in enumerate(train_idx):
        # LEAKAGE GUARD (per-iteration): only train indices are ever indexed.
        assert idx in train_set and idx not in forbidden, "non-train idx leaked!"
        feats, label, _ = ds[idx]
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)
        s = feats.float() @ axis                  # [N] per-patch fibrosis projection
        scores_per_grade[label].append(s)
        patch_counts[label] += int(s.numel())
        if (i + 1) % 50 == 0 or i + 1 == len(train_idx):
            print(f"  projected {i+1}/{len(train_idx)} TRAIN bags")

    levels_t = torch.tensor(LEVELS, dtype=torch.float32)
    R: dict[int, torch.Tensor] = {}
    R_mean: dict[int, float] = {}
    for g in range(4):
        assert scores_per_grade[g], f"no TRAIN patches for grade G{g} at seed={seed}"
        all_s = torch.cat(scores_per_grade[g])    # [P_g] all train patches of grade g
        # Fixed empirical quantiles at the model's levels (the reference dist).
        R[g] = torch.quantile(all_s, levels_t).float()        # [Q]
        R_mean[g] = float(all_s.mean())
        qstr = ", ".join(f"{v:+.3f}" for v in R[g].tolist())
        print(
            f"  G{g}: {patch_counts[g]:6d} patches | mean={R_mean[g]:+.3f} | "
            f"R=[{qstr}]"
        )

    out_path = ROOT / "data" / f"fibrosis_refs_seed{seed}.pt"
    torch.save(
        {
            "levels": tuple(float(x) for x in LEVELS),
            "R": R,
            "R_mean": R_mean,
            "axis_path": axis_path,
            "seed": seed,
            "backbone": backbone,
            "features_dir": str(features_dir),
            "train_bags": len(train_idx),
            "patch_counts": patch_counts,
            "split": "train-only",  # val/test patches NEVER used
        },
        out_path,
    )
    print(f"[build_fibrosis_refs] saved -> {out_path}  (TRAIN-only, seed={seed})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="virchow2")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    build(args.backbone, args.seed)


if __name__ == "__main__":
    main()
