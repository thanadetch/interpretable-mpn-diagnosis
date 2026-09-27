"""Compute per-grade Virchow2 patch-feature prototypes from TRAIN patients only.

Used by `src/models/novelty_attempts/a25_fibrosis_axis_projection.py`
(H3_projection_onto_fibrosis_axis). The fibrosis axis is defined offline as

    axis = (c_G3 - c_G0) / ||c_G3 - c_G0||

where `c_g` is the mean of every train-patient Virchow2 patch feature with
grade `g`. By computing it once, offline, the novelty module loads a
frozen [1280] direction at __init__ and never touches training-set
labels at training time. This is consistent with the patient_split
contract (only train patients are used to derive the axis).

Output: `data/prototypes_virchow2_reti_train_seed{seed}.pt`, a dict
    {
        "axis": [1280] unit vector (c_G3 - c_G0 normalised),
        "prototypes": {0: [1280], 1: [1280], 2: [1280], 3: [1280]},
        "patch_counts": {0: int, 1: int, 2: int, 3: int},
        "patient_count": {0: int, 1: int, 2: int, 3: int},
        "seed": int,
        "backbone": "virchow2",
        "features_dir": str,
    }

Usage:
    python scripts/compute_grade_prototypes.py
        # default: backbone=virchow2, seed=2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make `src/` importable so we reuse the locked patient_split.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from train_grading_reti import patient_split  # noqa: E402


def compute(backbone: str = "virchow2", seed: int = 2) -> Path:
    features_dir = ROOT / "data" / f"features_{backbone}_reti"
    assert features_dir.is_dir(), f"missing features dir: {features_dir}"

    ds = GradingBagDatasetFull(features_dir)
    train_idx, _, _ = patient_split(ds, seed=seed)
    print(f"[compute_grade_prototypes] features_dir={features_dir}")
    print(f"[compute_grade_prototypes] train bags = {len(train_idx)} (of {len(ds)})")

    sums: dict[int, torch.Tensor] = {}
    patch_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    patient_set: dict[int, set[str]] = {0: set(), 1: set(), 2: set(), 3: set()}

    for i, idx in enumerate(train_idx):
        feats, label, slide_id = ds[idx]
        # feats: [N, D]
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)
        feats = feats.float()
        if label not in sums:
            sums[label] = torch.zeros(feats.shape[1], dtype=torch.float64)
        sums[label] += feats.sum(dim=0).to(torch.float64)
        patch_counts[label] += feats.shape[0]
        patient_id = ds.samples[idx][0].parent.name
        patient_set[label].add(patient_id)

        if (i + 1) % 50 == 0 or i + 1 == len(train_idx):
            print(f"  processed {i+1}/{len(train_idx)} bags")

    prototypes: dict[int, torch.Tensor] = {}
    for g in sorted(sums.keys()):
        prototypes[g] = (sums[g] / max(patch_counts[g], 1)).float()
        print(
            f"  G{g}: {len(patient_set[g])} patients, "
            f"{patch_counts[g]} patches, "
            f"||c|| = {prototypes[g].norm().item():.4f}"
        )

    axis_raw = prototypes[3] - prototypes[0]
    axis = axis_raw / axis_raw.norm()
    print(
        f"  ||c_G3 - c_G0|| = {axis_raw.norm().item():.4f}  ->  "
        f"axis unit-normalised, dim={axis.shape[0]}"
    )

    out_path = ROOT / "data" / f"prototypes_{backbone}_reti_train_seed{seed}.pt"
    torch.save(
        {
            "axis": axis,
            "prototypes": prototypes,
            "patch_counts": patch_counts,
            "patient_count": {g: len(p) for g, p in patient_set.items()},
            "seed": seed,
            "backbone": backbone,
            "features_dir": str(features_dir),
        },
        out_path,
    )
    print(f"[compute_grade_prototypes] saved -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="virchow2")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    compute(args.backbone, args.seed)


if __name__ == "__main__":
    main()

