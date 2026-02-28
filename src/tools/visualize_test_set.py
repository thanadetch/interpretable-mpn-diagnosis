"""
Batch Test Set Heatmap Visualization.

Generates attention heatmaps for ALL patients in the test set of a trained
MIL experiment. Loads the checkpoint, recovers test_idx, and iterates over
every patient/image to produce sorted grid galleries.

Output: results/test_set_heatmaps/{ExperimentName}/{Class}/{PatientID}/

Usage:
    python -m src.tools.visualize_test_set \
        --mil_checkpoint experiments/simple_titan_.../best_simple_titan.pth

    # Custom data root:
    python -m src.tools.visualize_test_set \
        --mil_checkpoint experiments/simple_titan_.../best.pth \
        --data_root data
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Add src/ to path so imports work when run from project root
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import CLASS_MAP, CLASS_MAP_INV, RESULTS_DIR
from data.bag_dataset import MPNBagDatasetFull
from visualize_heatmap import (
    BACKBONE_CONFIG,
    generate_grid_gallery,
    group_patches_by_image,
    load_backbone,
    load_mil_model,
)

CLASS_NAMES = [CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))]


# =============================================================================
# Checkpoint Helpers
# =============================================================================


def load_checkpoint_metadata(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[List[int], str, str, dict]:
    """
    Load checkpoint and extract test set metadata.

    Returns:
        test_idx:      List of dataset indices belonging to the test set.
        backbone_name: Backbone used (e.g. 'titan').
        model_type:    MIL model type (e.g. 'simple').
        ckpt_args:     Original training arguments dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ckpt_args = checkpoint["args"]
    test_idx = checkpoint["test_idx"]
    backbone_name = ckpt_args["backbone"]
    model_type = ckpt_args["model_type"]
    return test_idx, backbone_name, model_type, ckpt_args


def get_test_patients(
    dataset: MPNBagDatasetFull,
    test_idx: List[int],
) -> Dict[str, Dict]:
    """
    Map test indices to unique patients with their metadata.

    Returns:
        Dict keyed by patient_id with values:
            {
                "class_name": str,
                "label": int,
                "feature_paths": List[Path],  # .pt files for this patient
            }
    """
    patients: Dict[str, Dict] = {}

    for idx in test_idx:
        pt_path, label = dataset.samples[idx]
        # Structure: features_dir/{Class}/{PatientID}/{ImageID}.pt
        patient_id = pt_path.parent.name
        class_name = pt_path.parent.parent.name

        if patient_id not in patients:
            patients[patient_id] = {
                "class_name": class_name,
                "label": label,
                "feature_paths": [],
            }
        patients[patient_id]["feature_paths"].append(pt_path)

    return patients


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate attention heatmaps for all test set patients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.tools.visualize_test_set \\
      --mil_checkpoint experiments/simple_titan_20260224/best_simple_titan.pth

  python -m src.tools.visualize_test_set \\
      --mil_checkpoint experiments/hybrid_titan_20260224/best_hybrid_titan.pth \\
      --data_root data --n_cols 6
        """,
    )
    parser.add_argument(
        "--mil_checkpoint",
        type=str,
        required=True,
        help="Path to trained MIL model checkpoint (.pth).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: results/test_set_heatmaps/{exp_name}.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Patch size in pixels (default: 224).",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=8,
        help="Number of columns in the grid (default: 8).",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="Optional string to append to the output directory name (e.g., 'run2' or 'filtered').",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default=None,
        choices=["ET", "PV", "PMF"],
        help="Specific subtype to visualize. If not provided, visualizes all subtypes.",
    )

    # Auto-detect device
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device,
        help="Device for inference (cuda, mps, or cpu).",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mil_checkpoint = Path(args.mil_checkpoint)

    # ── 1. Load checkpoint metadata ─────────────────────────────────
    print("=" * 60)
    print("Batch Test Set Heatmap Visualization")
    print("=" * 60)

    test_idx, backbone_name, model_type, ckpt_args = load_checkpoint_metadata(
        mil_checkpoint, device
    )
    cfg = BACKBONE_CONFIG[backbone_name]
    features_dir = Path(args.data_root) / cfg["feature_dir"]
    exp_name = mil_checkpoint.parent.name

    print(f"  Checkpoint:  {mil_checkpoint}")
    print(f"  Experiment:  {exp_name}")
    print(f"  Model:       {model_type.upper()}")
    print(f"  Backbone:    {cfg['display_name']}")
    print(f"  Test Indices: {len(test_idx)}")
    print(f"  Device:      {device}")

    # ── 2. Initialise dataset and discover test patients ────────────
    max_patches = ckpt_args.get("max_patches", None)
    dataset = MPNBagDatasetFull(features_dir, max_patches=max_patches)
    patients = get_test_patients(dataset, test_idx)

    # Filter by subtype if specified
    if args.subtype is not None:
        patients = {
            pid: info
            for pid, info in patients.items()
            if info["class_name"] == args.subtype
        }
        print(f"\n  Filtering to subtype: {args.subtype}")

    # Count per-class
    class_counts: Dict[str, int] = defaultdict(int)
    for info in patients.values():
        class_counts[info["class_name"]] += 1

    print(f"\n  Test Patients: {len(patients)}")
    for cn in CLASS_NAMES:
        print(f"    {cn}: {class_counts.get(cn, 0)} patients")

    # ── 3. Load MIL model ───────────────────────────────────────────
    print("\nLoading MIL model...")
    mil_model, _, _ = load_mil_model(mil_checkpoint, device)
    print("  ✅ MIL model loaded.\n")

    # ── 4. Load ViT backbone ────────────────────────────────────────
    print(f"Loading {cfg['display_name']} backbone...")
    backbone, backbone_transform, get_vit_attention = load_backbone(
        backbone_name, device
    )
    print("  ✅ Backbone loaded.\n")

    # ── 5. Setup output directory ───────────────────────────────────
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        folder_name = f"{exp_name}_{args.postfix}" if args.postfix else exp_name
        output_root = RESULTS_DIR / "test_set_heatmaps" / folder_name

    print(f"  Output: {output_root}")
    print("=" * 60)

    # ── 6. Locate raw patch directories ─────────────────────────────
    # Raw patches live at: data/processed_subtype/{Class}/{PatientID}/
    raw_patch_root = Path(args.data_root) / "processed_subtype"

    # ── 7. Generate heatmaps for each test patient ──────────────────
    total_images = 0
    skipped_patients = 0

    for patient_idx, (patient_id, info) in enumerate(sorted(patients.items()), start=1):
        class_name = info["class_name"]
        feature_paths = sorted(info["feature_paths"])

        print(
            f"\n[{patient_idx}/{len(patients)}] "
            f"{class_name}/{patient_id}  ({len(feature_paths)} images)"
        )

        # Locate raw patch directory
        patient_patch_dir = raw_patch_root / class_name / patient_id
        if not patient_patch_dir.exists():
            print(f"  ⚠ Raw patch directory not found: {patient_patch_dir} — skipping.")
            skipped_patients += 1
            continue

        # Discover available image groups from raw patches
        available_groups = group_patches_by_image(patient_patch_dir)

        # Patient output directory
        patient_output_dir = output_root / class_name / patient_id

        # Generate heatmap for each image
        for feat_path in feature_paths:
            img_id = feat_path.stem  # e.g. "1", "2", etc.

            if img_id not in available_groups:
                print(f"  ⚠ No raw patches for image {img_id} — skipping.")
                continue

            save_path = patient_output_dir / f"{img_id}_grid.png"

            print(f"  Image {img_id}...")
            generate_grid_gallery(
                patient_dir=patient_patch_dir,
                image_id=img_id,
                mil_model=mil_model,
                get_vit_attention=get_vit_attention,
                backbone_transform=backbone_transform,
                features_path=feat_path,
                class_name=class_name,
                save_path=save_path,
                patch_size=args.patch_size,
                n_cols=args.n_cols,
                device=device,
            )
            total_images += 1

    # ── 8. Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Batch Heatmap Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Patients processed: {len(patients) - skipped_patients}/{len(patients)}")
    print(f"  Images processed:   {total_images}")
    print(f"  Patients skipped:   {skipped_patients}")
    print(f"  Output directory:   {output_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
