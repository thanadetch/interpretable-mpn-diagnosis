"""
TITAN/CONCHv1.5 Feature Extraction — Reticulin-Stained Images (Silver Stain).

Simplified version for Reticulin images that skips color metric calculations
(originally designed for H&E). Extracts frozen CONCHv1.5 patch embeddings
grouped by source image ID. Each original image produces one .pt file
containing a stacked tensor of shape [N_patches, 768].

Input:  data/processed_grading/{Class}/{PatientID}/{ImgID}_r{Row}c{Col}.png
Output: data/features_titan_reti/{Class}/{PatientID}/{ImgID}.pt

Usage:
    python -m src.tools.extract_titan_reti \
        --data_root data/processed_grading \
        --output_dir data/features_titan_reti \
        --batch_size 64 \
        --device cuda

    # Dry-run (validates grouping without extracting):
    python -m src.tools.extract_titan_reti --dry_run
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure src/ is on sys.path when running directly (e.g., python src/tools/extract_titan_reti.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel
from core.config import hf_login

# =============================================================================
# Constants
# =============================================================================
DEFAULT_DATA_DIR = "data/processed_grading"
DEFAULT_OUTPUT_DIR = "data/features_titan_reti"
FEATURE_DIM = 768
CLASSES = ["ET", "PV", "PMF"]


# =============================================================================
# Patch Grouping Utilities
# =============================================================================


def parse_patch_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse a patch filename into (image_id, row, col).

    Supports alphanumeric characters, hyphens, and underscores for the Image ID.
    Example: "reti1_r2c5.png" → ("reti1", 2, 5)
             "3_r0c0.png" → ("3", 0, 0)
    """
    match = re.match(r"^([a-zA-Z0-9_-]+)_r(\d+)c(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Unexpected patch filename format: {filename}")
    img_id = match.group(1)
    row = int(match.group(2))
    col = int(match.group(3))
    return img_id, row, col


def group_patches_by_image(
    patient_dir: Path,
) -> Dict[str, List[Tuple[Path, int, int]]]:
    """
    Group all patches in a patient directory by source image ID.

    Returns:
        Dict mapping image_id → list of (patch_path, row, col),
        sorted by (row, col) within each group for spatial ordering.
    """
    groups: Dict[str, List[Tuple[Path, int, int]]] = defaultdict(list)

    for patch_path in patient_dir.iterdir():
        if not patch_path.suffix == ".png":
            continue
        try:
            img_id, row, col = parse_patch_filename(patch_path.name)
            groups[img_id].append((patch_path, row, col))
        except ValueError:
            print(f"  ⚠ Skipping unrecognised file: {patch_path.name}")

    # Sort each group by (row, col) for spatial ordering
    for img_id in groups:
        groups[img_id].sort(key=lambda x: (x[1], x[2]))

    return dict(groups)


# =============================================================================
# Patch Dataset (for batched feature extraction)
# =============================================================================


class PatchDataset(Dataset):
    """
    Simple dataset for loading patches from a single image group.

    Args:
        patch_paths: List of paths to patch images.
        transform:   Preprocessing transform from CONCHv1.5.
    """

    def __init__(self, patch_paths: List[Path], transform) -> None:
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        patch_path = self.patch_paths[idx]
        img = Image.open(patch_path).convert("RGB")
        tensor = self.transform(img)

        # Parse row and col from filename
        match = re.search(r"_r(\d+)c(\d+)\.png$", patch_path.name)
        row = int(match.group(1)) if match else -1
        col = int(match.group(2)) if match else -1

        return tensor, row, col


# =============================================================================
# Feature Extraction
# =============================================================================


@torch.inference_mode()
def extract_features_for_image(
    patch_paths: List[Path],
    model: torch.nn.Module,
    transform,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract CONCHv1.5 features for all patches in one image group.

    Args:
        patch_paths: Ordered list of patch file paths.
        model:       Frozen CONCHv1.5 model.
        transform:   CONCHv1.5 eval transform.
        batch_size:  Batch size for inference.
        device:      Device to run inference on.

    Returns:
        Tuple of (features [N_patches, 768], rc [N, 2] int32).
    """
    dataset = PatchDataset(patch_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    use_amp = device.type == "cuda"
    all_features = []
    all_rows = []
    all_cols = []

    for batch_tensor, batch_rows, batch_cols in loader:
        batch_tensor = batch_tensor.to(device)
        with torch.autocast(device.type, torch.float16, enabled=use_amp):
            features = model(batch_tensor)
        # Move to float32 on CPU to avoid MPS half-precision save issues
        all_features.append(features.float().cpu())
        all_rows.append(batch_rows)
        all_cols.append(batch_cols)

    concatenated_features = torch.cat(all_features, dim=0)  # [N, 768]

    # Integer coordinates tensor [N, 2]
    rc_tensor = torch.stack(
        [
            torch.cat(all_rows, dim=0),
            torch.cat(all_cols, dim=0),
        ],
        dim=1,
    ).to(torch.int32)

    return concatenated_features, rc_tensor


# =============================================================================
# Main Pipeline
# =============================================================================


def run_extraction(args: argparse.Namespace) -> None:
    """Main feature extraction pipeline."""
    data_dir = Path(args.data_root)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Discover all patients and their image groups
    # ------------------------------------------------------------------
    extraction_plan: List[Tuple[str, str, Dict[str, List[Tuple[Path, int, int]]]]] = []
    # Each entry: (class_name, patient_folder_name, image_groups)

    total_images = 0
    total_patches = 0

    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠ Class directory not found: {class_dir}")
            continue

        patient_dirs = sorted(
            [d for d in class_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        for patient_dir in patient_dirs:
            image_groups = group_patches_by_image(patient_dir)
            if not image_groups:
                print(f"  ⚠ No patches found in {patient_dir.name}")
                continue

            extraction_plan.append((class_name, patient_dir.name, image_groups))
            total_images += len(image_groups)
            total_patches += sum(len(v) for v in image_groups.values())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TITAN Feature Extraction Plan (Reticulin)")
    print("=" * 60)
    print(f"  {'Data dir':<14}: {data_dir}")
    print(f"  {'Output dir':<14}: {output_dir}")
    print(f"  {'Patients':<14}: {len(extraction_plan)}")
    print(f"  {'Total images':<14}: {total_images}")
    print(f"  {'Total patches':<14}: {total_patches}")
    print(f"  {'Feature dim':<14}: {FEATURE_DIM}")
    print(f"  {'Device':<14}: {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Dry-run mode: print plan and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n🔍 DRY RUN — Printing extraction plan:\n")
        for class_name, patient_name, groups in extraction_plan:
            print(f"  📁 {class_name}/{patient_name}")
            sorted_img_ids = sorted(groups.keys())
            for img_id in sorted_img_ids:
                patches = groups[img_id]
                out_path = output_dir / class_name / patient_name / f"{img_id}.pt"
                print(
                    f"      Image {img_id:>3}: {len(patches):4d} patches → {out_path}"
                )
        print(f"\n✅ Dry run complete. {total_images} .pt files would be created.")
        return

    # ------------------------------------------------------------------
    # Load CONCHv1.5 (via TITAN)
    # ------------------------------------------------------------------
    print("\nLoading TITAN + CONCHv1.5...")
    hf_login()
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    conch, eval_transform = titan.return_conch()
    conch = conch.to(device)
    conch.eval()
    print("✅ CONCHv1.5 loaded and frozen.\n")

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------
    processed = 0
    skipped = 0

    for class_name, patient_name, groups in tqdm(
        extraction_plan, desc="Patients", unit="patient"
    ):
        # Mirror input structure: output_dir/{Class}/{PatientID}/
        patient_output_dir = output_dir / class_name / patient_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        sorted_img_ids = sorted(groups.keys())

        for img_id in tqdm(
            sorted_img_ids,
            desc=f"  {patient_name}",
            leave=False,
            unit="img",
        ):
            out_path = patient_output_dir / f"{img_id}.pt"

            # Skip if already extracted
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            patches = groups[img_id]
            patch_paths = [p[0] for p in patches]  # Extract only paths

            features, rc_tensor = extract_features_for_image(
                patch_paths=patch_paths,
                model=conch,
                transform=eval_transform,
                batch_size=args.batch_size,
                device=device,
            )

            # Convert paths to string for saving
            patch_paths_str = [str(p) for p in patch_paths]

            # Save the final standardized dictionary (empty metrics for compatibility)
            torch.save(
                {
                    "feats": features,
                    "metrics": {},
                    "rc": rc_tensor,
                    "patch_paths": patch_paths_str,
                },
                out_path,
            )
            processed += 1

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TITAN Feature Extraction Complete (Reticulin)")
    print(f"{'=' * 60}")
    print(f"  Extracted : {processed} image bags")
    print(f"  Skipped   : {skipped} (already exist)")
    print(f"  Output    : {output_dir}")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CONCHv1.5 features (via TITAN) for Reticulin-stained images."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Input directory with {Class}/{PatientID}/ structure.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for .pt feature files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for patch inference.",
    )
    # Auto-detect best available device: CUDA > MPS > CPU
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print extraction plan without running inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
