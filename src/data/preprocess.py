"""
Preprocessing script for MPN bone marrow images.
Performs Sliding Window Patching with Edge-Anchored Tiling.

Instead of padding with black pixels, the last patch in each row/column
is snapped to the image edge so that every pixel is covered without
introducing artificial black borders.

Usage:
    python src/preprocess.py
    python src/preprocess.py --patch_size 512 --step_size 256
    python src/preprocess.py --crop_top 50 --crop_bottom 50
    python src/preprocess.py --use_od_filter --save_rejected
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from skimage.color import rgb2hsv
from skimage.morphology import flood

from PIL import Image
from tqdm import tqdm

# Ensure src/ is on sys.path when running directly (e.g., python src/data/preprocess.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess MPN images using sliding window patching"
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Size of each patch (default: 512)",
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=256,
        help="Step size for sliding window (default: 256 for 50%% overlap)",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"Input directory (default: {RAW_DATA_DIR})",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help=f"Output directory (default: {PROCESSED_DATA_DIR})",
    )

    parser.add_argument(
        "--stain",
        type=str,
        default="all",
        choices=["all", "reti", "he"],
        help="Filter by stain type: 'reti' (grading), 'he' (classification), or 'all'",
    )

    parser.add_argument(
        "--crop_top",
        type=int,
        default=0,
        help="Number of pixels to crop from the top of each image (default: 0)",
    )

    parser.add_argument(
        "--crop_bottom",
        type=int,
        default=0,
        help="Number of pixels to crop from the bottom of each image (default: 0)",
    )

    parser.add_argument(
        "--use_od_filter",
        action="store_true",
        help="Enable OD-based background filtering",
    )

    parser.add_argument(
        "--tissue_threshold",
        type=float,
        default=0.15,
        help="OD threshold to consider a pixel as tissue (default: 0.15)",
    )

    parser.add_argument(
        "--min_tissue_ratio",
        type=float,
        default=0.3,
        help="Minimum fraction of tissue pixels to keep a patch (default: 0.3)",
    )

    parser.add_argument(
        "--save_rejected",
        action="store_true",
        help="Save rejected background patches",
    )

    parser.add_argument(
        "--min_specimen_fraction",
        type=float,
        default=0.12,
        help="Minimum fraction of specimen (non-glass) pixels to keep a patch (default: 0.12)",
    )

    return parser.parse_args()


def compute_specimen_mask(img_array_uint8):
    """
    Creates a mask to remove outside glass background using Adaptive HSV.
    """
    img = img_array_uint8.astype(np.float32)
    h, w, _ = img.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

    hsv = rgb2hsv(img / 255.0)
    S = hsv[..., 1]
    V = hsv[..., 2]

    # Border stats
    border_S = np.concatenate([S[0, :], S[-1, :], S[:, 0], S[:, -1]])
    border_V = np.concatenate([V[0, :], V[-1, :], V[:, 0], V[:, -1]])

    # Adaptive Thresholds (Hardcoded percentiles)
    V_thr = np.percentile(border_V, 95)
    S_thr = np.percentile(border_S, 5)

    # Failsafe Clamp (crucial to protect 100% tissue patches)
    V_thr = max(V_thr, 0.60)
    S_thr = min(S_thr, 0.40)

    # Flood-fill on brightness map
    gray = np.clip(V * 255.0, 0, 255).astype(np.uint8)
    tolerance = 20

    outside = np.zeros((h, w), dtype=bool)
    for r, c in corners:
        if (V[r, c] >= V_thr) and (S[r, c] <= S_thr):
            outside |= flood(gray, (r, c), tolerance=tolerance)

    specimen_mask = ~outside
    return specimen_mask


def is_tissue(
    patch,
    threshold: float,
    min_ratio: float,
    min_specimen_fraction: float = 0.12,
) -> bool:
    """
    2-Stage Tissue Filter.
    """
    img_array = np.asarray(patch)

    # Stage 1: Specimen Mask
    specimen_mask = compute_specimen_mask(img_array)
    specimen_fraction = specimen_mask.mean()

    if specimen_fraction < min_specimen_fraction:
        return False

    # Stage 2: OD filter inside specimen with safe clipping
    od = -np.log10(np.clip(img_array.astype(np.float32), 1.0, 255.0) / 255.0)
    tissue_mask = od.mean(axis=2) > threshold

    tissue_ratio_in_spec = (tissue_mask & specimen_mask).sum() / (
        specimen_mask.sum() + 1e-6
    )

    return tissue_ratio_in_spec >= min_ratio


def extract_patches(
    image: Image.Image,
    patch_size: int,
    step_size: int,
    use_od_filter: bool = False,
    tissue_threshold: float = 0.15,
    min_tissue_ratio: float = 0.3,
    min_specimen_fraction: float = 0.12,
) -> Tuple[List[Tuple[Image.Image, int, int]], List[Tuple[Image.Image, int, int]]]:
    """
    Extract patches using Edge-Anchored Tiling with optional OD filtering.

    Iterates through the image with *step_size*.  When the next regular
    step would exceed the image boundary, the last patch is snapped to
    align exactly with the right / bottom edge.  A set of seen origins
    prevents duplicate patches when dimensions are an exact multiple of
    the step size.

    Args:
        image: PIL Image to extract patches from
        patch_size: Size of each patch (patch_size x patch_size)
        step_size: Step size for sliding window
        use_od_filter: Whether to apply OD-based tissue filtering
        tissue_threshold: OD threshold for tissue detection
        min_tissue_ratio: Minimum tissue fraction to accept a patch
        min_specimen_fraction: Minimum specimen fraction to accept a patch

    Returns:
        Tuple of (valid_patches, rejected_patches),
        each a list of (patch_image, row_idx, col_idx)
    """
    width, height = image.size
    valid_patches: List[Tuple[Image.Image, int, int]] = []
    rejected_patches: List[Tuple[Image.Image, int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    # Collect y-positions (edge-anchored)
    y_positions: List[int] = []
    y = 0
    while y + patch_size <= height:
        y_positions.append(y)
        y += step_size
    # Snap last row to bottom edge if not already covered
    last_y = height - patch_size
    if last_y >= 0 and (not y_positions or y_positions[-1] != last_y):
        y_positions.append(last_y)

    # Collect x-positions (edge-anchored)
    x_positions: List[int] = []
    x = 0
    while x + patch_size <= width:
        x_positions.append(x)
        x += step_size
    # Snap last column to right edge if not already covered
    last_x = width - patch_size
    if last_x >= 0 and (not x_positions or x_positions[-1] != last_x):
        x_positions.append(last_x)

    for row_idx, py in enumerate(y_positions):
        for col_idx, px in enumerate(x_positions):
            if (px, py) in seen:
                continue
            seen.add((px, py))
            patch = image.crop((px, py, px + patch_size, py + patch_size))

            is_valid = (
                is_tissue(
                    patch, tissue_threshold, min_tissue_ratio, min_specimen_fraction
                )
                if use_od_filter
                else True
            )

            if is_valid:
                valid_patches.append((patch, row_idx, col_idx))
            else:
                rejected_patches.append((patch, row_idx, col_idx))

    return valid_patches, rejected_patches


def process_image(
    image_path: Path,
    output_dir: Path,
    patch_size: int,
    step_size: int,
    crop_top: int = 0,
    crop_bottom: int = 0,
    use_od_filter: bool = False,
    tissue_threshold: float = 0.15,
    min_tissue_ratio: float = 0.3,
    min_specimen_fraction: float = 0.12,
    save_rejected: bool = False,
    rejected_dir: Optional[Path] = None,
) -> int:
    """
    Process a single image: optionally crop, extract patches via
    edge-anchored tiling with optional OD filtering, and save them.

    Args:
        image_path: Path to input image
        output_dir: Directory to save valid patches
        patch_size: Size of each patch
        step_size: Step size for sliding window
        crop_top: Pixels to remove from the top edge
        crop_bottom: Pixels to remove from the bottom edge
        use_od_filter: Whether to apply OD-based tissue filtering
        tissue_threshold: OD threshold for tissue detection
        min_tissue_ratio: Minimum tissue fraction to accept a patch
        min_specimen_fraction: Minimum specimen fraction to accept a patch
        save_rejected: Whether to save rejected background patches
        rejected_dir: Directory to save rejected patches

    Returns:
        Number of valid patches extracted
    """
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not load {image_path}: {e}")
        return 0

    original_width, original_height = image.size

    # Optional cropping
    if crop_top > 0 or crop_bottom > 0:
        image = image.crop((0, crop_top, original_width, original_height - crop_bottom))

    # Extract patches using edge-anchored tiling
    valid_patches, rejected_patches = extract_patches(
        image,
        patch_size,
        step_size,
        use_od_filter=use_od_filter,
        tissue_threshold=tissue_threshold,
        min_tissue_ratio=min_tissue_ratio,
        min_specimen_fraction=min_specimen_fraction,
    )

    if len(valid_patches) == 0:
        print(f"Warning: No valid patches extracted from {image_path}")
        return 0

    # Create output directory and save valid patches
    output_dir.mkdir(parents=True, exist_ok=True)
    original_stem = image_path.stem

    for patch, row_idx, col_idx in valid_patches:
        patch_name = f"{original_stem}_r{row_idx}c{col_idx}.png"
        patch_path = output_dir / patch_name
        patch.save(patch_path, "PNG")

    # Save rejected patches if requested
    if save_rejected and rejected_dir is not None and len(rejected_patches) > 0:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for patch, row_idx, col_idx in rejected_patches:
            patch_name = f"{original_stem}_r{row_idx}c{col_idx}_rejected.png"
            patch_path = rejected_dir / patch_name
            patch.save(patch_path, "PNG")

    return len(valid_patches)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    step_size: int,
    stain: str = "all",
    crop_top: int = 0,
    crop_bottom: int = 0,
    use_od_filter: bool = False,
    tissue_threshold: float = 0.15,
    min_tissue_ratio: float = 0.3,
    min_specimen_fraction: float = 0.12,
    save_rejected: bool = False,
) -> dict:
    """
    Process entire dataset: extract patches from all images.

    Directory structure:
        Input:  data/raw/{Class}/{PatientID}/{ImageFile}.tif
        Output: data/processed/{Class}/{PatientID}/{ImageFile}_r{row}c{col}.png

    All images are processed with edge-anchored tiling for complete coverage.

    Args:
        input_dir: Input directory (data/raw)
        output_dir: Output directory (data/processed)
        patch_size: Size of each patch
        step_size: Step size for sliding window
        crop_top: Pixels to remove from the top edge
        crop_bottom: Pixels to remove from the bottom edge
        use_od_filter: Whether to apply OD-based tissue filtering
        tissue_threshold: OD threshold for tissue detection
        min_tissue_ratio: Minimum tissue fraction to accept a patch
        min_specimen_fraction: Minimum specimen fraction to accept a patch
        save_rejected: Whether to save rejected background patches

    Returns:
        Statistics dictionary
    """
    stats = {
        "total_images": 0,
        "total_patches": 0,
        "skipped_images": 0,
        "classes": {},
    }

    # Find all class directories (ET, PV, PMF)
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"No class directories found in {input_dir}")
        return stats

    print(f"\n{'=' * 60}")
    print(f"Sliding Window Patching")
    print(f"{'=' * 60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size:       {patch_size}x{patch_size}")
    print(
        f"Step size:        {step_size} ({100 * (patch_size - step_size) / patch_size:.0f}% overlap)"
    )
    print(f"Tiling:           Edge-anchored (no black padding)")
    print(f"Crop top:         {crop_top}px")
    print(f"Crop bottom:      {crop_bottom}px")
    print(f"OD filter:        {'ON' if use_od_filter else 'OFF'}")
    if use_od_filter:
        print(f"  Tissue thresh:  {tissue_threshold}")
        print(f"  Min ratio:      {min_tissue_ratio}")
        print(f"  Min specimen:   {min_specimen_fraction}")
    print(f"Save rejected:    {'YES' if save_rejected else 'NO'}")
    print(f"Stain filter:     {stain}")
    print(f"{'=' * 60}\n")

    # Process each class
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name

        # Skip hidden directories and non-class directories
        if class_name.startswith("."):
            continue

        stats["classes"][class_name] = {
            "images": 0,
            "patches": 0,
            "patients": 0,
        }

        # Find all patient directories
        patient_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        stats["classes"][class_name]["patients"] = len(patient_dirs)

        print(f"Processing class: {class_name} ({len(patient_dirs)} patients)")

        # Process each patient
        for patient_dir in tqdm(patient_dirs, desc=f"  {class_name}", leave=True):
            patient_id = patient_dir.name

            # Skip directories containing "Variety"
            if "Variety" in patient_id:
                continue

            # Find all images in patient directory
            all_files = [
                f
                for f in patient_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            # Filter by stain
            image_files = []
            for f in all_files:
                is_reti = "reti" in f.name.lower()

                if stain == "reti" and is_reti:
                    image_files.append(f)
                elif stain == "he" and not is_reti:
                    image_files.append(f)
                elif stain == "all":
                    image_files.append(f)

            # Process each image
            for image_path in image_files:
                stats["total_images"] += 1
                stats["classes"][class_name]["images"] += 1

                # Create output path maintaining directory structure
                relative_path = image_path.relative_to(input_dir)
                patient_output_dir = output_dir / relative_path.parent

                # Determine task-specific rejected directory
                rejected_dir: Optional[Path] = None
                if save_rejected:
                    is_reti = "reti" in image_path.name.lower()
                    task_name = "grading" if is_reti else "subtype"
                    rejected_dir = (
                        output_dir.parent
                        / f"{task_name}_rejected_patches"
                        / relative_path.parent
                    )

                # Process image and extract patches
                num_patches = process_image(
                    image_path=image_path,
                    output_dir=patient_output_dir,
                    patch_size=patch_size,
                    step_size=step_size,
                    crop_top=crop_top,
                    crop_bottom=crop_bottom,
                    use_od_filter=use_od_filter,
                    tissue_threshold=tissue_threshold,
                    min_tissue_ratio=min_tissue_ratio,
                    min_specimen_fraction=min_specimen_fraction,
                    save_rejected=save_rejected,
                    rejected_dir=rejected_dir,
                )

                if num_patches == 0:
                    stats["skipped_images"] += 1
                else:
                    stats["total_patches"] += num_patches
                    stats["classes"][class_name]["patches"] += num_patches

    return stats


def print_stats(stats: dict) -> None:
    """Print processing statistics."""
    print(f"\n{'=' * 60}")
    print(f"Processing Complete!")
    print(f"{'=' * 60}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total patches created:  {stats['total_patches']}")
    print(f"Skipped images:         {stats['skipped_images']}")
    print(f"\nPer-class statistics:")
    print(f"{'-' * 60}")

    for class_name, class_stats in stats["classes"].items():
        print(f"  {class_name}:")
        print(f"    Patients: {class_stats['patients']}")
        print(f"    Images:   {class_stats['images']}")
        print(f"    Patches:  {class_stats['patches']}")
        if class_stats["images"] > 0:
            avg_patches = class_stats["patches"] / class_stats["images"]
            print(f"    Avg patches/image: {avg_patches:.1f}")

    print(f"{'=' * 60}\n")

    if stats["total_images"] > 0:
        expansion_factor = stats["total_patches"] / stats["total_images"]
        print(f"Dataset expansion factor: {expansion_factor:.1f}x")
        print(f"  Original images: {stats['total_images']}")
        print(f"  Total patches:   {stats['total_patches']}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Process dataset
    stats = process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        patch_size=args.patch_size,
        step_size=args.step_size,
        stain=args.stain,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        use_od_filter=args.use_od_filter,
        tissue_threshold=args.tissue_threshold,
        min_tissue_ratio=args.min_tissue_ratio,
        min_specimen_fraction=args.min_specimen_fraction,
        save_rejected=args.save_rejected,
    )

    # Print statistics
    print_stats(stats)

    print(f"\nPatches saved to: {output_dir}")
    print("\nTo use processed patches for training, update DATA_DIR in config.py:")
    print(f'  DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"')


if __name__ == "__main__":
    main()
