"""
Preprocessing script for MPN bone marrow images.
Performs Sliding Window Patching to preserve high-resolution details.

Images are padded with black pixels to ensure no part of the original
image is discarded during patching.

Usage:
    python src/preprocess.py
    python src/preprocess.py --patch_size 512 --step_size 256
"""
import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_EXTENSIONS


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

    return parser.parse_args()


def calculate_padded_size(
    original_size: int,
    patch_size: int,
    step_size: int,
) -> int:
    """
    Calculate the padded size needed to ensure full coverage with sliding window.

    The padded size ensures that the last patch captures the very last pixels
    of the original image (no pixels are discarded).

    Args:
        original_size: Original dimension (width or height)
        patch_size: Size of each patch
        step_size: Step size for sliding window

    Returns:
        Padded size that allows complete coverage
    """
    if original_size <= patch_size:
        # Image is smaller than patch, just need patch_size
        return patch_size

    # Calculate how many full steps we can take from position 0
    # The last patch starts at position: start_pos, and ends at: start_pos + patch_size
    # We need: start_pos + patch_size >= original_size for full coverage

    # Number of steps needed (ceiling division)
    # After n steps, last patch starts at n * step_size
    # We need: n * step_size + patch_size >= original_size
    # So: n >= (original_size - patch_size) / step_size
    # n = ceil((original_size - patch_size) / step_size)

    if original_size <= patch_size:
        num_steps = 0
    else:
        num_steps = -(-((original_size - patch_size)) // step_size)  # Ceiling division

    # The padded size should allow exactly num_steps + 1 patches
    # Last patch starts at: num_steps * step_size
    # Last patch ends at: num_steps * step_size + patch_size
    padded_size = num_steps * step_size + patch_size

    return padded_size


def pad_image_for_patching(
    image: Image.Image,
    patch_size: int,
    step_size: int,
) -> Image.Image:
    """
    Pad image with black pixels to ensure complete coverage during patching.

    The sliding window will capture every pixel of the original image.
    Padding is added to the right and bottom edges as needed.

    Args:
        image: PIL Image to pad
        patch_size: Size of each patch
        step_size: Step size for sliding window

    Returns:
        Padded PIL Image (or original if no padding needed)
    """
    width, height = image.size

    # Calculate required padded dimensions
    new_width = calculate_padded_size(width, patch_size, step_size)
    new_height = calculate_padded_size(height, patch_size, step_size)

    # Check if padding is needed
    if new_width == width and new_height == height:
        return image

    # Create new image with black background (0, 0, 0)
    padded = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # Paste original image at top-left corner
    padded.paste(image, (0, 0))


    return padded


def extract_patches(
    image: Image.Image,
    patch_size: int,
    step_size: int,
) -> List[Tuple[Image.Image, int, int]]:
    """
    Extract overlapping patches from an image using sliding window.

    Args:
        image: PIL Image to extract patches from
        patch_size: Size of each patch (patch_size x patch_size)
        step_size: Step size for sliding window

    Returns:
        List of tuples: (patch_image, row_idx, col_idx)
    """
    width, height = image.size
    patches: List[Tuple[Image.Image, int, int]] = []

    row_idx = 0
    y = 0
    while y + patch_size <= height:
        col_idx = 0
        x = 0
        while x + patch_size <= width:
            # Extract patch
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append((patch, row_idx, col_idx))

            x += step_size
            col_idx += 1

        y += step_size
        row_idx += 1

    return patches


def process_image(
    image_path: Path,
    output_dir: Path,
    patch_size: int,
    step_size: int,
) -> int:
    """
    Process a single image: pad if needed, extract patches, and save them.

    Images are padded with black pixels to ensure no part of the original
    image is discarded during patching.

    Args:
        image_path: Path to input image
        output_dir: Directory to save patches
        patch_size: Size of each patch
        step_size: Step size for sliding window

    Returns:
        Number of patches extracted
    """
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not load {image_path}: {e}")
        return 0

    original_width, original_height = image.size

    # Pad image to ensure complete coverage (no pixels discarded)
    image = pad_image_for_patching(image, patch_size, step_size)
    padded_width, padded_height = image.size

    # Log if padding was applied
    if padded_width != original_width or padded_height != original_height:
        pass  # Padding applied silently; can add logging if needed

    # Extract patches
    patches = extract_patches(image, patch_size, step_size)

    if len(patches) == 0:
        print(f"Warning: No patches extracted from {image_path}")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save patches
    original_stem = image_path.stem  # Filename without extension

    for patch, row_idx, col_idx in patches:
        patch_idx = row_idx * 100 + col_idx  # Unique index for each patch
        patch_name = f"{original_stem}_r{row_idx}c{col_idx}.png"
        patch_path = output_dir / patch_name

        patch.save(patch_path, "PNG")

    return len(patches)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    step_size: int,
) -> dict:
    """
    Process entire dataset: extract patches from all images.

    Directory structure:
        Input:  data/raw/{Class}/{PatientID}/{ImageFile}.tif
        Output: data/processed/{Class}/{PatientID}/{ImageFile}_r{row}c{col}.png

    All images are processed with padding to ensure complete coverage.

    Args:
        input_dir: Input directory (data/raw)
        output_dir: Output directory (data/processed)
        patch_size: Size of each patch
        step_size: Step size for sliding window

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

    print(f"\n{'='*60}")
    print(f"Sliding Window Patching")
    print(f"{'='*60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size:       {patch_size}x{patch_size}")
    print(f"Step size:        {step_size} ({100 * (patch_size - step_size) / patch_size:.0f}% overlap)")
    print(f"Padding:          Black pixels (0,0,0) for complete coverage")
    print(f"{'='*60}\n")

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
            image_files = [
                f for f in patient_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            # Process each image
            for image_path in image_files:
                stats["total_images"] += 1
                stats["classes"][class_name]["images"] += 1

                # Create output path maintaining directory structure
                relative_path = image_path.relative_to(input_dir)
                patient_output_dir = output_dir / relative_path.parent

                # Process image and extract patches
                num_patches = process_image(
                    image_path=image_path,
                    output_dir=patient_output_dir,
                    patch_size=patch_size,
                    step_size=step_size,
                )

                if num_patches == 0:
                    stats["skipped_images"] += 1
                else:
                    stats["total_patches"] += num_patches
                    stats["classes"][class_name]["patches"] += num_patches

    return stats


def print_stats(stats: dict) -> None:
    """Print processing statistics."""
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total patches created:  {stats['total_patches']}")
    print(f"Skipped images:         {stats['skipped_images']}")
    print(f"\nPer-class statistics:")
    print(f"{'-'*60}")

    for class_name, class_stats in stats["classes"].items():
        print(f"  {class_name}:")
        print(f"    Patients: {class_stats['patients']}")
        print(f"    Images:   {class_stats['images']}")
        print(f"    Patches:  {class_stats['patches']}")
        if class_stats['images'] > 0:
            avg_patches = class_stats['patches'] / class_stats['images']
            print(f"    Avg patches/image: {avg_patches:.1f}")

    print(f"{'='*60}\n")

    if stats['total_images'] > 0:
        expansion_factor = stats['total_patches'] / stats['total_images']
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
    )

    # Print statistics
    print_stats(stats)

    print(f"\nPatches saved to: {output_dir}")
    print("\nTo use processed patches for training, update DATA_DIR in config.py:")
    print(f'  DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"')


if __name__ == "__main__":
    main()

