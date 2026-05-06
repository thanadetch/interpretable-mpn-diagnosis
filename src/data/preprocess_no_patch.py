"""
No-Patch preprocessing for ablation: "do I need patches or not?".

For each input image we:
    1. Optionally crop the top/bottom (e.g. to remove scalebar overlays).
    2. Resize the entire ROI to 224x224.
    3. Save a single PNG per source image (no patch tiling).

Output layout mirrors the per-image grouping used by feature extractors,
so downstream tools that walk ``{Class}/{PatientID}/{ImgID}*.png`` keep
working unchanged:

    data/raw/{Class}/{PatientID}/{ImgID}.tif
        ->
    data/processed_grading_no_patch/{Class}/{PatientID}/{ImgID}.png

Usage:
    python -m src.data.preprocess_no_patch --stain reti
    python -m src.data.preprocess_no_patch --crop_top 50 --crop_bottom 50
    python -m src.data.preprocess_no_patch --resize 224
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

# Ensure src/ is on sys.path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_EXTENSIONS  # noqa: E402


DEFAULT_OUTPUT_NAME_BY_STAIN = {
    "reti": "processed_grading_no_patch",
    "he": "processed_subtype_no_patch",
    "all": "processed_no_patch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-Patch preprocessing (crop + resize, no patch tiling)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"Input directory (default: {RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default depends on --stain "
            "(reti -> processed_grading_no_patch, he -> processed_subtype_no_patch)."
        ),
    )
    parser.add_argument(
        "--stain",
        type=str,
        default="reti",
        choices=["all", "reti", "he"],
        help="Filter by stain type (default: reti).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Output square size after resizing the cropped ROI (default: 224).",
    )
    parser.add_argument(
        "--crop_top",
        type=int,
        default=0,
        help="Number of pixels to crop from the top of each image (default: 0).",
    )
    parser.add_argument(
        "--crop_bottom",
        type=int,
        default=0,
        help="Number of pixels to crop from the bottom of each image (default: 0).",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="PIL resampling filter for resizing (default: bicubic).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PNGs.",
    )
    return parser.parse_args()


def get_resample(name: str):
    return {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[name]


def matches_stain(filename: str, stain: str) -> bool:
    is_reti = "reti" in filename.lower()
    if stain == "reti":
        return is_reti
    if stain == "he":
        return not is_reti
    return True  # "all"


def collect_images(input_dir: Path, stain: str) -> List[Path]:
    """Walk ``{Class}/{PatientID}/*`` and collect matching image files."""
    image_paths: List[Path] = []
    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if class_dir.name.startswith("."):
            continue
        for patient_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            if "Variety" in patient_dir.name:
                continue
            for f in patient_dir.iterdir():
                if (
                    f.is_file()
                    and f.suffix.lower() in IMAGE_EXTENSIONS
                    and matches_stain(f.name, stain)
                ):
                    image_paths.append(f)
    return image_paths


def process_image(
    image_path: Path,
    out_path: Path,
    resize: int,
    crop_top: int,
    crop_bottom: int,
    resample,
    overwrite: bool,
) -> bool:
    if out_path.exists() and not overwrite:
        return False

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠ Could not load {image_path}: {e}")
        return False

    width, height = img.size
    if crop_top > 0 or crop_bottom > 0:
        top = max(0, crop_top)
        bottom = max(0, height - crop_bottom)
        if bottom <= top:
            print(f"⚠ Skipping {image_path.name}: crop removes entire image.")
            return False
        img = img.crop((0, top, width, bottom))

    img = img.resize((resize, resize), resample=resample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return True


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    if args.output_dir is None:
        out_root = (
            PROCESSED_DATA_DIR.parent / DEFAULT_OUTPUT_NAME_BY_STAIN[args.stain]
        )
    else:
        out_root = Path(args.output_dir)

    resample = get_resample(args.resample)

    image_paths = collect_images(input_dir, args.stain)

    print("=" * 60)
    print("No-Patch Preprocessing (crop + resize, no tiling)")
    print("=" * 60)
    print(f"  Input dir   : {input_dir}")
    print(f"  Output dir  : {out_root}")
    print(f"  Stain       : {args.stain}")
    print(f"  Resize      : {args.resize}x{args.resize} ({args.resample})")
    print(f"  Crop top    : {args.crop_top}px")
    print(f"  Crop bottom : {args.crop_bottom}px")
    print(f"  Images      : {len(image_paths)}")
    print(f"  Overwrite   : {args.overwrite}")
    print("=" * 60)

    written = 0
    skipped = 0
    for image_path in tqdm(image_paths, desc="Images", unit="img"):
        rel = image_path.relative_to(input_dir).with_suffix(".png")
        out_path = out_root / rel
        if process_image(
            image_path=image_path,
            out_path=out_path,
            resize=args.resize,
            crop_top=args.crop_top,
            crop_bottom=args.crop_bottom,
            resample=resample,
            overwrite=args.overwrite,
        ):
            written += 1
        else:
            skipped += 1

    print()
    print(f"✅ Wrote {written} images, skipped {skipped}.")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()

