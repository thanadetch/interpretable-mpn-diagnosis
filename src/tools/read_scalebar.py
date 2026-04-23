"""
Read Scale Bar from Reticulin (reti) Images using Gemini Vision.

This script scans all reti*.tif files under data/raw/{ET,PV,PMF}/*/,
crops the bottom-right corner, sends it to Gemini to identify the
scale bar value, and outputs a CSV with columns:
    subtype, patient, filename, scalebar_micron

Possible scalebar_micron values: 20, 50, 100, 200, unknown

Requirements:
    pip install google-generativeai Pillow

Usage:
    # Set your Gemini API key as an environment variable:
    export GOOGLE_API_KEY="your-api-key-here"

    python src/tools/read_scalebar.py

    # Or specify key inline:
    python src/tools/read_scalebar.py --gemini_key "your-api-key-here"

    # Custom raw dir / output:
    python src/tools/read_scalebar.py \
        --raw_dir data/raw \
        --output results/scalebar_results.csv
"""

import argparse
import concurrent.futures
import csv
import io
import os
import re
import threading
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai

VALID_SCALEBARS = {"20", "50", "100", "200"}

MODEL_NAME = "gemini-3-flash-preview"

PROMPT = (
    "This image is a crop from a histopathology slide. "
    "It may contain a scale bar labeled with a value in micrometers (µm or um). "
    "What is the scale bar value? "
    "Reply with ONLY one of: 20, 50, 100, 200, unknown"
)


def crop_bottom_right(
    img: Image.Image, crop_right: int = 95, crop_bottom: int = 40
) -> Image.Image:
    """Crop the bottom-right portion of an image where the scale bar usually appears.

    Args:
        img: Input image.
        crop_right: Width in pixels to crop from the right edge.
        crop_bottom: Height in pixels to crop from the bottom edge.
    """
    w, h = img.size
    left = max(0, w - crop_right)
    upper = max(0, h - crop_bottom)
    return img.crop((left, upper, w, h))


def parse_scalebar(text: str) -> str:
    """Parse Gemini response text into a scalebar value."""
    if text is None:
        return "unknown"
    text = text.strip().lower().replace("µm", "").replace("um", "").strip()
    for val in ["200", "100", "50", "20"]:
        if val in text:
            return val
    return "unknown"


def find_reti_files(raw_dir: Path):
    """Find all reti*.tif files under raw_dir/{subtype}/{patient}/."""
    results = []
    for subtype_dir in sorted(raw_dir.iterdir()):
        if not subtype_dir.is_dir() or subtype_dir.name.startswith("."):
            continue
        subtype = subtype_dir.name  # ET, PV, PMF
        for patient_dir in sorted(subtype_dir.iterdir()):
            if not patient_dir.is_dir() or patient_dir.name.startswith("."):
                continue
            patient = patient_dir.name
            for f in sorted(patient_dir.glob("*.tif")):
                if f.name.lower().startswith("reti"):
                    results.append((subtype, patient, f))
    return results


def process_one(subtype: str, patient: str, filepath: Path,
                model, crop_right: int, crop_bottom: int) -> dict:
    """Process a single reti image through Gemini and return the result dict."""
    try:
        img = Image.open(filepath).convert("RGB")
        crop = crop_bottom_right(img, crop_right=crop_right, crop_bottom=crop_bottom)

        # Convert crop to PNG bytes
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.getvalue()

        response = model.generate_content(
            [
                PROMPT,
                {"mime_type": "image/png", "data": img_bytes},
            ],
        )

        scalebar = parse_scalebar(response.text)
    except Exception as e:
        tqdm.write(f"  ⚠️ Error processing {filepath.name}: {e}")
        scalebar = "unknown"

    # Extract grade from patient folder name (e.g. "ET9 G0" -> "G0")
    match = re.search(r"G(\d+)", patient)
    grade = f"G{match.group(1)}" if match else "unknown"

    return {
        "subtype": subtype,
        "patient": patient,
        "grade": grade,
        "filename": filepath.name,
        "scalebar_micron": scalebar,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read scale bars from reti images using Gemini"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Root directory containing ET/PV/PMF subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scalebar_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--gemini_key",
        type=str,
        default=None,
        help="Gemini API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--crop_right",
        type=int,
        default=95,
        help="Width in pixels to crop from right edge (default: 95)",
    )
    parser.add_argument(
        "--crop_bottom",
        type=int,
        default=40,
        help="Height in pixels to crop from bottom edge (default: 40)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N files (for testing, e.g. --limit 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of concurrent API requests (workers). Default is 8.",
    )
    args = parser.parse_args()

    # Configure Gemini
    api_key = args.gemini_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Set GOOGLE_API_KEY env var or pass --gemini_key"
        )
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"temperature": 0.0},
    )

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    # Find all reti files
    reti_files = find_reti_files(raw_dir)
    print(f"Found {len(reti_files)} reti files across all subtypes.\n")

    if not reti_files:
        print("No reti*.tif files found. Check your --raw_dir path.")
        return

    # Log breakdown by subtype and grade
    grade_counts = {}  # {(subtype, grade): count}
    for subtype, patient, _ in reti_files:
        match = re.search(r"G(\d+)", patient)
        grade = f"G{match.group(1)}" if match else "unknown"
        key = (subtype, grade)
        grade_counts[key] = grade_counts.get(key, 0) + 1

    subtypes_found = sorted(set(k[0] for k in grade_counts))
    grades_found = sorted(set(k[1] for k in grade_counts))

    print("Reti file breakdown (subtype × grade):")
    header = f"  {'Subtype':<8}" + "".join(f"{g:>8}" for g in grades_found) + f"{'Total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for st in subtypes_found:
        row_counts = [grade_counts.get((st, g), 0) for g in grades_found]
        row_total = sum(row_counts)
        row = f"  {st:<8}" + "".join(f"{c:>8}" for c in row_counts) + f"{row_total:>8}"
        print(row)
    total_row = [sum(grade_counts.get((st, g), 0) for st in subtypes_found) for g in grades_found]
    print("  " + "-" * (len(header) - 2))
    print(f"  {'Total':<8}" + "".join(f"{c:>8}" for c in total_row) + f"{sum(total_row):>8}")
    print()

    # Apply limit if specified
    if args.limit is not None:
        reti_files = reti_files[: args.limit]
        print(f"⚡ Limited to first {args.limit} files for testing.\n")

    # =========================================================================
    # Resume support: skip already-processed files
    # =========================================================================
    out_path = Path(args.output)
    fieldnames = ["subtype", "patient", "grade", "filename", "scalebar_micron"]

    processed_keys: set[str] = set()
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        for _, row in existing_df.iterrows():
            key = f"{row['subtype']}/{row['patient']}/{row['filename']}"
            processed_keys.add(key)
        print(f"Resuming: {len(processed_keys)} images already processed.")

    files_to_process = []
    for subtype, patient, filepath in reti_files:
        key = f"{subtype}/{patient}/{filepath.name}"
        if key not in processed_keys:
            files_to_process.append((subtype, patient, filepath))

    print(f"{len(files_to_process)} images remaining to process.\n")

    if not files_to_process:
        print("All images already processed. Nothing to do.")
        return

    # =========================================================================
    # Thread-safe CSV writing + concurrent processing
    # =========================================================================
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_lock = threading.Lock()
    write_header = not out_path.exists()
    csv_file = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(
                process_one, subtype, patient, filepath,
                model, args.crop_right, args.crop_bottom,
            ): (subtype, patient, filepath)
            for subtype, patient, filepath in files_to_process
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reading scale bars",
        ):
            filepath_info = futures[future]
            try:
                info = future.result()
                if info:
                    with csv_lock:
                        writer.writerow(info)
                        csv_file.flush()
            except Exception as e:
                tqdm.write(f"  ⚠️ Unhandled error for {filepath_info[2].name}: {e}")

    csv_file.close()

    # =========================================================================
    # Print summary from saved CSV
    # =========================================================================
    df = pd.read_csv(out_path)
    print(f"\n✅ Done! Saved {len(df)} rows to {out_path}")

    counts = Counter(df["scalebar_micron"].astype(str).tolist())
    print("\nSummary:")
    for val in ["20", "50", "100", "200", "unknown"]:
        print(f"  {val:>7s} µm: {counts.get(val, 0)} images")


if __name__ == "__main__":
    main()

