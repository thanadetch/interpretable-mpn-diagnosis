"""
Extract expert morphological metrics from H&E bone marrow images via Gemini.

This script uses the google-generativeai library to evaluate
histopathological metrics relevant to MPN subtyping.

Usage:
    python src/data/evaluate_cellularity.py
    python src/data/evaluate_cellularity.py --subtype ET
    python src/data/evaluate_cellularity.py --subtype PV --num-rois 10
    python src/data/evaluate_cellularity.py --subtype PMF --postfix pmf_v2
    python src/data/evaluate_cellularity.py --batch_size 8

Output:
    results/expert_metrics/expert_metrics_results.csv
    results/expert_metrics/expert_metrics_results_<postfix>.csv
"""

import argparse
import concurrent.futures
import csv
import json
import os
import tempfile
import threading
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw")
MODEL_NAME = "gemini-3-flash-preview"
SLEEP_SECONDS = 2

# Schema enforcing a single JSON object with exactly these 2 keys
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Cellularity": {"type": "STRING"},
        "Cellularity_Confidence": {"type": "INTEGER"},
    },
    "required": [
        "Cellularity",
        "Cellularity_Confidence",
    ],
}

# The 2 keys returned by the model (1 metric + 1 confidence score)
METRIC_KEYS = [
    "Cellularity",
    "Cellularity_Confidence",
]

SYSTEM_PROMPT = (
    "You are an expert Hematopathologist evaluating a single 20x H&E-stained bone marrow biopsy ROI for a Myeloproliferative Neoplasm (MPN) research study.\n\n"
    "Your task is to assess ONLY the overall marrow cellularity of this ROI.\n\n"
    "This is a 20x ROI-level assessment, not a high-power nuclear-detail assessment.\n"
    "Focus only on the overall hematopoietic cellularity pattern across the full ROI.\n\n"
    "IMPORTANT RULES (you MUST follow these):\n"
    "1. Assess the ENTIRE ROI, but evaluate cellularity ONLY within evaluable marrow spaces.\n"
    "2. Do NOT let one focal hotspot, one edge region, one isolated dense area, or one isolated fatty area dominate the final label.\n"
    "3. Ignore empty slide background, tissue dropout, folds, crushed tissue, stain artifact, blur, hemorrhage, and edge artifact.\n"
    "4. Do NOT let large bony trabeculae dominate the final cellularity label.\n"
    "5. Base the final category on the PREDOMINANT and REPRESENTATIVE overall marrow pattern across the image.\n"
    "6. Evaluate cellularity from the overall hematopoietic-to-fat balance across the evaluable marrow spaces of the ROI.\n"
    "7. Do NOT overcall hypercellularity from one dense focus alone.\n"
    "8. Do NOT overcall hypocellularity merely because the ROI contains much bone, blank background, tissue dropout, or limited tissue.\n"
    "9. If evaluable marrow is limited, patchy, or mixed, choose the best overall summary label and lower confidence accordingly rather than overcalling Hypocellular or Hypercellular.\n"
    '10. Use "Hypocellular" only when reduced hematopoietic cellularity is clearly present across multiple representative marrow spaces.\n'
    '11. Use "Hypercellular / Panmyelosis" only when increased hematopoietic cellularity is clearly present across representative marrow spaces, not just in one focal dense region.\n'
    "12. If the image is blurry, obscured, artifact-prone, or difficult to assess, assign a LOWER confidence rather than guessing.\n\n"
    "CELLULARITY DEFINITIONS:\n"
    '- "Hypocellular" = the evaluable marrow spaces overall show clearly reduced hematopoietic cellularity relative to fat/background across representative areas.\n'
    '- "Normocellular" = the evaluable marrow spaces overall show a balanced or expected hematopoietic cellularity pattern without clear diffuse hypercellularity or hypocellularity.\n'
    '- "Hypercellular / Panmyelosis" = the evaluable marrow spaces overall show clearly increased hematopoietic cellularity across representative areas, not just a focal dense region.\n\n'
    "CONFIDENCE SCORING:\n"
    'Provide "Cellularity_Confidence" as an integer from 1 to 10:\n'
    "- 9-10 = obvious and well supported across the ROI\n"
    "- 7-8 = likely present and reasonably clear\n"
    "- 5-6 = mixed, borderline, or partially assessable\n"
    "- 3-4 = difficult due to blur, artifact, limited evaluable marrow, or ambiguity\n"
    "- 1-2 = essentially not assessable\n\n"
    "Return ONLY a JSON object with EXACTLY these 2 keys and no extra text.\n\n"
    "METRICS:\n"
    '  "Cellularity": one of "Hypocellular" | "Normocellular" | "Hypercellular / Panmyelosis"\n'
    '  "Cellularity_Confidence": <integer 1-10>\n\n'
    "Do NOT include any text outside the JSON object."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_image_paths(data_dir: Path) -> list[Path]:
    """Recursively find all .tif files, ignoring those with 'reti' in the name."""
    paths = []
    for path in sorted(data_dir.rglob("*.tif")):
        if "reti" in path.name.lower():
            continue
        paths.append(path)
    return paths


def parse_path_info(image_path: Path, data_dir: Path) -> dict:
    """Extract Subtype and Patient_ID from the directory structure.

    Expected structure: data/raw/<Subtype>/<Patient_ID>/<image_name>.tif
    """
    rel = image_path.relative_to(data_dir)
    parts = rel.parts  # e.g. ('ET', 'ET12 G1', '1.tif')
    subtype = parts[0] if len(parts) >= 3 else ""
    patient_id = parts[1] if len(parts) >= 3 else ""
    return {
        "Subtype": subtype,
        "Patient_ID": patient_id,
        "Filename": image_path.name,
        "Full_Path": str(image_path),
    }


def process_image(img_path: Path, model) -> dict:
    """Process a single image through the Gemini API and return the info dict."""
    info = parse_path_info(img_path, DATA_DIR)

    uploaded_file = None
    tmp_path = None
    try:
        # Convert TIFF to PNG (Gemini does not support image/tiff)
        img = Image.open(img_path)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(tmp_fd)
        img.save(tmp_path, format="PNG")
        img.close()

        # Upload converted image to Gemini
        uploaded_file = genai.upload_file(tmp_path)

        # Generate content
        response = model.generate_content(
            [
                "Please evaluate this 20x H&E-stained bone marrow biopsy "
                "image and return the 5 expert morphological metrics "
                "with confidence scores.",
                uploaded_file,
            ],
        )

        # Parse the JSON response directly (JSON mode is enforced)
        data = json.loads(response.text)
        for key in METRIC_KEYS:
            info[key] = data.get(key, "")

    except Exception as e:
        tqdm.write(f"    !! Error processing {img_path.name}: {e}")
        for key in METRIC_KEYS:
            info[key] = "ERROR"

    finally:
        # Clean up temporary PNG file
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
        # Clean up uploaded file to avoid filling the API storage quota
        if uploaded_file is not None:
            try:
                uploaded_file.delete()
            except Exception:
                pass

    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract expert morphological metrics from H&E bone "
        "marrow images via Gemini.",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default=None,
        help="Process only a specific subtype (e.g. ET, PV, PMF).",
    )
    parser.add_argument(
        "--num-rois",
        type=int,
        default=None,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="Optional postfix to append to the output filename.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of concurrent API requests (workers). Default is 1 (sequential).",
    )
    args = parser.parse_args()

    # Configure the API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
        },
    )

    # Collect images
    image_paths = collect_image_paths(DATA_DIR)

    # Filter by subtype if specified
    if args.subtype:
        image_paths = [
            p
            for p in image_paths
            if p.relative_to(DATA_DIR).parts[0].lower() == args.subtype.lower()
        ]

    # Limit number of ROIs if specified
    if args.num_rois is not None:
        image_paths = image_paths[: args.num_rois]

    print(f"Found {len(image_paths)} images to process.\n")

    if not image_paths:
        print("No images found. Check the data directory and file structure.")
        return

    # Prepare output directory and CSV path
    output_dir = Path("results/expert_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"expert_metrics_results_{args.postfix}.csv"
        if args.postfix
        else "expert_metrics_results.csv"
    )
    output_csv = output_dir / filename
    fieldnames = ["Subtype", "Patient_ID", "Filename", "Full_Path"] + METRIC_KEYS

    # Resume support: load already-processed paths from existing CSV
    processed_paths: set[str] = set()
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        processed_paths = set(existing_df["Full_Path"].astype(str).tolist())
        print(f"Resuming: {len(processed_paths)} images already processed.\n")

    # Filter out already-processed images
    images_to_process = [p for p in image_paths if str(p) not in processed_paths]
    print(f"{len(images_to_process)} images remaining to process.\n")

    if not images_to_process:
        print("All images already processed. Nothing to do.")
        return

    # Thread-safe CSV writing setup
    csv_lock = threading.Lock()
    write_header = not output_csv.exists()
    csv_file = open(output_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # Process images concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(process_image, img_path, model): img_path
            for img_path in images_to_process
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing",
        ):
            img_path = futures[future]
            try:
                info = future.result()
                if info:
                    with csv_lock:
                        writer.writerow(info)
                        csv_file.flush()
            except Exception as e:
                tqdm.write(f"    !! Unhandled error for {img_path.name}: {e}")

    csv_file.close()
    print(f"\nDone! Results saved to {output_csv}")


if __name__ == "__main__":
    main()
