#!/usr/bin/env python3
"""
Dataset Statistics Script for MPN Classification and Fibrosis Grading.

Analyzes raw data and processed patches, providing summary statistics
for both subtype (H&E) and grading (Reticulin) tasks.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure src/ is on sys.path when running directly (e.g., python src/tools/data_stats.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    RAW_DATA_DIR,
    CLASS_MAP,
    GRADE_MAP,
    IMAGE_EXTENSIONS,
)


def count_raw_he_images(raw_dir: Path) -> dict:
    """
    Count patients and H&E images grouped by subtype (ET, PV, PMF).

    H&E images are identified as files that do NOT start with 'reti'.

    Returns:
        dict: {subtype: {'patients': int, 'images': int, 'patient_dict': {name: count}}}
    """
    stats = {
        subtype: {"patients": 0, "images": 0, "patient_dict": {}}
        for subtype in CLASS_MAP.keys()
    }

    for subtype in CLASS_MAP.keys():
        subtype_dir = raw_dir / subtype
        if not subtype_dir.exists():
            continue

        patient_dirs = [d for d in subtype_dir.iterdir() if d.is_dir()]
        stats[subtype]["patients"] = len(patient_dirs)

        image_count = 0
        patient_dict = {}
        for patient_dir in patient_dirs:
            patient_images = 0
            for ext in IMAGE_EXTENSIONS:
                # Count only H&E images (NOT starting with 'reti')
                he_images = [
                    f
                    for f in patient_dir.glob(f"*{ext}")
                    if not f.name.lower().startswith("reti")
                ]
                patient_images += len(he_images)
            patient_dict[patient_dir.name] = patient_images
            image_count += patient_images
        stats[subtype]["images"] = image_count
        stats[subtype]["patient_dict"] = patient_dict

    return stats


def count_raw_reticulin_images(raw_dir: Path) -> dict:
    """
    Count patients and Reticulin images grouped by grade (G0-G3).

    Reticulin images are identified as files starting with 'reti'.
    Grade is extracted from the patient folder name (e.g., 'ET1 G1').

    Returns:
        dict: {grade: {'patients': set, 'images': int}}
    """
    stats = {grade: {"patients": set(), "images": 0} for grade in GRADE_MAP.keys()}
    grade_pattern = re.compile(r"\b(G[0-3])\b", re.IGNORECASE)

    for subtype in CLASS_MAP.keys():
        subtype_dir = raw_dir / subtype
        if not subtype_dir.exists():
            continue

        patient_dirs = [d for d in subtype_dir.iterdir() if d.is_dir()]

        for patient_dir in patient_dirs:
            # Extract grade from folder name
            match = grade_pattern.search(patient_dir.name)
            if not match:
                continue
            grade = match.group(1).upper()

            if grade not in stats:
                continue

            # Count reticulin images
            reti_count = 0
            for ext in IMAGE_EXTENSIONS:
                reti_images = [
                    f
                    for f in patient_dir.glob(f"*{ext}")
                    if f.name.lower().startswith("reti")
                ]
                reti_count += len(reti_images)

            if reti_count > 0:
                stats[grade]["patients"].add(patient_dir.name)
                stats[grade]["images"] += reti_count

    # Convert sets to counts
    return {
        grade: {"patients": len(data["patients"]), "images": data["images"]}
        for grade, data in stats.items()
    }


def count_patches(processed_dir: Path) -> dict:
    """
    Count patches in a processed directory.

    Directory structure: processed_dir/CLASS/PATIENT/*.png

    Returns:
        dict: {class_name: patch_count}
    """
    stats = defaultdict(int)

    if not processed_dir.exists():
        return dict(stats)

    for class_dir in processed_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        patch_count = 0
        for patient_dir in class_dir.iterdir():
            if not patient_dir.is_dir() or patient_dir.name.startswith("."):
                continue
            patch_count += len(list(patient_dir.glob("*.png")))

        stats[class_dir.name] = patch_count

    return dict(stats)


def count_grading_patches(processed_dir: Path) -> dict:
    """
    Count patches for grading task, grouped by grade.

    Grade is extracted from patient folder name.

    Returns:
        dict: {grade: patch_count}
    """
    stats = {grade: 0 for grade in GRADE_MAP.keys()}
    grade_pattern = re.compile(r"\b(G[0-3])\b", re.IGNORECASE)

    if not processed_dir.exists():
        return stats

    for class_dir in processed_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        for patient_dir in class_dir.iterdir():
            if not patient_dir.is_dir() or patient_dir.name.startswith("."):
                continue

            match = grade_pattern.search(patient_dir.name)
            if not match:
                continue
            grade = match.group(1).upper()

            if grade in stats:
                stats[grade] += len(list(patient_dir.glob("*.png")))

    return stats


def print_table(title: str, headers: list, rows: list, col_widths: list = None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [
            max(len(str(row[i])) for row in [headers] + rows) + 2
            for i in range(len(headers))
        ]

    # Title
    total_width = sum(col_widths) + len(col_widths) - 1
    print("\n" + "=" * total_width)
    print(title.center(total_width))
    print("=" * total_width)

    # Header
    header_row = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * total_width)

    # Separate caller-provided "Total" row from data rows
    caller_total = None
    if len(rows) > 1 and str(rows[-1][0]).strip() == "Total":
        caller_total = rows[-1]
        rows = rows[:-1]

    # Data rows
    for row in rows:
        row_str = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        print(row_str)

    # Total row (auto-compute or use caller-provided)
    if len(rows) > 1 or caller_total:
        print("-" * total_width)
        if caller_total:
            totals = caller_total
        else:
            totals = ["Total"]
            for i in range(1, len(headers)):
                try:
                    totals.append(sum(row[i] for row in rows))
                except TypeError:
                    totals.append("")
        total_row = "|".join(str(cell).center(w) for cell, w in zip(totals, col_widths))
        print(total_row)

    print("=" * total_width)


def calculate_actual_split(features_dir: Path, seed: int = 42) -> tuple:
    """
    Calculate the actual train/val/test split using the same logic as train_mil.py.

    Returns:
        (headers, rows) suitable for print_table.
    """
    from data.bag_dataset import MPNBagDatasetFull
    from train_mil import patient_split
    from core.config import CLASS_MAP_INV

    dataset = MPNBagDatasetFull(features_dir)
    train_idx, val_idx, test_idx = patient_split(dataset, seed=seed)

    # Collect stats per split per label
    stats = {
        split: {label: {"patients": set(), "images": 0} for label in CLASS_MAP_INV}
        for split in ["Train", "Val", "Test"]
    }

    for split_name, indices in [
        ("Train", train_idx),
        ("Val", val_idx),
        ("Test", test_idx),
    ]:
        for idx in indices:
            pt_path, label = dataset.samples[idx]
            patient_id = pt_path.parent.name
            stats[split_name][label]["patients"].add(patient_id)
            stats[split_name][label]["images"] += 1

    # Build pivot table
    headers = [
        "Subtype",
        "Train (Pat/Img)",
        "Val (Pat/Img)",
        "Test (Pat/Img)",
        "Total (Pat/Img)",
    ]
    rows = []
    totals = {"Train": [0, 0], "Val": [0, 0], "Test": [0, 0]}

    for label in sorted(CLASS_MAP_INV.keys()):
        subtype = CLASS_MAP_INV[label]
        cells = {}
        total_pat, total_img = 0, 0
        for split_name in ["Train", "Val", "Test"]:
            pat = len(stats[split_name][label]["patients"])
            img = stats[split_name][label]["images"]
            cells[split_name] = f"{pat} ({img})"
            totals[split_name][0] += pat
            totals[split_name][1] += img
            total_pat += pat
            total_img += img

        rows.append(
            [
                subtype,
                cells["Train"],
                cells["Val"],
                cells["Test"],
                f"{total_pat} ({total_img})",
            ]
        )

    # Grand total row
    grand_pat = sum(v[0] for v in totals.values())
    grand_img = sum(v[1] for v in totals.values())
    rows.append(
        [
            "Total",
            f"{totals['Train'][0]} ({totals['Train'][1]})",
            f"{totals['Val'][0]} ({totals['Val'][1]})",
            f"{totals['Test'][0]} ({totals['Test'][1]})",
            f"{grand_pat} ({grand_img})",
        ]
    )

    return headers, rows


def calculate_actual_grading_split(features_dir: Path, seed: int = 42) -> tuple:
    """
    Calculate the actual grading train/val/test split using the same logic
    as train_grading_reti.py.

    Returns:
        (headers, rows) suitable for print_table.
    """
    from data.bag_dataset import GradingBagDatasetFull
    from train_grading_reti import patient_split

    LABEL_MAP = {0: "G0", 1: "G1", 2: "G2", 3: "G3"}

    dataset = GradingBagDatasetFull(features_dir)
    train_idx, val_idx, test_idx = patient_split(dataset, seed=seed)

    # Collect stats per split per label
    stats = {
        split: {label: {"patients": set(), "images": 0} for label in LABEL_MAP}
        for split in ["Train", "Val", "Test"]
    }

    for split_name, indices in [
        ("Train", train_idx),
        ("Val", val_idx),
        ("Test", test_idx),
    ]:
        for idx in indices:
            pt_path, label = dataset.samples[idx]
            patient_id = pt_path.parent.name
            stats[split_name][label]["patients"].add(patient_id)
            stats[split_name][label]["images"] += 1

    # Build pivot table
    headers = [
        "Grade",
        "Train (Pat/Img)",
        "Val (Pat/Img)",
        "Test (Pat/Img)",
        "Total (Pat/Img)",
    ]
    rows = []
    totals = {"Train": [0, 0], "Val": [0, 0], "Test": [0, 0]}

    for label in sorted(LABEL_MAP.keys()):
        grade = LABEL_MAP[label]
        cells = {}
        total_pat, total_img = 0, 0
        for split_name in ["Train", "Val", "Test"]:
            pat = len(stats[split_name][label]["patients"])
            img = stats[split_name][label]["images"]
            cells[split_name] = f"{pat} ({img})"
            totals[split_name][0] += pat
            totals[split_name][1] += img
            total_pat += pat
            total_img += img

        rows.append(
            [
                grade,
                cells["Train"],
                cells["Val"],
                cells["Test"],
                f"{total_pat} ({total_img})",
            ]
        )

    # Grand total row
    grand_pat = sum(v[0] for v in totals.values())
    grand_img = sum(v[1] for v in totals.values())
    rows.append(
        [
            "Total",
            f"{totals['Train'][0]} ({totals['Train'][1]})",
            f"{totals['Val'][0]} ({totals['Val'][1]})",
            f"{totals['Test'][0]} ({totals['Test'][1]})",
            f"{grand_pat} ({grand_img})",
        ]
    )

    return headers, rows


def main():
    """Main entry point for dataset statistics."""
    parser = argparse.ArgumentParser(description="MPN Dataset Statistics")
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Explicit path to the features directory for split calculation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test splitting (default: 42).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MPN Dataset Statistics".center(70))
    print("=" * 70)

    # ==================================================================
    # 1. Raw Data Analysis - H&E Images (Subtype Classification)
    # ==================================================================
    he_stats = count_raw_he_images(RAW_DATA_DIR)

    headers = ["Subtype", "Patients", "Raw Images"]
    rows = []
    for subtype in CLASS_MAP.keys():
        rows.append(
            [
                subtype,
                he_stats.get(subtype, {}).get("patients", 0),
                he_stats.get(subtype, {}).get("images", 0),
            ]
        )

    print_table("H&E Subtype Classification (ET, PV, PMF)", headers, rows)

    # ==================================================================
    # 1b. Actual Train/Val/Test Split (from train_mil.py)
    # ==================================================================
    features_dir = None

    if args.features_dir:
        candidate = Path(args.features_dir)
        if candidate.exists() and candidate.is_dir():
            features_dir = candidate
        else:
            print(
                f"\n  ⚠️  Specified features directory '{args.features_dir}' not found. Skipping split statistics."
            )
    else:
        data_root = Path("data")
        if data_root.exists():
            feature_dirs = sorted(
                [
                    d
                    for d in data_root.iterdir()
                    if d.is_dir() and d.name.startswith("features_")
                ]
            )
            if feature_dirs:
                features_dir = feature_dirs[0]

    if features_dir is not None:
        print(f"\n  ℹ️  Using '{features_dir}' as reference for split statistics.")
        split_headers, split_rows = calculate_actual_split(features_dir, seed=args.seed)
        print_table(
            f"ACTUAL Data Split (Sourced from train_mil.py, Seed={args.seed}, Ref: {features_dir.name})",
            split_headers,
            split_rows,
        )
    elif not args.features_dir:
        print(
            "\n  ⚠️  No 'features_*' directories found in data/. Skipping split statistics."
        )

    # ==================================================================
    # 2. Raw Data Analysis - Reticulin Images (Fibrosis Grading)
    # ==================================================================
    reti_stats = count_raw_reticulin_images(RAW_DATA_DIR)

    headers = ["Grade", "Patients", "Raw Images"]
    rows = []
    for grade in GRADE_MAP.keys():
        rows.append(
            [
                grade,
                reti_stats.get(grade, {}).get("patients", 0),
                reti_stats.get(grade, {}).get("images", 0),
            ]
        )

    print_table("Reticulin Fibrosis Grading (G0-G3)", headers, rows)

    # ==================================================================
    # 2b. Actual Train/Val/Test Split for Grading (from train_grading_reti.py)
    # ==================================================================
    if features_dir is not None:
        grading_headers, grading_rows = calculate_actual_grading_split(
            features_dir, seed=args.seed
        )
        print_table(
            f"ACTUAL Grading Data Split (Sourced from train_grading_reti.py, Seed={args.seed}, Ref: {features_dir.name})",
            grading_headers,
            grading_rows,
        )

    # ==================================================================
    # 3. Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("Summary".center(70))
    print("=" * 70)

    total_he_patients = sum(s.get("patients", 0) for s in he_stats.values())
    total_he_images = sum(s.get("images", 0) for s in he_stats.values())
    total_reti_patients = sum(s.get("patients", 0) for s in reti_stats.values())
    total_reti_images = sum(s.get("images", 0) for s in reti_stats.values())

    print(
        f"  H&E (Subtype):      {total_he_patients:>4} patients, {total_he_images:>5} images"
    )
    print(
        f"  Reticulin (Grading): {total_reti_patients:>4} patients, {total_reti_images:>5} images"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
