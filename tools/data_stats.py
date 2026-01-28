#!/usr/bin/env python3
"""
Dataset Statistics Script for MPN Classification and Fibrosis Grading.

Analyzes raw data and processed patches, providing summary statistics
for both subtype (H&E) and grading (Reticulin) tasks.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_SUBTYPE_DIR,
    PROCESSED_SUBTYPE_CLEAN_DIR,
    PROCESSED_GRADING_DIR,
    PROCESSED_GRADING_CLEAN_DIR,
    CLASS_MAP,
    GRADE_MAP,
    IMAGE_EXTENSIONS,
)


def count_raw_he_images(raw_dir: Path) -> dict:
    """
    Count patients and H&E images grouped by subtype (ET, PV, PMF).
    
    H&E images are identified as files that do NOT start with 'reti'.
    
    Returns:
        dict: {subtype: {'patients': int, 'images': int}}
    """
    stats = {subtype: {"patients": 0, "images": 0} for subtype in CLASS_MAP.keys()}

    for subtype in CLASS_MAP.keys():
        subtype_dir = raw_dir / subtype
        if not subtype_dir.exists():
            continue

        patient_dirs = [d for d in subtype_dir.iterdir() if d.is_dir()]
        stats[subtype]["patients"] = len(patient_dirs)

        image_count = 0
        for patient_dir in patient_dirs:
            for ext in IMAGE_EXTENSIONS:
                # Count only H&E images (NOT starting with 'reti')
                he_images = [
                    f for f in patient_dir.glob(f"*{ext}")
                    if not f.name.lower().startswith("reti")
                ]
                image_count += len(he_images)
        stats[subtype]["images"] = image_count

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
                    f for f in patient_dir.glob(f"*{ext}")
                    if f.name.lower().startswith("reti")
                ]
                reti_count += len(reti_images)

            if reti_count > 0:
                stats[grade]["patients"].add(patient_dir.name)
                stats[grade]["images"] += reti_count

    # Convert sets to counts
    return {grade: {"patients": len(data["patients"]), "images": data["images"]}
            for grade, data in stats.items()}


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
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]

    # Title
    total_width = sum(col_widths) + len(col_widths) - 1
    print("\n" + "=" * total_width)
    print(title.center(total_width))
    print("=" * total_width)

    # Header
    header_row = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * total_width)

    # Rows
    for row in rows:
        row_str = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        print(row_str)

    # Total row
    if len(rows) > 1:
        print("-" * total_width)
        totals = ["Total"]
        for i in range(1, len(headers)):
            totals.append(sum(row[i] for row in rows))
        total_row = "|".join(str(cell).center(w) for cell, w in zip(totals, col_widths))
        print(total_row)

    print("=" * total_width)


def main():
    """Main entry point for dataset statistics."""
    print("\n" + "=" * 70)
    print("MPN Dataset Statistics".center(70))
    print("=" * 70)

    # ==================================================================
    # 1. Raw Data Analysis - H&E Images (Subtype Classification)
    # ==================================================================
    he_stats = count_raw_he_images(RAW_DATA_DIR)
    subtype_patches = count_patches(PROCESSED_SUBTYPE_DIR)
    subtype_patches_clean = count_patches(PROCESSED_SUBTYPE_CLEAN_DIR)

    headers = ["Subtype", "Patients", "Raw Images", "Patches (Orig)", "Patches (Clean)"]
    rows = []
    for subtype in CLASS_MAP.keys():
        rows.append([
            subtype,
            he_stats.get(subtype, {}).get("patients", 0),
            he_stats.get(subtype, {}).get("images", 0),
            subtype_patches.get(subtype, 0),
            subtype_patches_clean.get(subtype, 0),
        ])

    print_table("H&E Subtype Classification (ET, PV, PMF)", headers, rows)

    # ==================================================================
    # 2. Raw Data Analysis - Reticulin Images (Fibrosis Grading)
    # ==================================================================
    reti_stats = count_raw_reticulin_images(RAW_DATA_DIR)
    grading_patches = count_grading_patches(PROCESSED_GRADING_DIR)
    grading_patches_clean = count_grading_patches(PROCESSED_GRADING_CLEAN_DIR)

    headers = ["Grade", "Patients", "Raw Images", "Patches (Orig)", "Patches (Clean)"]
    rows = []
    for grade in GRADE_MAP.keys():
        rows.append([
            grade,
            reti_stats.get(grade, {}).get("patients", 0),
            reti_stats.get(grade, {}).get("images", 0),
            grading_patches.get(grade, 0),
            grading_patches_clean.get(grade, 0),
        ])

    print_table("Reticulin Fibrosis Grading (G0-G3)", headers, rows)

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

    print(f"  H&E (Subtype):      {total_he_patients:>4} patients, {total_he_images:>5} images")
    print(f"  Reticulin (Grading): {total_reti_patients:>4} patients, {total_reti_images:>5} images")
    print()
    print(
        f"  Subtype Patches:     {sum(subtype_patches.values()):>6} original, {sum(subtype_patches_clean.values()):>6} clean")
    print(
        f"  Grading Patches:     {sum(grading_patches.values()):>6} original, {sum(grading_patches_clean.values()):>6} clean")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
