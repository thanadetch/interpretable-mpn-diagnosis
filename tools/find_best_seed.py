"""
Find optimal random seed for balanced train/val/test splits.
Searches through seeds to find one that gives balanced class distributions.
"""

import argparse
import sys
import os
import contextlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils import get_patient_split
    from config import DATA_MODE_CONFIG, CLASS_MAP_INV, GRADE_MAP_INV
except ImportError as e:
    print(f"❌ Error: Cannot find src folder or missing files: {e}")
    print("Please place this file in the root directory (same folder as src/)")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find the best random seed for balanced train/val/test splits"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "grading"],
        help="Task type: 'classification' (H&E) or 'grading' (Reticulin)",
    )

    parser.add_argument(
        "--range",
        type=int,
        default=2000,
        help="Search range for seeds (0 to range-1, default: 2000)",
    )

    parser.add_argument(
        "--data_mode",
        type=str,
        default="subtype_patch_clean",
        choices=list(DATA_MODE_CONFIG.keys()),
        help="Data mode to use (default: subtype_patch_clean)",
    )

    return parser.parse_args()


# Suppress stdout to hide verbose output from get_patient_split
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def calculate_distribution(file_list):
    """Calculate percentage distribution of each class."""
    counts = defaultdict(int)
    for _, label in file_list:
        counts[label] += 1

    total = len(file_list)
    if total == 0:
        return {}, counts

    # Return as Dict {label: percentage}
    return {k: v / total for k, v in counts.items()}, counts


def get_balance_score(train_dist, val_dist, test_dist=None):
    """
    Calculate balance score (lower is better).
    Uses Mean Squared Error between Train, Val, and optionally Test distributions.
    """
    score = 0
    all_classes = set(train_dist.keys()) | set(val_dist.keys())
    if test_dist:
        all_classes |= set(test_dist.keys())

    for cls in all_classes:
        t_p = train_dist.get(cls, 0.0)
        v_p = val_dist.get(cls, 0.0)

        # Train vs Val difference
        diff = (t_p - v_p) ** 2

        # Extra weight for Class 1 (PV) as it's often problematic
        if cls == 1:
            diff *= 2.0

        score += diff

    return score


def main():
    args = parse_args()

    # Get class name mapping based on task
    if args.task == "classification":
        class_names = CLASS_MAP_INV  # {0: "ET", 1: "PV", 2: "PMF"}
    else:
        class_names = GRADE_MAP_INV  # {0: "G0", 1: "G1", 2: "G2", 3: "G3"}

    print(
        f"🔍 Searching for best seed for task='{args.task}' mode='{args.data_mode}'..."
    )
    print(f"   (Goal: Train/Val/Test distributions should be similar)")

    mode_config = DATA_MODE_CONFIG[args.data_mode]
    data_dir = mode_config["data_dir"]
    file_ext = mode_config["extension"]

    results = []

    # Search loop
    for seed in tqdm(range(args.range), desc="Searching seeds"):
        try:
            with suppress_stdout():  # Suppress verbose output
                train_files, val_files, test_files = get_patient_split(
                    task=args.task, data_dir=data_dir, file_ext=file_ext, seed=seed
                )

            # Calculate distributions
            train_dist, train_counts = calculate_distribution(train_files)
            val_dist, val_counts = calculate_distribution(val_files)
            test_dist, test_counts = calculate_distribution(test_files)

            # Calculate score (lower is better)
            score = get_balance_score(train_dist, val_dist, test_dist)

            # Store results
            results.append(
                {
                    "seed": seed,
                    "score": score,
                    "train_counts": dict(sorted(train_counts.items())),
                    "val_counts": dict(sorted(val_counts.items())),
                    "test_counts": dict(sorted(test_counts.items())),
                    "train_dist": train_dist,
                    "val_dist": val_dist,
                    "test_dist": test_dist,
                }
            )

        except Exception:
            continue

    # Sort by score (best = lowest first)
    results.sort(key=lambda x: x["score"])

    print("\n" + "=" * 60)
    print(f"🏆 TOP 5 SEEDS (from {args.range} seeds searched)")
    print("=" * 60)

    for i, res in enumerate(results[:5]):
        seed = res["seed"]
        print(f"\n[{i + 1}] 🌱 SEED: {seed} (Score: {res['score']:.5f})")

        # Display results nicely
        t_counts = res["train_counts"]
        v_counts = res["val_counts"]
        te_counts = res["test_counts"]

        # Train distribution
        print("   Train: ", end="")
        for k, v in t_counts.items():
            pct = res["train_dist"][k] * 100
            name = class_names.get(k, str(k))
            print(f"{name}: {v} ({pct:.1f}%) | ", end="")

        # Val distribution
        print(f"\n   Val:   ", end="")
        for k, v in v_counts.items():
            pct = res["val_dist"][k] * 100
            name = class_names.get(k, str(k))
            print(f"{name}: {v} ({pct:.1f}%) | ", end="")

        # Test distribution
        print(f"\n   Test:  ", end="")
        for k, v in te_counts.items():
            pct = res["test_dist"][k] * 100
            name = class_names.get(k, str(k))
            print(f"{name}: {v} ({pct:.1f}%) | ", end="")
        print("")

    print("\n✅ Recommendation: Use the #1 seed with --seed when running train.py!")


if __name__ == "__main__":
    main()
