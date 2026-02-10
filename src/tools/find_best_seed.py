"""
Golden Seed Search - Find optimal random seeds for balanced train/val/test splits.

Uses Strategy Pattern to handle different validation requirements:
    - GradingSeedFinder: For G0-G3 task, focuses on G3 minority class constraints
    - SubtypeSeedFinder: For ET/PV/PMF task, ensures stratified distribution

Usage:
    python tools/find_best_seed.py --task grading --data_mode grading_patch_clean
    python tools/find_best_seed.py --task classification --data_mode subtype_patch_clean
"""

import argparse
import contextlib
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path

# Ensure src/ is on sys.path when running directly (e.g., python src/tools/data_stats.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CLASS_MAP_INV, DATA_MODE_CONFIG, GRADE_MAP_INV
from core.utils import get_patient_split


# Utility Functions
# ============================================================================


@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout to hide verbose output from get_patient_split."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_class_counts(file_list: List[Tuple[str, int]]) -> Dict[int, int]:
    """Get counts for all classes in file list."""
    counts = defaultdict(int)
    for _, label in file_list:
        counts[label] += 1
    return dict(sorted(counts.items()))


def get_class_distribution(file_list: List[Tuple[str, int]]) -> Dict[int, float]:
    """Get percentage distribution of each class."""
    counts = get_class_counts(file_list)
    total = len(file_list)
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


# ============================================================================
# Abstract Base Class: SeedFinder
# ============================================================================


class SeedFinder(ABC):
    """
    Abstract base class for seed finding strategies.

    Subclasses must implement:
        - validate_split(): Check if a split satisfies constraints
        - get_result_summary(): Format result details for output
        - print_constraints(): Display the validation rules
    """

    def __init__(self, task: str, data_mode: str, search_range: int, max_seeds: int):
        self.task = task
        self.data_mode = data_mode
        self.search_range = search_range
        self.max_seeds = max_seeds
        self.class_names = self._get_class_names()

        # Get data configuration
        mode_config = DATA_MODE_CONFIG[data_mode]
        self.data_dir = mode_config["data_dir"]
        self.file_ext = mode_config["extension"]

    def _get_class_names(self) -> Dict[int, str]:
        """Get class name mapping based on task."""
        if self.task == "classification":
            return CLASS_MAP_INV
        return GRADE_MAP_INV

    @abstractmethod
    def validate_split(
            self,
            train_files: List[Tuple[str, int]],
            val_files: List[Tuple[str, int]],
            test_files: List[Tuple[str, int]],
    ) -> Tuple[bool, Dict]:
        """
        Validate if the split satisfies task-specific constraints.

        Returns:
            Tuple of (is_valid, result_dict with details)
        """
        pass

    @abstractmethod
    def print_constraints(self) -> None:
        """Print the validation constraints for this strategy."""
        pass

    @abstractmethod
    def print_result_details(self, result: Dict, index: int) -> None:
        """Print detailed results for a valid seed."""
        pass

    def search(self) -> List[Dict]:
        """
        Main search loop - common for all strategies.

        Returns:
            List of valid seed results
        """
        print("=" * 70)
        print(f"🔍 GOLDEN SEED SEARCH - {self.__class__.__name__}")
        print("=" * 70)
        print(f"   Task:      {self.task}")
        print(f"   Data Mode: {self.data_mode}")
        print(f"   Searching: seeds 0 to {self.search_range - 1}")
        print(f"   Target:    Find {self.max_seeds} valid seeds")
        print()
        self.print_constraints()
        print("=" * 70)

        valid_seeds = []

        pbar = tqdm(range(self.search_range), desc="Searching seeds")
        for seed in pbar:
            try:
                with suppress_stdout():
                    train_files, val_files, test_files = get_patient_split(
                        task=self.task,
                        data_dir=self.data_dir,
                        file_ext=self.file_ext,
                        seed=seed,
                    )

                is_valid, result = self.validate_split(
                    train_files, val_files, test_files
                )

                if is_valid:
                    result["seed"] = seed
                    result["train_counts"] = get_class_counts(train_files)
                    result["val_counts"] = get_class_counts(val_files)
                    result["test_counts"] = get_class_counts(test_files)
                    result["train_dist"] = get_class_distribution(train_files)
                    result["val_dist"] = get_class_distribution(val_files)
                    result["test_dist"] = get_class_distribution(test_files)
                    valid_seeds.append(result)

                    pbar.set_postfix({"found": len(valid_seeds)})

                    if len(valid_seeds) >= self.max_seeds:
                        print(
                            f"\n\n✅ Found {self.max_seeds} valid seeds! Stopping early."
                        )
                        break

            except Exception:
                continue

        return valid_seeds

    def print_results(self, valid_seeds: List[Dict]) -> None:
        """Print all valid seed results."""
        print("\n" + "=" * 70)

        if len(valid_seeds) == 0:
            print("❌ NO VALID SEEDS FOUND!")
            print("   Consider relaxing constraints or increasing --range")
            print("=" * 70)
            return

        print(f"🏆 FOUND {len(valid_seeds)} GOLDEN SEED(S)")
        print("=" * 70)

        for i, result in enumerate(valid_seeds):
            self.print_result_details(result, i + 1)

        print("\n" + "=" * 70)
        print(
            f"✅ Recommendation: Use --seed {valid_seeds[0]['seed']} when running train.py!"
        )
        print("=" * 70)

    def _print_full_distribution(self, result: Dict) -> None:
        """Helper to print full class distribution table."""
        col_width = 8

        # Header
        print(f"    {'Split':<8}", end="")
        for label in sorted(result["train_counts"].keys()):
            name = self.class_names.get(label, str(label))
            print(f"{name:>{col_width}}", end="")
        print(f"{'Total':>{col_width}}")

        # Train row
        print(f"    {'Train':<8}", end="")
        for label in sorted(result["train_counts"].keys()):
            count = result["train_counts"].get(label, 0)
            print(f"{count:>{col_width}}", end="")
        print(f"{sum(result['train_counts'].values()):>{col_width}}")

        # Val row
        print(f"    {'Val':<8}", end="")
        for label in sorted(result["val_counts"].keys()):
            count = result["val_counts"].get(label, 0)
            print(f"{count:>{col_width}}", end="")
        print(f"{sum(result['val_counts'].values()):>{col_width}}")

        # Test row
        print(f"    {'Test':<8}", end="")
        for label in sorted(result["test_counts"].keys()):
            count = result["test_counts"].get(label, 0)
            print(f"{count:>{col_width}}", end="")
        print(f"{sum(result['test_counts'].values()):>{col_width}}")


# ============================================================================
# Strategy 1: GradingSeedFinder (G0-G3, focus on G3 minority class)
# ============================================================================


class GradingSeedFinder(SeedFinder):
    """
    Seed finder for grading task (G0-G3).

    Focuses on ensuring the minority class (G3) is properly distributed
    to avoid Train having too few G3 samples.
    """

    # Hard Constraints for G3
    MIN_TRAIN_G3_RATIO = 0.55  # Train must have >= 55% of total G3
    MAX_TEST_G3_RATIO = 0.25  # Test must have <= 25% of total G3
    MIN_VAL_G3_COUNT = 1  # Val must have at least 1 G3 sample
    MINORITY_CLASS = 3  # G3 label

    def print_constraints(self) -> None:
        print("   Constraints (G3 Minority Class Protection):")
        print(f"     ✓ Train must have >= {self.MIN_TRAIN_G3_RATIO:.0%} of all G3")
        print(f"     ✓ Test must have  <= {self.MAX_TEST_G3_RATIO:.0%} of all G3")
        print(f"     ✓ Val must have   >= {self.MIN_VAL_G3_COUNT} G3 sample(s)")

    def validate_split(
            self,
            train_files: List[Tuple[str, int]],
            val_files: List[Tuple[str, int]],
            test_files: List[Tuple[str, int]],
    ) -> Tuple[bool, Dict]:
        train_g3 = sum(1 for _, l in train_files if l == self.MINORITY_CLASS)
        val_g3 = sum(1 for _, l in val_files if l == self.MINORITY_CLASS)
        test_g3 = sum(1 for _, l in test_files if l == self.MINORITY_CLASS)
        total_g3 = train_g3 + val_g3 + test_g3

        if total_g3 == 0:
            return False, {}

        train_pct = train_g3 / total_g3
        test_pct = test_g3 / total_g3

        is_valid = (
                train_pct >= self.MIN_TRAIN_G3_RATIO
                and test_pct <= self.MAX_TEST_G3_RATIO
                and val_g3 >= self.MIN_VAL_G3_COUNT
        )

        return is_valid, {
            "train_g3": train_g3,
            "val_g3": val_g3,
            "test_g3": test_g3,
            "total_g3": total_g3,
            "train_g3_pct": train_pct,
            "test_g3_pct": test_pct,
        }

    def print_result_details(self, result: Dict, index: int) -> None:
        print(f"\n[{index}] 🌱 SEED: {result['seed']}")
        print(f"    G3 Distribution (minority class):")
        print(f"      Train: {result['train_g3']:>5} ({result['train_g3_pct']:.1%})")
        print(f"      Val:   {result['val_g3']:>5}")
        print(f"      Test:  {result['test_g3']:>5} ({result['test_g3_pct']:.1%})")
        print(f"      Total: {result['total_g3']:>5}")
        print(f"    Full Distribution:")
        self._print_full_distribution(result)


# ============================================================================
# Strategy 2: SubtypeSeedFinder (ET/PV/PMF, stratified split)
# ============================================================================


class SubtypeSeedFinder(SeedFinder):
    """
    Seed finder for classification/subtype task (ET, PV, PMF).

    Ensures stratified split where each class maintains similar proportions
    across Train/Val/Test sets. Prevents scenarios like Train=28% ET, Test=8% ET.
    """

    # Stratification Constraints
    MIN_TEST_CLASS_RATIO = 0.15  # Each class must be >= 15% of test set
    MAX_TEST_CLASS_RATIO = 0.35  # Each class must be <= 35% of test set
    MAX_DRIFT = 0.15  # Max difference between train and test proportions

    def print_constraints(self) -> None:
        print("   Constraints (Stratified Split for ET/PV/PMF):")
        print(
            f"     ✓ Each class in Test must be between {self.MIN_TEST_CLASS_RATIO:.0%}-{self.MAX_TEST_CLASS_RATIO:.0%}"
        )
        print(f"     ✓ Train/Test distribution drift <= {self.MAX_DRIFT:.0%} per class")

    def validate_split(
            self,
            train_files: List[Tuple[str, int]],
            val_files: List[Tuple[str, int]],
            test_files: List[Tuple[str, int]],
    ) -> Tuple[bool, Dict]:
        train_dist = get_class_distribution(train_files)
        test_dist = get_class_distribution(test_files)

        all_classes = set(train_dist.keys()) | set(test_dist.keys())

        # Check constraints for each class
        violations = []
        max_drift = 0.0

        for cls in all_classes:
            test_pct = test_dist.get(cls, 0.0)
            train_pct = train_dist.get(cls, 0.0)
            drift = abs(train_pct - test_pct)
            max_drift = max(max_drift, drift)

            # Check test proportion bounds
            if test_pct < self.MIN_TEST_CLASS_RATIO:
                violations.append(
                    f"{self.class_names.get(cls, cls)}: Test={test_pct:.1%} < {self.MIN_TEST_CLASS_RATIO:.0%}"
                )
            if test_pct > self.MAX_TEST_CLASS_RATIO:
                violations.append(
                    f"{self.class_names.get(cls, cls)}: Test={test_pct:.1%} > {self.MAX_TEST_CLASS_RATIO:.0%}"
                )

            # Check drift
            if drift > self.MAX_DRIFT:
                violations.append(
                    f"{self.class_names.get(cls, cls)}: Drift={drift:.1%} > {self.MAX_DRIFT:.0%}"
                )

        is_valid = len(violations) == 0

        return is_valid, {
            "max_drift": max_drift,
            "violations": violations,
        }

    def print_result_details(self, result: Dict, index: int) -> None:
        print(f"\n[{index}] 🌱 SEED: {result['seed']}")
        print(f"    Max Drift: {result['max_drift']:.1%}")
        print(f"    Class Proportions (Train vs Test):")

        for cls in sorted(result["train_dist"].keys()):
            name = self.class_names.get(cls, str(cls))
            train_pct = result["train_dist"].get(cls, 0.0)
            test_pct = result["test_dist"].get(cls, 0.0)
            drift = abs(train_pct - test_pct)
            print(
                f"      {name}: Train={train_pct:.1%}, Test={test_pct:.1%} (drift={drift:.1%})"
            )

        print(f"    Full Distribution:")
        self._print_full_distribution(result)


# ============================================================================
# Main Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Golden Seed Search - Find optimal seeds for balanced splits"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "grading"],
        help="Task: 'classification' (ET/PV/PMF) or 'grading' (G0-G3)",
    )

    parser.add_argument(
        "--data_mode",
        type=str,
        required=True,
        choices=list(DATA_MODE_CONFIG.keys()),
        help="Data mode configuration to use",
    )

    parser.add_argument(
        "--range",
        type=int,
        default=5000,
        help="Search range for seeds (default: 5000)",
    )

    parser.add_argument(
        "--max_seeds",
        type=int,
        default=5,
        help="Stop after finding this many valid seeds (default: 5)",
    )

    return parser.parse_args()


def get_seed_finder(args: argparse.Namespace) -> SeedFinder:
    """Factory function to create the appropriate SeedFinder strategy."""
    if args.task == "grading":
        return GradingSeedFinder(
            task=args.task,
            data_mode=args.data_mode,
            search_range=args.range,
            max_seeds=args.max_seeds,
        )
    else:  # classification / subtype
        return SubtypeSeedFinder(
            task=args.task,
            data_mode=args.data_mode,
            search_range=args.range,
            max_seeds=args.max_seeds,
        )


def main():
    args = parse_args()

    # Create the appropriate strategy
    finder = get_seed_finder(args)

    # Run search
    valid_seeds = finder.search()

    # Print results
    finder.print_results(valid_seeds)


if __name__ == "__main__":
    main()
