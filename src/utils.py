"""
Utility functions for MPN research framework.
Implements patient-level data splitting and reproducibility helpers.
"""
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import (
    DATA_MODE_CONFIG,
    DEFAULT_DATA_MODE,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    CLASS_MAP,
    GRADE_MAP,
)


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _extract_grade_from_folder(folder_name: str) -> str:
    """
    Extract grade (G0-G3) from patient folder name.

    Args:
        folder_name: Patient folder name (e.g., "ET1 G1", "PMF7 G3")

    Returns:
        Grade string (e.g., "G1", "G3")
    """
    match = re.search(r"G(\d)", folder_name)
    if match:
        return f"G{match.group(1)}"
    raise ValueError(f"Could not extract grade from folder: {folder_name}")


def _get_patient_folders(
    task: str,
    data_dir: Path,
) -> List[Tuple[Path, int]]:
    """
    Get all valid patient folders with their labels.

    Args:
        task: Either 'classification' or 'grading'
        data_dir: Root data directory containing class folders

    Returns:
        List of tuples: (patient_folder_path, label)
    """
    patient_folders: List[Tuple[Path, int]] = []

    for class_name in CLASS_MAP.keys():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        for patient_folder in class_dir.iterdir():
            if not patient_folder.is_dir():
                continue

            # Exclude folders containing "Variety"
            if "Variety" in patient_folder.name:
                continue

            # Determine label based on task
            if task == "classification":
                label = CLASS_MAP[class_name]
            elif task == "grading":
                try:
                    grade = _extract_grade_from_folder(patient_folder.name)
                    label = GRADE_MAP[grade]
                except (ValueError, KeyError):
                    continue
            else:
                raise ValueError(f"Unknown task: {task}. Use 'classification' or 'grading'")

            patient_folders.append((patient_folder, label))

    return patient_folders


def _get_files_from_patient(
    patient_folder: Path,
    task: str,
    file_ext: str,
) -> List[Path]:
    """
    Get filtered image files from a patient folder based on task.

    Args:
        patient_folder: Path to patient folder
        task: Either 'classification' (H&E only) or 'grading' (Reticulin only)
        file_ext: File extension to filter (e.g., 'tif', 'png')

    Returns:
        List of valid image file paths
    """
    files: List[Path] = []

    # Normalize extension (handle with or without dot)
    ext_lower = file_ext.lower().lstrip(".")

    for file_path in patient_folder.iterdir():
        if not file_path.is_file():
            continue

        # Check if file has matching extension
        if file_path.suffix.lower().lstrip(".") != ext_lower:
            continue

        filename_lower = file_path.name.lower()

        # Task-based filtering
        if task == "classification":
            # H&E images only: exclude 'reti' files
            if "reti" not in filename_lower:
                files.append(file_path)
        elif task == "grading":
            # Reticulin images only: keep ONLY 'reti' files
            if "reti" in filename_lower:
                files.append(file_path)

    return files


def get_patient_split(
    task: str,
    data_dir: Path = None,
    file_ext: str = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> Tuple[
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
]:
    """
    Split data at PATIENT level to prevent data leakage.

    This function ensures that all patches from the same patient
    are in the same subset (train/val/test).

    Args:
        task: Either 'classification' or 'grading'
        data_dir: Root data directory containing class folders (default: from config)
        file_ext: File extension to filter (default: from config)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files, test_files)
        Each is a list of tuples: (file_path, label)
    """
    # Apply defaults from config if not provided
    if data_dir is None:
        data_dir = DATA_MODE_CONFIG[DEFAULT_DATA_MODE]["data_dir"]
    if file_ext is None:
        file_ext = DATA_MODE_CONFIG[DEFAULT_DATA_MODE]["extension"]

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Get all patient folders with labels
    patient_folders = _get_patient_folders(task, data_dir)

    if len(patient_folders) == 0:
        raise ValueError(f"No valid patient folders found for task: {task}")

    # Extract paths and labels for stratification
    folders = [pf[0] for pf in patient_folders]
    labels = [pf[1] for pf in patient_folders]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio

    try:
        # Attempt stratified split
        train_folders, val_test_folders, train_labels, val_test_labels = train_test_split(
            folders,
            labels,
            test_size=val_test_ratio,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        # Fall back to non-stratified split if classes are too small
        print("Warning: Could not perform stratified split. Using random split.")
        train_folders, val_test_folders, train_labels, val_test_labels = train_test_split(
            folders,
            labels,
            test_size=val_test_ratio,
            random_state=seed,
        )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio

    try:
        val_folders, test_folders, _, _ = train_test_split(
            val_test_folders,
            val_test_labels,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=val_test_labels,
        )
    except ValueError:
        val_folders, test_folders, _, _ = train_test_split(
            val_test_folders,
            val_test_labels,
            test_size=relative_test_ratio,
            random_state=seed,
        )

    # Convert folders to file paths
    def folders_to_files(folder_list: List[Path]) -> List[Tuple[Path, int]]:
        """Convert patient folders to individual file paths with labels."""
        files: List[Tuple[Path, int]] = []

        # Create folder to label mapping
        folder_to_label = {pf[0]: pf[1] for pf in patient_folders}

        for folder in folder_list:
            label = folder_to_label[folder]
            patient_files = _get_files_from_patient(folder, task, file_ext)

            for file_path in patient_files:
                files.append((file_path, label))

        return files

    train_files = folders_to_files(train_folders)
    val_files = folders_to_files(val_folders)
    test_files = folders_to_files(test_folders)

    # Print split statistics
    print(f"\n{'='*60}")
    print(f"Data Split Statistics (Task: {task})")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"File extension: .{file_ext}")
    print(f"Total patients: {len(patient_folders)}")
    print(f"Train patients: {len(train_folders)} | Images: {len(train_files)}")
    print(f"Val patients:   {len(val_folders)} | Images: {len(val_files)}")
    print(f"Test patients:  {len(test_folders)} | Images: {len(test_files)}")
    print(f"{'='*60}\n")

    return train_files, val_files, test_files


def get_class_weights(
    file_list: List[Tuple[Path, int]],
    num_classes: int,
) -> torch.Tensor:
    """
    Calculate class weights for WeightedRandomSampler.

    Args:
        file_list: List of (file_path, label) tuples
        num_classes: Number of classes

    Returns:
        Tensor of sample weights (one weight per sample)
    """
    # Count samples per class
    class_counts = [0] * num_classes
    for _, label in file_list:
        class_counts[label] += 1

    # Calculate inverse frequency weights
    class_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for _, label in file_list]

    return torch.tensor(sample_weights, dtype=torch.float64)


def get_num_classes(task: str) -> int:
    """
    Get number of classes based on task.

    Args:
        task: Either 'classification' or 'grading'

    Returns:
        Number of classes
    """
    if task == "classification":
        return len(CLASS_MAP)
    elif task == "grading":
        return len(GRADE_MAP)
    else:
        raise ValueError(f"Unknown task: {task}")

