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
import torch.nn as nn
import torch.nn.functional as F

from core.config import (
    DATA_MODE_CONFIG,
    DEFAULT_DATA_MODE,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    CLASS_MAP,
    CLASS_MAP_INV,
    GRADE_MAP,
    GRADE_MAP_INV,
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
                raise ValueError(
                    f"Unknown task: {task}. Use 'classification' or 'grading'"
                )

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

    For small imbalanced datasets, uses Manual Patient Stratification to ensure
    every class has at least one sample in the test set.

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
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    # Get all patient folders with labels
    patient_folders = _get_patient_folders(task, data_dir)

    if len(patient_folders) == 0:
        raise ValueError(f"No valid patient folders found for task: {task}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Group patients by class/grade for manual stratification
    from collections import defaultdict

    patients_by_class: dict = defaultdict(list)
    for folder, label in patient_folders:
        patients_by_class[label].append(folder)

    # Determine number of classes
    num_classes = max(patients_by_class.keys()) + 1

    # Manual Patient Stratification for small imbalanced datasets
    train_folders: List[Path] = []
    val_folders: List[Path] = []
    test_folders: List[Path] = []

    for label in range(num_classes):
        class_patients = patients_by_class.get(label, [])
        n_patients = len(class_patients)

        if n_patients == 0:
            continue

        # Sort to ensure deterministic order before shuffling
        class_patients.sort()

        # Shuffle patients for this class
        shuffled = class_patients.copy()
        random.shuffle(shuffled)

        if n_patients == 1:
            # Only 1 patient: put in train (can't have in test without train data)
            train_folders.extend(shuffled)
            print(f"Warning: Class {label} has only 1 patient. Assigned to train only.")
        elif n_patients == 2:
            # 2 patients: 1 train, 1 test (skip val)
            train_folders.append(shuffled[0])
            test_folders.append(shuffled[1])
            print(
                f"Warning: Class {label} has only 2 patients. Split: 1 train, 1 test (no val)."
            )
        elif n_patients < 5:
            # 3-4 patients: ensure at least 1 in each split
            # Priority: test (for evaluation), train (for learning), val (for tuning)
            test_folders.append(shuffled[0])
            train_folders.append(shuffled[1])
            if n_patients >= 3:
                val_folders.append(shuffled[2])
            if n_patients >= 4:
                train_folders.append(shuffled[3])
        else:
            # 5+ patients: use ratio-based split
            n_test = max(1, int(n_patients * test_ratio))
            n_val = max(1, int(n_patients * val_ratio))
            n_train = n_patients - n_test - n_val

            # Ensure at least 1 in train
            if n_train < 1:
                n_train = 1
                n_val = max(0, n_patients - n_test - n_train)

            test_folders.extend(shuffled[:n_test])
            val_folders.extend(shuffled[n_test : n_test + n_val])
            train_folders.extend(shuffled[n_test + n_val :])

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
    print(f"\n{'=' * 60}")
    print(f"Data Split Statistics (Task: {task})")
    print(f"{'=' * 60}")
    print(f"Data directory: {data_dir}")
    print(f"File extension: .{file_ext}")
    print(f"Total patients: {len(patient_folders)}")
    print(f"Train patients: {len(train_folders)} | Images: {len(train_files)}")
    print(f"Val patients:   {len(val_folders)} | Images: {len(val_files)}")
    print(f"Test patients:  {len(test_folders)} | Images: {len(test_files)}")

    # Print per-class distribution for all splits with readable names
    label_map_inv = CLASS_MAP_INV if task == "classification" else GRADE_MAP_INV

    train_class_counts: dict = defaultdict(int)
    for _, label in train_files:
        train_class_counts[label] += 1

    val_class_counts: dict = defaultdict(int)
    for _, label in val_files:
        val_class_counts[label] += 1

    test_class_counts: dict = defaultdict(int)
    for _, label in test_files:
        test_class_counts[label] += 1

    # Convert integer keys to readable names (sorted by original int key)
    def to_named_dist(counts: dict) -> dict:
        return {label_map_inv[k]: counts[k] for k in sorted(counts.keys())}

    print(f"Train Class Dist: {to_named_dist(train_class_counts)}")
    print(f"Val   Class Dist: {to_named_dist(val_class_counts)}")
    print(f"Test  Class Dist: {to_named_dist(test_class_counts)}")
    print(f"{'=' * 60}\n")

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


def get_loss_weights(
    file_list: List[Tuple[Path, int]],
    num_classes: int,
) -> torch.FloatTensor:
    """
    Calculate per-class weights for CrossEntropyLoss to handle class imbalance.

    Uses Square Root Smoothed inverse frequency weighting to avoid over-predicting
    minority classes. Classes with fewer samples get higher weights, but the
    difference is smoothed by taking the square root.

    Args:
        file_list: List of (file_path, label) tuples
        num_classes: Number of classes

    Returns:
        FloatTensor of shape (num_classes,) with per-class weights
    """
    import math

    # Count samples per class
    class_counts = [0] * num_classes
    for _, label in file_list:
        class_counts[label] += 1

    total_samples = sum(class_counts)

    # Square Root Smoothed inverse frequency weights
    # This reduces the aggressive penalty difference between minority/majority classes
    class_weights = []
    for count in class_counts:
        if count > 0:
            # Apply square root smoothing to reduce extreme weight differences
            weight = math.sqrt(total_samples / (num_classes * count))
        else:
            weight = 0.0  # Zero weight for missing classes
        class_weights.append(weight)

    # Normalize weights to sum to num_classes (keeps loss scale stable)
    weight_sum = sum(class_weights)
    if weight_sum > 0:
        class_weights = [w * num_classes / weight_sum for w in class_weights]

    return torch.FloatTensor(class_weights)


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


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., 2017).
    Focuses training on hard examples by down-weighting easy positives/negatives.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Unweighted CE to get true probability pt
        ce_loss_unweighted = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-ce_loss_unweighted)

        # Weighted CE for class-balanced loss
        ce_loss_weighted = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha
        )

        # Focal modulation: down-weight easy examples
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls embeddings of the same class together while pushing embeddings
    of different classes apart in the representation space.

    Args:
        temperature: Scaling factor for similarity (default: 0.1).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SupCon loss.

        Args:
            features: L2-normalized embeddings [B, n_views, D].
            labels: Ground truth labels [B].

        Returns:
            Scalar loss value.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Flatten views: [B, n_views, D] → [B * n_views, D]
        n_views = features.shape[1]
        contrast_features = features.reshape(batch_size * n_views, -1)

        # Repeat labels for each view: [B] → [B * n_views]
        labels = labels.contiguous().view(-1)
        if n_views > 1:
            labels = labels.repeat(n_views)

        total = contrast_features.shape[0]

        # Pairwise cosine similarity scaled by temperature: [total, total]
        similarity = torch.mm(contrast_features, contrast_features.T) / self.temperature

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(total, dtype=torch.bool, device=device)

        # Positive mask: same label, different index
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [total, total]
        pos_mask = label_eq & ~self_mask

        # Check: need at least 1 positive pair in the batch
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # For numerical stability, subtract max from each row
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Denominator: sum of exp(similarity) over all non-self entries
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Log-prob for each pair: log(exp(sim_ij) / sum_k≠i exp(sim_ik))
        log_prob = logits - log_denom

        # Mean of log-prob over positive pairs, per anchor
        pos_per_anchor = pos_mask.float().sum(dim=1)  # [total]
        valid = pos_per_anchor > 0

        mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1)[
            valid
        ] / pos_per_anchor[valid]

        loss = -mean_log_prob.mean()
        return loss


class EMDLoss(nn.Module):
    """
    Earth Mover's Distance (EMD) Loss for Ordinal Regression.
    Calculates the squared error between the cumulative distribution functions (CDFs).

    This loss is ideal for ordinal problems like fibrosis grading (G0-G3) where
    penalizing distant errors (G0 vs G3) more than neighbor errors (G2 vs G3)
    is desirable.
    """

    def __init__(self, num_classes: int = 4):
        super(EMDLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate EMD Loss.

        Args:
            logits: Model output logits of shape [Batch, NumClasses]
            targets: Ground truth labels of shape [Batch]

        Returns:
            Scalar loss value
        """
        # Convert logits to probabilities (Softmax)
        probs = F.softmax(logits, dim=1)

        # Calculate predicted CDF (Cumulative Sum)
        # pred_cdf shape: [Batch, NumClasses]
        pred_cdf = torch.cumsum(probs, dim=1)

        # Create Target CDF
        # For target class k, true_cdf[i] = 1 if i >= k, else 0
        # Example: If target is 2 (G2) out of 4 classes:
        # Class: 0  1  2  3
        # CDF:   0  0  1  1  (P(y<=i) = 1 if target <= i)
        true_cdf = torch.zeros_like(pred_cdf)
        for i in range(self.num_classes):
            # P(y <= i) is 1 if target <= i, else 0
            true_cdf[:, i] = (targets <= i).float()

        # Calculate Squared EMD (Wasserstein-2 distance for discrete ordinal)
        loss = torch.mean(torch.sum((pred_cdf - true_cdf) ** 2, dim=1))
        return loss
