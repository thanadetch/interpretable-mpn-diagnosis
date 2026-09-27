"""
Training pipeline for MIL Reticulin Fibrosis Grading (4-class).

Supports:
    - ABMIL: Lightweight gated-attention MIL (recommended for small datasets)
    - HybridMIL, DualStreamMIL, MultiBranchMIL, MeanPoolMIL

Trains models on pre-extracted reticulin backbone features to classify
Whole Slide Images into 4 fibrosis grades: G0, G1, G2, G3.

Usage:
    python -m src.train_grading_reti --backbone titan --model_type simple --epochs 50
"""

import argparse
import csv
import random
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Headless matplotlib for confusion matrix images
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from tqdm import tqdm

from core.config import EXPERIMENTS_DIR, SEED
from data.bag_dataset import GradingBagDatasetFull
from data.augmentations import load_augmentation

from models.hybrid_mil import HybridMIL
from models.simple_mil import SimpleGatedMIL
from models.dual_stream_mil import DualStreamMIL
from models.multi_branch_mil import MultiBranchMIL
from models.mean_pool_mil import MeanPoolMIL
from models.per_patch_score_pool_mil import PerPatchScorePoolingMIL

# ── class definitions for reticulin fibrosis grading ─────────────────────
CLASS_MAP = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}
CLASS_MAP_INV = {v: k for k, v in CLASS_MAP.items()}
CLASS_NAMES = ["G0", "G1", "G2", "G3"]

# ── backbone configuration ───────────────────────────────────────────────
BACKBONE_CONFIG: Dict[str, Dict] = {
    "titan": {
        "dim": 768,
        "feature_dir": "features_titan_reti",
        "display_name": "TITAN",
    },
    "uni2": {
        "dim": 1536,
        "feature_dir": "features_uni2_reti",
        "display_name": "UNI2-h",
    },
    "uni2_no_patch": {
        "dim": 1536,
        "feature_dir": "features_uni2_reti_no_patch",
        "display_name": "UNI2-h (no patch)",
    },
    "virchow2": {
        "dim": 1280,
        "feature_dir": "features_virchow2_reti",
        "display_name": "Virchow2",
    },
    "titan_no_patch": {
        "dim": 768,
        "feature_dir": "features_titan_reti_no_patch",
        "display_name": "TITAN (no patch)",
    },
    "virchow2_no_patch": {
        "dim": 1280,
        "feature_dir": "features_virchow2_reti_no_patch",
        "display_name": "Virchow2 (no patch)",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        # MPS has its own RNG; seed it so train/val/test are reproducible on Apple Silicon.
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass


def log(msg: str, log_file: Optional[Path] = None) -> None:
    """Print to console and optionally append to a log file."""
    print(msg)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def _resolve_device(pref: str) -> str:
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return pref


def patient_split(
    dataset: GradingBagDatasetFull,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split by patient ID and fibrosis grade label.

    Groups patients by their class label, then performs a hardcoded split
    independently within each class to guarantee every class is represented
    in Train, Val, and Test.

    Dataset: 42 patients -> Train 28, Val 7, Test 7.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    rng = random.Random(seed)

    # Map patient -> (sample indices, grade label)
    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    patient_to_label: Dict[str, int] = {}
    for idx, (pt_path, label) in enumerate(dataset.samples):
        patient_id = pt_path.parent.name
        patient_to_indices[patient_id].append(idx)
        patient_to_label[patient_id] = label

    # Group patients by their grade label
    label_to_patients: Dict[int, List[str]] = defaultdict(list)
    for patient_id, label in patient_to_label.items():
        label_to_patients[label].append(patient_id)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    # Hardcoded splits: {label_idx: (n_val, n_test)}
    # G0: 4 total -> Val 1, Test 1, Train 2
    # G1: 15 total -> Val 2, Test 2, Train 11
    # G2: 18 total -> Val 3, Test 3, Train 12
    # G3: 5 total -> Val 1, Test 1, Train 3
    split_targets = {
        0: (2, 2),  # G0 -> Val 2, Test 2
        1: (3, 3),  # G1 -> Val 3, Test 3
        2: (3, 3),  # G2 -> Val 3, Test 3 (Leaves 12 for Train)
        3: (2, 2),  # G3 -> Val 2, Test 2
    }

    for label in sorted(label_to_patients.keys()):
        patients = label_to_patients[label]
        rng.shuffle(patients)

        # Get the target counts for this class
        n_val, n_test = split_targets.get(label, (1, 1))
        n_train = len(patients) - n_val - n_test

        if n_train < 0:
            raise ValueError(f"Not enough patients in class {label} to split!")

        train_patients = patients[:n_train]
        val_patients = patients[n_train : n_train + n_val]
        test_patients = patients[n_train + n_val :]

        train_idx.extend(i for p in train_patients for i in patient_to_indices[p])
        val_idx.extend(i for p in val_patients for i in patient_to_indices[p])
        test_idx.extend(i for p in test_patients for i in patient_to_indices[p])

    return train_idx, val_idx, test_idx


def patient_kfold_split(
    dataset: GradingBagDatasetFull,
    n_folds: int = 5,
    fold: int = 0,
    seed: int = 2,
) -> Tuple[List[int], List[int], List[int]]:
    """Grade-stratified k-fold split at the PATIENT level.

    Why this is NOT the same as re-running ``patient_split`` with different seeds: that draws a
    fresh random split each time, so patients recur in several test sets while ~24% never land in
    one at all. Here the cohort is partitioned into ``n_folds`` DISJOINT patient folds, so across
    the full sweep every patient is predicted exactly once as test and once as val:

        test  = fold k
        val   = fold (k+1) % n_folds        (rotating, so val is disjoint from test and train)
        train = the remaining folds

    Folds are built inside each grade so every fold keeps the cohort's grade mix, and the shuffle
    is seeded, so fold membership is fixed and reproducible across models and backbones.

    ``patient_split`` is left untouched; this is only reached when ``--n_folds > 0``.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"--fold must be in [0, {n_folds}), got {fold}")

    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    patient_to_label: Dict[str, int] = {}
    for idx, (pt_path, label) in enumerate(dataset.samples):
        pid = pt_path.parent.name
        patient_to_indices[pid].append(idx)
        patient_to_label[pid] = label

    label_to_patients: Dict[int, List[str]] = defaultdict(list)
    for pid, label in patient_to_label.items():
        label_to_patients[label].append(pid)

    # deal each grade's patients round-robin into the folds -> stratified and near-equal sizes
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for label in sorted(label_to_patients):
        patients = sorted(label_to_patients[label])       # sort first: order must not depend on walk order
        rng.shuffle(patients)
        for j, pid in enumerate(patients):
            folds[j % n_folds].append(pid)

    test_patients = set(folds[fold])
    val_patients = set(folds[(fold + 1) % n_folds])
    train_patients = {p for j, f in enumerate(folds) if j not in (fold, (fold + 1) % n_folds)
                      for p in f}

    def collect(ps):
        return [i for p in sorted(ps) for i in patient_to_indices[p]]

    return collect(train_patients), collect(val_patients), collect(test_patients)


# ── collate ──────────────────────────────────────────────────────────────
def collate_bags(batch):
    """Custom collate function for variable-length bags."""
    features_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    return features_list, labels, slide_ids


# ── training ─────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str = "simple",
    formulation: str = "classification",
    augmentation=None,
    train_pool=None,
    adv_eps: float = 0.0,
) -> Tuple[float, float, float, List[float], float, float, float, dict]:
    """Train for one epoch. Handles standard MIL model types."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loss_sums = defaultdict(float)
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for features_list, labels, slide_ids in pbar:
        labels = labels.to(device)
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        batch_correct = 0

        for i, features in enumerate(features_list):
            features = features.to(device)  # [N, D]
            label = labels[i].item()

            # ── feature-space augmentation (train-only; OFF when augmentation is None) ──
            target_val = float(label)
            if augmentation is not None and formulation == "regression":
                features, target_val = augmentation(features, float(label), pool=train_pool)

            # ── FGSM adversarial-margin perturbation (train-only; OFF when adv_eps<=0) ──
            # Worst-case L-inf perturbation in the gradient-sign direction; per-bag and
            # ~zero-mean across the cohort (no systematic class-mean shift), so it adds
            # decision-margin robustness rather than mass-shifting the pooled representation.
            if adv_eps > 0.0 and formulation == "regression":
                f_adv = features.detach().clone().requires_grad_(True)
                logit_adv, _, _ = model(f_adv)
                l_adv = criterion(
                    logit_adv.view(-1),
                    torch.tensor([target_val], dtype=torch.float32, device=device),
                )
                g = torch.autograd.grad(l_adv, f_adv)[0]
                radius = adv_eps * features.detach().norm(dim=1, keepdim=True).mean()
                features = (features.detach() + radius * g.sign()).detach()

            # Standard MIL models (simple, hybrid, dual_stream, multi_branch, mean_pool)
            logits, _, _ = model(features)
            if formulation == "regression":
                label_tensor = torch.tensor([target_val], dtype=torch.float32, device=device)
                loss = criterion(logits.view(-1), label_tensor)
            else:
                label_tensor = torch.tensor([label], device=device)
                loss = criterion(logits.view(1, -1), label_tensor)
            loss_sums["bag_loss"] += loss.item()

            batch_loss = batch_loss + loss

            if formulation == "regression":
                pred = int(max(0, min(3, round(logits.view(-1).item()))))
            else:
                pred = logits.argmax().item()
            batch_correct += int(pred == label)
            all_preds.append(pred)
            all_labels.append(label)

        batch_size = len(features_list)
        batch_loss = batch_loss / batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += batch_loss.item() * batch_size
        correct += batch_correct
        total += batch_size

        pbar.set_postfix(
            loss=f"{batch_loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%"
        )

    avg_loss = running_loss / total
    avg_acc = 100.0 * correct / total
    train_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    # Per-class recall
    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    recall_list = [100.0 * r for r in per_class_recall]
    train_macro_recall = sum(recall_list) / len(recall_list)

    # QWK (Quadratic Weighted Kappa) — primary metric
    train_qwk = cohen_kappa_score(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))), weights="quadratic"
    )

    # MAE (Mean Absolute Error) — ordinal distance metric
    train_mae = mean_absolute_error(all_labels, all_preds)

    avg_components = {k: v / total for k, v in loss_sums.items()}
    return (
        avg_loss,
        avg_acc,
        train_f1,
        recall_list,
        train_macro_recall,
        train_qwk,
        train_mae,
        avg_components,
    )


# ── validation & evaluation ──────────────────────────────────────────────
@torch.no_grad()
def validate_and_evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "  Val  ",
    formulation: str = "classification",
    return_predictions: bool = False,
):
    """
    Validate/Test a MIL model and return metric strings.

    Returns:
        avg_loss: Average loss over all samples.
        accuracy: Accuracy as a percentage.
        f1_macro: Macro-averaged F1 score.
        recall_list: Per-class recall as percentages.
        qwk: Quadratic Weighted Kappa score.
        mae: Mean Absolute Error.
        cm_str: Formatted confusion matrix string.
        report_str: Formatted classification report string (precision/recall/F1).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_slide_ids: List[str] = []
    all_raw_outputs: List[float] = []  # logits (regression scalar) or argmax pred

    for features_list, labels, slide_ids in tqdm(loader, desc=desc, leave=False):
        labels = labels.to(device)

        for i, features in enumerate(features_list):
            features = features.to(device)
            label = labels[i : i + 1]

            if isinstance(model, SimpleGatedMIL):
                logits, _, _ = model(features, return_attention=False)
            else:
                logits, _, _ = model(features)

            if formulation == "regression":
                label_float = label.float()
                loss = criterion(logits.view(-1), label_float)
                running_loss += loss.item()

                raw = float(logits.view(-1).item())
                pred_val = int(max(0, min(3, round(raw))))
                pred = torch.tensor([pred_val], device=device)
                correct += pred.eq(label).sum().item()
                total += 1

                all_preds.append(pred_val)
                all_labels.append(label.item())
                all_slide_ids.append(slide_ids[i])
                all_raw_outputs.append(raw)
            else:
                logits = logits.view(1, -1)

                loss = criterion(logits, label)
                running_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += pred.eq(label).sum().item()
                total += 1

                pred_val = int(pred.item())
                all_preds.append(pred_val)
                all_labels.append(label.item())
                all_slide_ids.append(slide_ids[i])
                all_raw_outputs.append(float(pred_val))

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    # Compute imbalance-aware metrics
    f1_macro = f1_score(
        all_labels,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    # Build confusion matrix string
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    cm_lines = ["  Confusion Matrix:"]
    header = "         " + "  ".join(f"{name:>5}" for name in CLASS_NAMES)
    cm_lines.append(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:5d}" for v in row)
        cm_lines.append(f"  {CLASS_NAMES[i]:>5}   {row_str}")
    cm_str = "\n".join(cm_lines)

    # Build classification report string
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    )
    report_lines = ["  Classification Report:"]
    # Do NOT use .strip() on the full string as it messes up header alignment
    for line in report.splitlines():
        if line.rstrip():
            report_lines.append(f"  {line}")
    report_str = "\n".join(report_lines)

    # Per-class recall
    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    recall_list = [100.0 * r for r in per_class_recall]
    macro_recall = sum(recall_list) / len(recall_list)

    # QWK (Quadratic Weighted Kappa) — primary metric
    qwk = cohen_kappa_score(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))), weights="quadratic"
    )

    # MAE (Mean Absolute Error) — ordinal distance metric
    mae = mean_absolute_error(all_labels, all_preds)

    if return_predictions:
        return (
            avg_loss,
            accuracy,
            f1_macro,
            recall_list,
            macro_recall,
            qwk,
            mae,
            cm_str,
            report_str,
            all_slide_ids,
            all_labels,
            all_preds,
            all_raw_outputs,
            cm,
        )

    return (
        avg_loss,
        accuracy,
        f1_macro,
        recall_list,
        macro_recall,
        qwk,
        mae,
        cm_str,
        report_str,
    )


# ── reproducibility / experiment artifact helpers ─────────────────────────
def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_predictions_csv(
    path: Path,
    slide_ids: List[str],
    labels: List[int],
    preds: List[int],
    raw_outputs: List[float],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["slide_id", "label_idx", "label_name", "pred_idx", "pred_name", "raw_output"]
        )
        for sid, y, p, r in zip(slide_ids, labels, preds, raw_outputs):
            writer.writerow([sid, y, CLASS_NAMES[y], p, CLASS_NAMES[p], f"{r:.6f}"])


def save_confusion_matrix(path_csv: Path, path_png: Path, cm: np.ndarray, title: str) -> None:
    # CSV
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + CLASS_NAMES)
        for i, row in enumerate(cm):
            w.writerow([CLASS_NAMES[i]] + [int(v) for v in row])
    # PNG
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)


def append_leaderboard(
    leaderboard_path: Path,
    row: Dict,
    fieldnames: List[str],
) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not leaderboard_path.exists()
    # Re-read existing header if present to preserve schema
    if not write_header:
        with open(leaderboard_path) as f:
            existing_header = f.readline().strip().split(",")
        if existing_header != fieldnames:
            # Keep both: write a sidecar leaderboard with our schema
            sidecar = leaderboard_path.parent / "leaderboard_v2.csv"
            write_header = not sidecar.exists()
            leaderboard_path = sidecar
    with open(leaderboard_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def write_method_note(path: Path, args: argparse.Namespace, model_name: str) -> None:
    note = f"""# Method Note — {path.parent.name}

**Model**: `{args.model_type}` ({model_name})
**Backbone**: `{args.backbone}`
**Formulation**: `{args.formulation}` (loss: SmoothL1 for regression, weighted CE+LS for classification)
**Selection metric**: validation `{args.main_metric}` (max), patience {args.early_stop_patience}
**Seed**: {args.seed}  |  **LR**: {args.lr}  |  **Epochs**: {args.epochs}
**Sampler**: {"patient-balanced WeightedRandomSampler" if args.sampler_weights else "shuffled (uniform)"}
**Patient-level split**: hardcoded stratified per grade (Train 28 / Val 7 / Test 7 patients) — UNCHANGED across runs.

## Rationale
Plain mean pooling collapses an ROI to its average patch embedding, which contradicts
the clinical reality that reticulin fibrosis grading is driven by the *most-affected*
regions of the ROI, not the average. Variants tested here aggregate patch information
in ways that preserve heterogeneity (std), worst-region evidence (max / top-k / quantile),
or learn a per-patch latent severity score (`patch_score_pool`) that doubles as an
interpretability heatmap with negligible extra parameters. We hold all training
hyperparameters, splits, loss, and the foundation-model features fixed and ablate
only the aggregator, so any change in validation QWK can be attributed to the pooling
choice.

## Reproduce
```
python -m src.train_grading_reti \\
  --backbone {args.backbone} --model_type {args.model_type} \\
  --epochs {args.epochs} --lr {args.lr} --batch_size {args.batch_size} \\
  --seed {args.seed} --num_workers {args.num_workers} \\
  --formulation {args.formulation} --main_metric {args.main_metric}{(' --postfix ' + args.postfix) if args.postfix else ''}
```
"""
    path.write_text(note)

LEADERBOARD_FIELDS = [
    "timestamp",
    "run_name",
    "experiment_dir",
    "model_type",
    "backbone",
    "seed",
    "split_seed",
    "formulation",
    "main_metric",
    "best_epoch",
    "val_qwk",
    "val_mae",
    "val_accuracy",
    "val_f1_macro",
    "val_macro_recall",
    "val_loss",
    "test_qwk",
    "test_mae",
    "test_accuracy",
    "test_f1_macro",
    "test_macro_recall",
    "test_loss",
    "checkpoint",
]



# ── argument parsing ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MIL models for Reticulin Fibrosis Grading (G0–G3)."
    )
    parser.add_argument(
        "--backbone",
        required=True,
        choices=list(BACKBONE_CONFIG.keys()),
        help="Foundation model backbone.",
    )
    parser.add_argument(
        "--model_type",
        default="simple",
        choices=[
            "simple",
            "hybrid",
            "dual_stream",
            "multi_branch",
            "mean_pool",
            "patch_score_pool",
            "novelty_attempt",
        ],
        help="MIL model type. Use 'novelty_attempt' with --novelty_id <NAME> to load a model from src/models/novelty_attempts/<NAME>.py.",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers."
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Top-k pooling: use mean of k highest-attention patches. 0 = standard attention (default: 0).",
    )

    # Per-patch score pooling MIL hyperparameters (used when --model_type patch_score_pool)
    parser.add_argument(
        "--score_pool_mode",
        type=str,
        default="mean_quantile_hybrid",
        choices=["mean", "quantile", "topk_mean", "mean_quantile_hybrid"],
        help="Aggregation mode for PerPatchScorePoolingMIL (default: mean_quantile_hybrid).",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.75,
        help="Quantile q used by patch_score_pool quantile/hybrid modes (default: 0.75).",
    )
    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=0.20,
        help="Top-k ratio used by patch_score_pool topk_mean mode (default: 0.20).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of patch_score_pool MLP head (default: 128).",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=0.50,
        help="Upper bound for learnable mixing λ in mean_quantile_hybrid (default: 0.50).",
    )
    parser.add_argument(
        "--init_lambda",
        type=float,
        default=0.20,
        help="Initial value for learnable mixing λ in mean_quantile_hybrid (default: 0.20).",
    )
    parser.add_argument(
        "--use_linear_patch_head",
        action="store_true",
        help="Use a single Linear layer instead of an MLP for the patch score head.",
    )

    parser.add_argument(
        "--novelty_id",
        type=str,
        default="",
        help="Novelty module name in src/models/novelty_attempts/ (without .py). "
             "Used only when --model_type novelty_attempt.",
    )

    # Experiment tracking
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="String prepended to the experiment directory, model checkpoint, and "
             "log file names. Useful for numbering ablation cells (e.g. --prefix 01) "
             "so runs sort in table order in `ls`.",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="String to append to experiment directory, model checkpoint, and log file names.",
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=15,
        help="Stop training after this many epochs without val QWK improvement (default: 15).",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Formulation of the problem: 'classification' (4 classes) or 'regression' (scalar 0-3).",
    )
    parser.add_argument(
        "--sampler_weights",
        action="store_true",
        help="Enable patient-balanced WeightedRandomSampler for the training loader (default: off).",
    )
    parser.add_argument(
        "--sampler_temp",
        type=float,
        default=1.0,
        help="Temperature beta on the balanced sampler weights (w**beta). 1.0=full balance (default), "
             "0.0=uniform; intermediate values partially balance. Only used with --sampler_weights.",
    )
    parser.add_argument(
        "--adv_eps",
        type=float,
        default=0.0,
        help="FGSM adversarial-margin perturbation radius (fraction of mean per-patch feature norm). "
             "0.0 = off (default, byte-identical baseline). Train-time, regression only.",
    )
    parser.add_argument(
        "--main_metric",
        type=str,
        default="qwk",
        choices=["qwk", "f1", "acc", "macro_recall"],
        help="Metric used to determine the best model for early stopping and saving.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device. 'auto' picks cuda > mps > cpu (default: auto).",
    )
    parser.add_argument(
        "--leaderboard_path",
        type=str,
        default="results/leaderboard.csv",
        help="Path to the leaderboard CSV (default: results/leaderboard.csv).",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Append a row to the global leaderboard CSV. OFF by default — leaderboard "
             "writes are opt-in to keep results/leaderboard.csv clean during ad-hoc / "
             "sanity runs. Per-run artifacts (config, logs, checkpoint, predictions, "
             "CMs, method_note) are always saved regardless of this flag. Batch / "
             "novelty-search scripts pass --leaderboard automatically.",
    )
    parser.add_argument(
        "--method_note",
        type=str,
        default="",
        help="Optional extra free-text rationale appended to method_note.md.",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        help="Name of a feature-space augmentation module in src/data/augmentations/ "
             "(e.g. 'mixup_instance_pool'). Omit / 'none' = OFF (code path identical to baseline). "
             "Train-only; applied under regression formulation. "
             "Add a new augmentation by dropping a new file in src/data/augmentations/.",
    )
    parser.add_argument(
        "--aug_strength",
        type=float,
        default=None,
        help="Optional override of the augmentation's primary strength hyper-parameter "
             "(e.g. MixUp Beta-alpha). If unset, the module's KWARGS default is used.",
    )
    parser.add_argument(
        "--max_roi_per_patient",
        type=int,
        default=0,
        help="Cap #ROIs (bags) per patient in the TRAIN split only (val/test untouched). "
             "0 = no cap (use all). For the ROI/patient learning-curve ablation. "
             "Selection is seeded (args.seed) for reproducibility.",
    )

    # ── patient-level k-fold cross-validation (OFF by default) ───────────────
    # The default single split evaluates only 10 of the 50 patients, so 80% of the cohort never
    # contributes to a reported number. With --n_folds the cohort is partitioned into disjoint
    # patient folds and every patient is predicted exactly once across the sweep.
    parser.add_argument(
        "--n_folds",
        type=int,
        default=0,
        help="Patient-level k-fold CV: number of DISJOINT patient folds. 0 = OFF (default), "
             "which keeps the original single stratified split byte-for-byte. Use with --fold.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold is the TEST set (0-based). val = fold+1 (rotating), train = the rest.",
    )
    parser.add_argument(
        "--fold_seed",
        type=int,
        default=2,
        help="Seed for the patient shuffle that builds the folds. Must be IDENTICAL across every "
             "model and backbone in a sweep, otherwise folds are not comparable. Kept separate "
             "from --seed so weight init can vary without changing fold membership.",
    )

    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Setup Experiment Directory
    now = datetime.now()
    date_dir = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Build unified run_name for directory, log, and checkpoint
    run_name = f"reti_{args.model_type}_{args.backbone}"
    if args.model_type == "simple" and args.topk > 0:
        run_name += f"_topk{args.topk}"
    if args.postfix:
        run_name += f"_{args.postfix}"
    if args.prefix:
        # Prepend so runs sort by ablation index in `ls` (e.g. 01_reti_mean_pool_uni2_...).
        run_name = f"{args.prefix}_{run_name}"

    # Group experiments by date so a single `experiments/` folder doesn't
    # explode into hundreds of siblings: experiments/<YYYYMMDD>/<run>_<ts>/
    exp_dir = EXPERIMENTS_DIR / date_dir / f"{run_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_file = exp_dir / f"{run_name}.log"

    # Reproducibility artifacts (config + per-epoch CSV log)
    config_path = exp_dir / "config.json"
    training_log_csv = exp_dir / "training_log.csv"
    save_json(config_path, {**vars(args), "timestamp": timestamp, "run_name": run_name})
    with open(training_log_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "train_f1",
                "train_macro_recall",
                "train_qwk",
                "train_mae",
                "val_loss",
                "val_acc",
                "val_f1",
                "val_macro_recall",
                "val_qwk",
                "val_mae",
                "lr",
            ]
        )

    # ── LOG CONFIGURATION ─────────────────────────────────────────────
    log(f"{'=' * 60}", log_file)
    log(f"Experiment: {exp_dir.name}", log_file)
    log(f"Log File:   {log_file}", log_file)
    log(f"{'=' * 60}", log_file)

    log(f"\nConfiguration:", log_file)
    for key, value in vars(args).items():
        log(f"  --{key}: {value}", log_file)

    cfg = BACKBONE_CONFIG[args.backbone]
    log(f"\nBackbone Config: {json.dumps(cfg, indent=3)}", log_file)

    features_dir = Path(args.data_root) / cfg["feature_dir"]
    input_dim = cfg["dim"]
    display_name = cfg["display_name"]
    num_classes = 1 if args.formulation == "regression" else 4

    device = torch.device(_resolve_device(args.device))
    log(f"Device: {device}", log_file)

    # ── Data ──────────────────────────────────────────────────────────
    log(f"\nLoading {display_name} features from: {features_dir}", log_file)
    full_dataset = GradingBagDatasetFull(features_dir)
    log(f"  Total bags: {len(full_dataset)}", log_file)

    if args.n_folds > 0:
        train_idx, val_idx, test_idx = patient_kfold_split(
            full_dataset, n_folds=args.n_folds, fold=args.fold, seed=args.fold_seed)
        n_pat = lambda ix: len({full_dataset.get_slide_path(i).parent.name for i in ix})
        log(
            f"  Patient {args.n_folds}-fold CV, fold {args.fold} "
            f"(fold_seed={args.fold_seed}): "
            f"train {n_pat(train_idx)} pat / {len(train_idx)} ROI | "
            f"val {n_pat(val_idx)} / {len(val_idx)} | test {n_pat(test_idx)} / {len(test_idx)}",
            log_file,
        )
    else:
        train_idx, val_idx, test_idx = patient_split(full_dataset, seed=args.seed)
    log(
        f"  Split -- Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}",
        log_file,
    )

    # ── optional: cap #ROIs per patient in TRAIN only (ROI/patient learning-curve) ──
    if args.max_roi_per_patient and args.max_roi_per_patient > 0:
        cap = args.max_roi_per_patient
        by_pat: Dict[str, List[int]] = defaultdict(list)
        for i in train_idx:
            by_pat[full_dataset.samples[i][0].parent.name].append(i)
        rng_cap = random.Random(args.seed)
        capped: List[int] = []
        for _p, idxs in by_pat.items():
            capped.extend(rng_cap.sample(idxs, cap) if len(idxs) > cap else idxs)
        capped.sort()
        log(
            f"  ROI/patient cap = {cap}: train bags {len(train_idx)} -> {len(capped)} "
            f"(over {len(by_pat)} patients; val/test untouched)",
            log_file,
        )
        train_idx = capped

    log("  Detailed Breakdown (Patients / Bags):", log_file)
    splits = {"Train": train_idx, "Val": val_idx, "Test": test_idx}
    for split_name, indices in splits.items():
        stats = {
            label: {"patients": set(), "bags": 0} for label in range(len(CLASS_NAMES))
        }
        for idx in indices:
            pt_path, label = full_dataset.samples[idx]
            patient_id = pt_path.parent.name
            stats[label]["patients"].add(patient_id)
            stats[label]["bags"] += 1

        parts = []
        for label in range(len(CLASS_NAMES)):
            grade = CLASS_NAMES[label]
            n_pat = len(stats[label]["patients"])
            n_bags = stats[label]["bags"]
            parts.append(f"{grade}: {n_pat:2d} ({n_bags:3d})")

        log(f"    {split_name:<5} : " + " | ".join(parts), log_file)

    # ── Patient-balanced sampling (optional) ──────────────────────────
    train_sampler = None
    if args.sampler_weights:
        patient_to_label: Dict[str, int] = {}
        bags_per_patient: Dict[str, int] = defaultdict(int)
        patients_per_class: Dict[int, set] = defaultdict(set)

        for idx in train_idx:
            pt_path, label = full_dataset.samples[idx]
            patient_id = pt_path.parent.name
            patient_to_label[patient_id] = label
            bags_per_patient[patient_id] += 1
            patients_per_class[label].add(patient_id)

        num_patients_per_class = {k: len(v) for k, v in patients_per_class.items()}

        sample_weights: List[float] = []
        for idx in train_idx:
            pt_path, label = full_dataset.samples[idx]
            patient_id = pt_path.parent.name
            weight = 1.0 / (
                num_patients_per_class[label] * bags_per_patient[patient_id]
            )
            sample_weights.append(weight)

        beta = float(getattr(args, "sampler_temp", 1.0))
        if beta != 1.0:
            sample_weights = [w ** beta for w in sample_weights]
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        log(f"  Using patient-balanced WeightedRandomSampler (temp beta={beta}) for training.", log_file)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bags,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bags,
    )

    # ── Augmentation (train-only; None = OFF, baseline behaviour) ──────
    augmentation = load_augmentation(args.augmentation, args.aug_strength)
    if augmentation is not None:
        log(
            f"  Train augmentation: {args.augmentation} "
            f"(strength={args.aug_strength if args.aug_strength is not None else 'default'})",
            log_file,
        )

    # ── Model ─────────────────────────────────────────────────────────
    checkpoint_name = f"best_{run_name}.pth"

    if args.model_type == "simple":
        model = SimpleGatedMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=args.topk,
        ).to(device)
    elif args.model_type == "hybrid":
        k = args.topk if args.topk > 0 else 5
        model = HybridMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=k,
        ).to(device)
    elif args.model_type == "dual_stream":
        k = args.topk if args.topk > 0 else 5
        model = DualStreamMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=k,
        ).to(device)
    elif args.model_type == "multi_branch":
        model = MultiBranchMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk_focal=5,
        ).to(device)
    elif args.model_type == "mean_pool":
        model = MeanPoolMIL(
            vision_dim=cfg["dim"],
            num_classes=num_classes,
        ).to(device)
    elif args.model_type == "patch_score_pool":
        if args.formulation != "regression":
            raise ValueError(
                "patch_score_pool model requires --formulation regression."
            )
        model = PerPatchScorePoolingMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            mode=args.score_pool_mode,
            quantile=args.quantile,
            topk_ratio=args.topk_ratio,
            max_lambda=args.max_lambda,
            init_lambda=args.init_lambda,
            use_mlp_head=(not args.use_linear_patch_head),
        ).to(device)
    elif args.model_type == "novelty_attempt":
        if not args.novelty_id:
            raise ValueError(
                "Pass --novelty_id <module_name> when --model_type novelty_attempt."
            )
        import importlib

        mod = importlib.import_module(
            f"models.novelty_attempts.{args.novelty_id}"
        )
        # Override input_dim to match selected backbone
        kwargs = dict(getattr(mod, "KWARGS", {}))
        kwargs["input_dim"] = input_dim
        kwargs["num_classes"] = num_classes
        model = mod.Model(**kwargs).to(device)
        log(f"Loaded novelty model: {args.novelty_id} -> {mod.Model.__name__}", log_file)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\nModel: {args.model_type.upper()}", log_file)
    log(
        f"  Parameters: {trainable_params:,} trainable / {total_params:,} total",
        log_file,
    )

    # ── Optimiser & loss ──────────────────────────────────────────────
    # Class weights: G0=1.0, G1=1.0, G2=1.0, G3=1.0
    # G0 and G3 get higher weights to handle severe minority classes
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    if args.formulation == "regression":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    log(f"\nLoss: {criterion.__class__.__name__}", log_file)
    log(f"Formulation: {args.formulation}", log_file)
    if args.formulation == "classification":
        weights_str = "  ".join(
            f"{n}={w:.1f}" for n, w in zip(CLASS_NAMES, class_weights.tolist())
        )
        log(f"Weights: {weights_str}", log_file)

    # ── Training loop ─────────────────────────────────────────────────
    best_metric_val = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    log(f"\n{'=' * 60}", log_file)
    log(f"Start Training ({args.epochs} epochs)", log_file)
    log(f"{'=' * 60}", log_file)

    # Table header
    hdr = "Ep   | Mode  | Loss  | Acc   | F1    | MacRec| MAE   | QWK   | Recall ( G0 / G1 / G2 / G3 )"
    sep = "-" * len(hdr)

    def fmt_recall(recall_list: List[float]) -> str:
        """Format per-class recall as compact aligned columns."""
        return "  ".join(f"{r:5.1f}" for r in recall_list)

    def print_header() -> None:
        log(hdr, log_file)
        log(sep, log_file)

    print_header()

    for epoch in range(1, args.epochs + 1):
        # Repeat header every 10 epochs for readability
        if epoch > 1 and (epoch - 1) % 10 == 0:
            print_header()

        # Train
        (
            train_loss,
            train_acc,
            train_f1,
            train_recall,
            train_macro_recall,
            train_qwk,
            train_mae,
            loss_components,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            model_type=args.model_type,
            formulation=args.formulation,
            augmentation=augmentation,
            train_pool=train_loader.dataset,
            adv_eps=args.adv_eps,
        )

        # Validate
        (
            val_loss,
            val_acc,
            val_f1,
            val_recall,
            val_macro_recall,
            val_qwk,
            val_mae,
            val_cm_str,
            val_report_str,
            val_slide_ids,
            val_labels_list,
            val_preds_list,
            val_raw_list,
            val_cm_arr,
        ) = validate_and_evaluate(
            model,
            val_loader,
            criterion,
            device,
            formulation=args.formulation,
            return_predictions=True,
        )

        # Per-epoch logging (2-line table)
        ep_str = f"{epoch}/{args.epochs}"
        log(
            f"{ep_str:<5}| Train | {train_loss:<5.3f} | {train_acc:<5.1f} "
            f"| {train_f1:<5.3f} | {train_macro_recall:<5.1f} | {train_mae:<5.3f} | {train_qwk:<5.3f} | {fmt_recall(train_recall)}",
            log_file,
        )
        log(
            f"     | Val   | {val_loss:<5.3f} | {val_acc:<5.1f} "
            f"| {val_f1:<5.3f} | {val_macro_recall:<5.1f} | {val_mae:<5.3f} | {val_qwk:<5.3f} | {fmt_recall(val_recall)}",
            log_file,
        )
        log(sep, log_file)

        # Append per-epoch row to training_log.csv
        with open(training_log_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.4f}",
                    f"{train_f1:.6f}",
                    f"{train_macro_recall:.4f}",
                    f"{train_qwk:.6f}",
                    f"{train_mae:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.4f}",
                    f"{val_f1:.6f}",
                    f"{val_macro_recall:.4f}",
                    f"{val_qwk:.6f}",
                    f"{val_mae:.6f}",
                    f"{optimizer.param_groups[0]['lr']:.2e}",
                ]
            )

        scheduler.step()

        # Save best model (based on selected main metric)
        metrics_dict = {
            "qwk": val_qwk,
            "f1": val_f1,
            "acc": val_acc,
            "macro_recall": val_macro_recall,
        }
        current_metric = metrics_dict[args.main_metric]

        if current_metric > best_metric_val:
            best_metric_val = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_main_metric": current_metric,
                "val_qwk": val_qwk,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "backbone": args.backbone,
                "model_type": args.model_type,
                "args": vars(args),
            }
            torch.save(checkpoint, exp_dir / checkpoint_name)
            log(
                f"     >>> ⭐ New Best Model! Val {args.main_metric.upper()}: {current_metric:.3f} | Acc: {val_acc:.1f}%",
                log_file,
            )
            log(f"\n{val_cm_str}", log_file)
            log(f"\n{val_report_str}", log_file)
            # NOTE: val_predictions.csv, val_confusion_matrix.{csv,png},
            # and val_metrics.json are written ONCE after training, after the
            # best checkpoint is reloaded — see the "Final val on best ckpt"
            # block below. This avoids rewriting them on every new-best epoch.
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                log(
                    f"\nEarly stopping triggered at epoch {epoch} "
                    f"(no val {args.main_metric} improvement for {args.early_stop_patience} epochs)",
                    log_file,
                )
                break

    # ── Test Phase ────────────────────────────────────────────────────
    log(f"\n{'=' * 60}", log_file)
    log(f"Final Evaluation on TEST SET", log_file)
    log(f"{'=' * 60}", log_file)

    best_ckpt_path = exp_dir / checkpoint_name
    test_metrics: Dict[str, float] = {}
    if best_ckpt_path.exists():
        log(f"Loading best checkpoint from: {best_ckpt_path}", log_file)
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # ── Final val on best ckpt (write artifacts once) ─────────────
        log(f"\nRe-evaluating best checkpoint on VAL set for final artifacts...", log_file)
        (
            f_val_loss,
            f_val_acc,
            f_val_f1,
            f_val_recall,
            f_val_macro_recall,
            f_val_qwk,
            f_val_mae,
            f_val_cm_str,
            f_val_report_str,
            f_val_slide_ids,
            f_val_labels,
            f_val_preds,
            f_val_raw,
            f_val_cm_arr,
        ) = validate_and_evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc="  Val* ",
            formulation=args.formulation,
            return_predictions=True,
        )
        save_predictions_csv(
            exp_dir / "val_predictions.csv",
            f_val_slide_ids,
            f_val_labels,
            f_val_preds,
            f_val_raw,
        )
        save_confusion_matrix(
            exp_dir / "val_confusion_matrix.csv",
            exp_dir / "val_confusion_matrix.png",
            f_val_cm_arr,
            title=f"{run_name} — Val (best epoch {best_epoch})",
        )
        save_json(
            exp_dir / "val_metrics.json",
            {
                "best_epoch": best_epoch,
                "main_metric": args.main_metric,
                "val_qwk": float(f_val_qwk),
                "val_mae": float(f_val_mae),
                "val_accuracy": float(f_val_acc),
                "val_f1_macro": float(f_val_f1),
                "val_macro_recall": float(f_val_macro_recall),
                "val_loss": float(f_val_loss),
                "val_recall_per_class": {
                    n: float(r) for n, r in zip(CLASS_NAMES, f_val_recall)
                },
            },
        )

        (
            test_loss,
            test_acc,
            test_f1,
            test_recall,
            test_macro_recall,
            test_qwk,
            test_mae,
            test_cm_str,
            test_report_str,
            test_slide_ids,
            test_labels_list,
            test_preds_list,
            test_raw_list,
            test_cm_arr,
        ) = validate_and_evaluate(
            model,
            test_loader,
            criterion,
            device,
            desc="  Test ",
            formulation=args.formulation,
            return_predictions=True,
        )

        log(f"\n{test_cm_str}", log_file)
        log(f"\n{test_report_str}", log_file)

        # Persist test artifacts
        save_predictions_csv(
            exp_dir / "test_predictions.csv",
            test_slide_ids,
            test_labels_list,
            test_preds_list,
            test_raw_list,
        )
        save_confusion_matrix(
            exp_dir / "test_confusion_matrix.csv",
            exp_dir / "test_confusion_matrix.png",
            test_cm_arr,
            title=f"{run_name} — Test",
        )
        save_json(
            exp_dir / "test_metrics.json",
            {
                "test_qwk": float(test_qwk),
                "test_mae": float(test_mae),
                "test_accuracy": float(test_acc),
                "test_f1_macro": float(test_f1),
                "test_macro_recall": float(test_macro_recall),
                "test_loss": float(test_loss),
                "test_recall_per_class": {
                    n: float(r) for n, r in zip(CLASS_NAMES, test_recall)
                },
            },
        )
        test_metrics = {
            "test_qwk": float(test_qwk),
            "test_mae": float(test_mae),
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
            "test_macro_recall": float(test_macro_recall),
            "test_loss": float(test_loss),
        }

        log(f"\nFINAL RESULTS:", log_file)
        log(
            f"  Best Val {args.main_metric.upper()}:{' ' * max(1, 13 - len(args.main_metric))}{best_metric_val:.3f} (Epoch {best_epoch})",
            log_file,
        )
        log(f"  Test Accuracy:         {test_acc:.2f}%", log_file)
        log(f"  Test F1 (Macro):       {test_f1:.3f}", log_file)
        log(f"  Test Macro Recall:     {test_macro_recall:.2f}%", log_file)
        log(f"  Test QWK:              {test_qwk:.3f}", log_file)
        log(f"  Test MAE:              {test_mae:.3f}", log_file)
        log(f"  Test Loss:             {test_loss:.4f}", log_file)
    else:
        log("ERROR: Best checkpoint not found. Cannot run test phase.", log_file)

    # ── Method note + leaderboard append ──────────────────────────────
    write_method_note(
        exp_dir / "method_note.md",
        args,
        model_name=model.__class__.__name__,
    )

    # Read best val metrics back
    val_metrics_path = exp_dir / "val_metrics.json"
    val_metrics = (
        json.loads(val_metrics_path.read_text()) if val_metrics_path.exists() else {}
    )

    leaderboard_row = {
        "timestamp": timestamp,
        "run_name": run_name,
        "experiment_dir": str(exp_dir),
        "model_type": args.model_type,
        "backbone": args.backbone,
        "seed": args.seed,
        "split_seed": args.seed,
        "formulation": args.formulation,
        "main_metric": args.main_metric,
        "best_epoch": val_metrics.get("best_epoch", best_epoch),
        "val_qwk": f"{val_metrics.get('val_qwk', 0.0):.6f}",
        "val_mae": f"{val_metrics.get('val_mae', 0.0):.6f}",
        "val_accuracy": f"{val_metrics.get('val_accuracy', 0.0):.6f}",
        "val_f1_macro": f"{val_metrics.get('val_f1_macro', 0.0):.6f}",
        "val_macro_recall": f"{val_metrics.get('val_macro_recall', 0.0):.6f}",
        "val_loss": f"{val_metrics.get('val_loss', 0.0):.6f}",
        "test_qwk": f"{test_metrics.get('test_qwk', float('nan')):.6f}" if test_metrics else "",
        "test_mae": f"{test_metrics.get('test_mae', float('nan')):.6f}" if test_metrics else "",
        "test_accuracy": f"{test_metrics.get('test_accuracy', float('nan')):.6f}" if test_metrics else "",
        "test_f1_macro": f"{test_metrics.get('test_f1_macro', float('nan')):.6f}" if test_metrics else "",
        "test_macro_recall": f"{test_metrics.get('test_macro_recall', float('nan')):.6f}" if test_metrics else "",
        "test_loss": f"{test_metrics.get('test_loss', float('nan')):.6f}" if test_metrics else "",
        "checkpoint": str(best_ckpt_path),
    }
    leaderboard_path = Path(args.leaderboard_path)
    if not leaderboard_path.is_absolute():
        from core.config import PROJECT_ROOT

        leaderboard_path = PROJECT_ROOT / leaderboard_path
    # Leaderboard write is opt-in; pass --leaderboard to record this run.
    if args.leaderboard:
        append_leaderboard(leaderboard_path, leaderboard_row, LEADERBOARD_FIELDS)
        log(f"\nAppended leaderboard row -> {leaderboard_path}", log_file)

    log(f"\n{'=' * 60}", log_file)
    log(f"Experiment Complete.", log_file)


if __name__ == "__main__":
    main()
