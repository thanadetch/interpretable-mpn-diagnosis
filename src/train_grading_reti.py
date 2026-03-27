"""
Training pipeline for MIL Reticulin Fibrosis Grading (4-class).

Supports:
    - SimpleGatedMIL: Lightweight gated-attention MIL (recommended for small datasets)
    - DTFD-MIL: Double-Tier Feature Distillation (Zhang et al., CVPR 2022)

Trains models on pre-extracted reticulin backbone features to classify
Whole Slide Images into 4 fibrosis grades: G0, G1, G2, G3.

Usage:
    python -m src.train_grading_reti --backbone titan --model_type simple --epochs 50
    python -m src.train_grading_reti --backbone titan --model_type dtfd --num_pseudo_bags 8
"""

import argparse
import random
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
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
from models.clam import CLAM_SB
from models.dtfd_mil import DTFDMIL, compute_dtfd_loss
from models.hybrid_mil import HybridMIL
from models.simple_mil import SimpleGatedMIL
from models.dual_stream_mil import DualStreamMIL
from models.multi_branch_mil import MultiBranchMIL
from models.mean_pool_mil import MeanPoolMIL

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
    "virchow2": {
        "dim": 1280,
        "feature_dir": "features_virchow2_reti",
        "display_name": "Virchow2",
    },
}

# Suppress sklearn warning when y_pred contains classes absent from y_true
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


# ── helpers ───────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def log(msg: str, log_file: Optional[Path] = None) -> None:
    """Print to console and optionally append to a log file."""
    print(msg)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


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

    Dataset: 42 patients -> Train 26, Val 8, Test 8.

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
        0: (1, 1),  # G0 -> Val 1, Test 1 (Leaves 2 for Train)
        1: (2, 2),  # G1 -> Val 2, Test 2 (Leaves 11 for Train)
        2: (3, 3),  # G2 -> Val 3, Test 3 (Leaves 12 for Train)
        3: (1, 1),  # G3 -> Val 1, Test 1 (Leaves 3 for Train)
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
    tier1_weight: float = 0.5,
    instance_weight: float = 0.2,
    bag_weight: float = 0.7,
    inst_weight: float = 0.3,
    formulation: str = "classification",
) -> Tuple[float, float, float, List[float], float, float, dict]:
    """Train for one epoch. Handles 'simple', 'dtfd', and 'clam_sb' model types."""
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

            if model_type == "dtfd":
                bag_logits, pseudo_bag_logits, instance_logits_list = (
                    model.forward_training(features)
                )
                loss, loss_dict = compute_dtfd_loss(
                    bag_logits=bag_logits,
                    pseudo_bag_logits=pseudo_bag_logits,
                    instance_logits_list=instance_logits_list,
                    bag_label=label,
                    criterion=criterion,
                    tier1_weight=tier1_weight,
                    instance_weight=instance_weight,
                )
                for key, value in loss_dict.items():
                    loss_sums[key] += value
            elif model_type == "clam_sb":
                # CLAM-SB: bag loss + instance clustering loss
                logits, inst_dict = model.forward_training(features, bag_label=label)
                label_tensor = torch.tensor([label], device=device)
                bag_loss = criterion(logits.unsqueeze(0), label_tensor)
                inst_loss = inst_dict["inst_loss"]
                loss = bag_weight * bag_loss + inst_weight * inst_loss
                loss_sums["bag_loss"] += bag_loss.item()
                loss_sums["inst_loss"] += inst_loss.item()
            else:
                # Standard MIL models (simple, hybrid, dual_stream, multi_branch)
                logits, _, _ = model(features)
                if formulation == "regression":
                    label_tensor = torch.tensor(
                        [label], dtype=torch.float32, device=device
                    )
                    loss = criterion(logits.view(-1), label_tensor)
                else:
                    label_tensor = torch.tensor([label], device=device)
                    loss = criterion(logits.unsqueeze(0), label_tensor)
                loss_sums["bag_loss"] += loss.item()

            batch_loss = batch_loss + loss

            if formulation == "regression":
                pred = int(max(0, min(3, round(logits.view(-1).item()))))
            else:
                pred = (bag_logits if model_type == "dtfd" else logits).argmax().item()
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
    train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class recall
    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    recall_list = [100.0 * r for r in per_class_recall]

    # QWK (Quadratic Weighted Kappa) — primary metric
    train_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    # MAE (Mean Absolute Error) — ordinal distance metric
    train_mae = mean_absolute_error(all_labels, all_preds)

    avg_components = {k: v / total for k, v in loss_sums.items()}
    return (
        avg_loss,
        avg_acc,
        train_f1,
        recall_list,
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
) -> Tuple[float, float, float, List[float], float, float, str, str]:
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

                pred_val = int(max(0, min(3, round(logits.view(-1).item()))))
                pred = torch.tensor([pred_val], device=device)
                correct += pred.eq(label).sum().item()
                total += 1

                all_preds.append(pred_val)
                all_labels.append(label.item())
            else:
                logits = logits.unsqueeze(0)

                loss = criterion(logits, label)
                running_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += pred.eq(label).sum().item()
                total += 1

                all_preds.append(pred.item())
                all_labels.append(label.item())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    # Compute imbalance-aware metrics
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

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

    # QWK (Quadratic Weighted Kappa) — primary metric
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    # MAE (Mean Absolute Error) — ordinal distance metric
    mae = mean_absolute_error(all_labels, all_preds)

    return (
        avg_loss,
        accuracy,
        f1_macro,
        recall_list,
        qwk,
        mae,
        cm_str,
        report_str,
    )


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
            "dtfd",
            "clam_sb",
            "hybrid",
            "dual_stream",
            "multi_branch",
            "mean_pool",
        ],
        help="MIL model type: 'simple' | 'dtfd' | 'clam_sb' | 'hybrid' | 'dual_stream' | 'multi_branch' | 'mean_pool'. Default: simple.",
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

    # DTFD-MIL specific arguments
    parser.add_argument(
        "--num_pseudo_bags",
        type=int,
        default=8,
        help="Number of pseudo-bags for DTFD-MIL (default: 8).",
    )
    parser.add_argument(
        "--tier1_weight",
        type=float,
        default=0.5,
        help="Weight for pseudo-bag auxiliary loss (default: 0.5).",
    )
    parser.add_argument(
        "--instance_weight",
        type=float,
        default=0.2,
        help="Weight for instance-level loss (default: 0.2). Set to 0.0 to disable.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Top-k pooling: use mean of k highest-attention patches. 0 = standard attention (default: 0).",
    )

    # CLAM-SB specific arguments
    parser.add_argument(
        "--bag_weight",
        type=float,
        default=0.7,
        help="Weight for bag-level loss in CLAM-SB (default: 0.7).",
    )
    parser.add_argument(
        "--inst_weight",
        type=float,
        default=0.3,
        help="Weight for instance clustering loss in CLAM-SB (default: 0.3).",
    )

    # Experiment tracking
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

    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Setup Experiment Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build unified run_name for directory, log, and checkpoint
    run_name = f"reti_{args.model_type}_{args.backbone}"
    if args.model_type == "simple" and args.topk > 0:
        run_name += f"_topk{args.topk}"
    if args.postfix:
        run_name += f"_{args.postfix}"

    exp_dir = EXPERIMENTS_DIR / f"{run_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_file = exp_dir / f"{run_name}.log"

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}", log_file)

    # ── Data ──────────────────────────────────────────────────────────
    log(f"\nLoading {display_name} features from: {features_dir}", log_file)
    full_dataset = GradingBagDatasetFull(features_dir)
    log(f"  Total bags: {len(full_dataset)}", log_file)

    train_idx, val_idx, test_idx = patient_split(full_dataset, seed=args.seed)
    log(
        f"  Split -- Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}",
        log_file,
    )

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

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
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

    # ── Model ─────────────────────────────────────────────────────────
    checkpoint_name = f"best_{run_name}.pth"

    if args.model_type == "simple":
        model = SimpleGatedMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=args.topk,
        ).to(device)
    elif args.model_type == "clam_sb":
        model = CLAM_SB(
            input_dim=input_dim,
            num_classes=num_classes,
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
    else:
        model = DTFDMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            num_pseudo_bags=args.num_pseudo_bags,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\nModel: {args.model_type.upper()}", log_file)
    log(
        f"  Parameters: {trainable_params:,} trainable / {total_params:,} total",
        log_file,
    )

    # ── Optimiser & loss ──────────────────────────────────────────────
    if args.formulation == "regression":
        criterion = nn.SmoothL1Loss()
    else:
        # Class weights: G0=1.0, G1=1.0, G2=1.0, G3=1.0
        # G0 and G3 get higher weights to handle severe minority classes
        class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
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
    best_qwk = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    log(f"\n{'=' * 60}", log_file)
    log(f"Start Training ({args.epochs} epochs)", log_file)
    log(f"{'=' * 60}", log_file)

    # Table header
    hdr = "Ep   | Mode  | Loss  | Acc   | F1    | MAE   | QWK   | Recall ( G0 / G1 / G2 / G3 )"
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
            tier1_weight=args.tier1_weight,
            instance_weight=args.instance_weight,
            bag_weight=args.bag_weight,
            inst_weight=args.inst_weight,
            formulation=args.formulation,
        )

        # Validate
        (
            val_loss,
            val_acc,
            val_f1,
            val_recall,
            val_qwk,
            val_mae,
            val_cm_str,
            val_report_str,
        ) = validate_and_evaluate(
            model,
            val_loader,
            criterion,
            device,
            formulation=args.formulation,
        )

        # Per-epoch logging (2-line table)
        ep_str = f"{epoch}/{args.epochs}"
        log(
            f"{ep_str:<5}| Train | {train_loss:<5.3f} | {train_acc:<5.1f} "
            f"| {train_f1:<5.3f} | {train_mae:<5.3f} | {train_qwk:<5.3f} | {fmt_recall(train_recall)}",
            log_file,
        )
        log(
            f"     | Val   | {val_loss:<5.3f} | {val_acc:<5.1f} "
            f"| {val_f1:<5.3f} | {val_mae:<5.3f} | {val_qwk:<5.3f} | {fmt_recall(val_recall)}",
            log_file,
        )
        log(sep, log_file)

        scheduler.step()

        # Save best model (based on val QWK)
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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
                f"     >>> ⭐ New Best Model! Val QWK: {val_qwk:.3f} | Acc: {val_acc:.1f}%",
                log_file,
            )
            log(f"\n{val_cm_str}", log_file)
            log(f"\n{val_report_str}", log_file)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                log(
                    f"\nEarly stopping triggered at epoch {epoch} "
                    f"(no val QWK improvement for {args.early_stop_patience} epochs)",
                    log_file,
                )
                break

    # ── Test Phase ────────────────────────────────────────────────────
    log(f"\n{'=' * 60}", log_file)
    log(f"Final Evaluation on TEST SET", log_file)
    log(f"{'=' * 60}", log_file)

    best_ckpt_path = exp_dir / checkpoint_name
    if best_ckpt_path.exists():
        log(f"Loading best checkpoint from: {best_ckpt_path}", log_file)
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        (
            test_loss,
            test_acc,
            test_f1,
            test_recall,
            test_qwk,
            test_mae,
            test_cm_str,
            test_report_str,
        ) = validate_and_evaluate(
            model,
            test_loader,
            criterion,
            device,
            desc="  Test ",
            formulation=args.formulation,
        )

        log(f"\n{test_cm_str}", log_file)
        log(f"\n{test_report_str}", log_file)

        log(f"\nFINAL RESULTS:", log_file)
        log(
            f"  Best Val QWK:          {best_qwk:.3f} (Epoch {best_epoch})",
            log_file,
        )
        log(f"  Test Accuracy:         {test_acc:.2f}%", log_file)
        log(f"  Test F1 (Macro):       {test_f1:.3f}", log_file)
        log(f"  Test QWK:              {test_qwk:.3f}", log_file)
        log(f"  Test MAE:              {test_mae:.3f}", log_file)
        log(f"  Test Loss:             {test_loss:.4f}", log_file)
    else:
        log("ERROR: Best checkpoint not found. Cannot run test phase.", log_file)

    log(f"\n{'=' * 60}", log_file)
    log(f"Experiment Complete.", log_file)


if __name__ == "__main__":
    main()
