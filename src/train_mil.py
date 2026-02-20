"""
Training pipeline for MIL WSI classification.

Supports:
    - SimpleGatedMIL: Lightweight gated-attention MIL (recommended for small datasets)
    - DTFD-MIL: Double-Tier Feature Distillation (Zhang et al., CVPR 2022)

Trains models on pre-extracted backbone features to classify
Whole Slide Images into 3 MPN subtypes: ET, PV, PMF.

Usage:
    python -m src.train_mil --backbone titan --model_type simple --epochs 50
    python -m src.train_mil --backbone titan --model_type dtfd --num_pseudo_bags 8
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    recall_score,
)
from tqdm import tqdm

from core.config import CLASS_MAP, CLASS_MAP_INV, EXPERIMENTS_DIR, SEED
from core.utils import FocalLoss, SupConLoss
from data.bag_dataset import MPNBagDatasetFull
from models.clam import CLAM_SB
from models.dtfd_mil import DTFDMIL, compute_dtfd_loss
from models.hybrid_mil import HybridMIL
from models.simple_mil import SimpleGatedMIL

# ── backbone configuration ───────────────────────────────────────────────
BACKBONE_CONFIG: Dict[str, Dict] = {
    "titan": {
        "dim": 768,
        "feature_dir": "features_titan",
        "display_name": "TITAN",
    },
    "uni2": {
        "dim": 1536,
        "feature_dir": "features_uni2",
        "display_name": "UNI2-h",
    },
    "virchow": {
        "dim": 1280,
        "feature_dir": "features_virchow2",
        "display_name": "Virchow2",
    },
}

CLASS_NAMES = [CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))]

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
    dataset: MPNBagDatasetFull,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split by patient ID and disease label.

    Groups patients by their class label, then performs the 70/15/15 split
    independently within each class to guarantee every class is represented
    in Train, Val, and Test (provided the class has >= 3 patients).

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    rng = random.Random(seed)

    # Map patient -> (sample indices, disease label)
    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    patient_to_label: Dict[str, int] = {}
    for idx, (pt_path, label) in enumerate(dataset.samples):
        patient_id = pt_path.parent.name
        patient_to_indices[patient_id].append(idx)
        patient_to_label[patient_id] = label

    # Group patients by their disease label
    label_to_patients: Dict[int, List[str]] = defaultdict(list)
    for patient_id, label in patient_to_label.items():
        label_to_patients[label].append(patient_id)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label in sorted(label_to_patients.keys()):
        patients = label_to_patients[label]
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Guarantee at least 1 patient in val and test when possible
        if n >= 3:
            n_val = max(n_val, 1)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                n_train = n - n_val - n_test
        else:
            n_test = n - n_train - n_val

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
    criterion_supcon: Optional[nn.Module] = None,
    supcon_weight: float = 0.5,
) -> Tuple[float, float, float, float, float, List[float], dict]:
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

        # Collect batch-level embeddings for SupConLoss (simple/hybrid only)
        batch_logits_list = []
        batch_embeddings_list = []
        batch_labels_list = []

        for i, features in enumerate(features_list):
            features = features.to(device)  # [N, D]
            label = labels[i].item()

            if model_type == "dtfd":
                bag_logits, pseudo_bag_logits, instance_logits_list, bag_embedding = (
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
                batch_loss = batch_loss + loss
                # Collect for SupCon
                bag_emb_norm = torch.nn.functional.normalize(bag_embedding, dim=0)
                batch_embeddings_list.append(bag_emb_norm)
                batch_labels_list.append(label)
            elif model_type == "clam_sb":
                # CLAM-SB: bag loss + instance clustering loss
                logits, inst_dict = model.forward_training(features, bag_label=label)
                label_tensor = torch.tensor([label], device=device)
                bag_loss = criterion(logits.unsqueeze(0), label_tensor)
                inst_loss = inst_dict["inst_loss"]
                loss = bag_weight * bag_loss + inst_weight * inst_loss
                loss_sums["bag_loss"] += bag_loss.item()
                loss_sums["inst_loss"] += inst_loss.item()
                batch_loss = batch_loss + loss
            else:
                # Simple / Hybrid MIL: collect logits & embeddings
                logits, _, bag_embedding = model(features)
                batch_logits_list.append(logits.unsqueeze(0))  # [1, C]
                # L2-normalize bag embedding for contrastive learning
                bag_emb_norm = torch.nn.functional.normalize(bag_embedding, dim=0)
                batch_embeddings_list.append(bag_emb_norm)
                batch_labels_list.append(label)

            pred = (bag_logits if model_type == "dtfd" else logits).argmax().item()
            batch_correct += int(pred == label)
            all_preds.append(pred)
            all_labels.append(label)

        batch_size = len(features_list)

        # 1. Classification Loss (Only for Simple/Hybrid)
        if batch_logits_list:
            stacked_logits = torch.cat(batch_logits_list, dim=0)  # [B, C]
            stacked_labels_cls = torch.tensor(batch_labels_list, device=device)
            loss_cls = criterion(stacked_logits, stacked_labels_cls)
            loss_sums["cls_loss"] += loss_cls.item() * batch_size
            batch_loss = batch_loss + loss_cls

        # 2. Average the accumulated sums (Only for DTFD/CLAM)
        if model_type in ("dtfd", "clam_sb"):
            batch_loss = batch_loss / batch_size

        # 3. SupConLoss (For all applicable models)
        if criterion_supcon is not None and len(batch_embeddings_list) >= 2:
            stacked_labels_sup = torch.tensor(batch_labels_list, device=device)
            stacked_emb = torch.stack(batch_embeddings_list)  # [B, D]
            stacked_emb = stacked_emb.unsqueeze(1)  # [B, 1, D]
            loss_sup = criterion_supcon(stacked_emb, stacked_labels_sup)
            loss_sums["supcon_loss"] += loss_sup.item() * batch_size
            # Add directly without dividing, as loss_sup is already a batch mean
            batch_loss = batch_loss + (supcon_weight * loss_sup)

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
    train_f2 = fbeta_score(
        all_labels, all_preds, beta=2, average="macro", zero_division=0
    )
    train_bacc = 100.0 * balanced_accuracy_score(all_labels, all_preds)

    # Per-class recall
    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    recall_list = [100.0 * r for r in per_class_recall]

    avg_components = {k: v / total for k, v in loss_sums.items()}
    return (
        avg_loss,
        avg_acc,
        train_f1,
        train_f2,
        train_bacc,
        recall_list,
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
) -> Tuple[float, float, float, float, float, List[float], str, str]:
    """
    Validate/Test a MIL model and return metric strings.

    Returns:
        avg_loss: Average loss over all samples.
        accuracy: Accuracy as a percentage.
        f1_macro: Macro-averaged F1 score.
        f2_macro: Macro-averaged F2 score.
        balanced_acc: Balanced accuracy as a percentage.
        recall_list: Per-class recall as percentages.
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

            # Safely handle models returning any number of items
            outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # [C] → [1, C]

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
    f2_macro = fbeta_score(
        all_labels, all_preds, beta=2, average="macro", zero_division=0
    )
    balanced_acc = 100.0 * balanced_accuracy_score(all_labels, all_preds)

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

    return (
        avg_loss,
        accuracy,
        f1_macro,
        f2_macro,
        balanced_acc,
        recall_list,
        cm_str,
        report_str,
    )


# ── argument parsing ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MIL models on pre-extracted WSI features."
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
        choices=["simple", "dtfd", "clam_sb", "hybrid"],
        help="MIL model type: 'simple' | 'dtfd' | 'clam_sb' | 'hybrid'. Default: simple.",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
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
        "--max_patches",
        type=int,
        default=None,
        help="Max patches per bag (for memory management).",
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

    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Setup Experiment Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build unified run_name for directory, log, and checkpoint
    run_name = f"{args.model_type}_{args.backbone}"
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
    num_classes = len(CLASS_MAP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}", log_file)

    # ── Data ──────────────────────────────────────────────────────────
    log(f"\nLoading {display_name} features from: {features_dir}", log_file)
    base_dataset = MPNBagDatasetFull(
        features_dir, max_patches=args.max_patches, is_train=False
    )
    log(f"  Total bags: {len(base_dataset)}", log_file)

    train_idx, val_idx, test_idx = patient_split(base_dataset, seed=args.seed)
    log(
        f"  Split -- Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}",
        log_file,
    )

    # Separate datasets: stochastic for training, deterministic for eval
    train_dataset = MPNBagDatasetFull(
        features_dir, max_patches=args.max_patches, is_train=True
    )
    eval_dataset = MPNBagDatasetFull(
        features_dir, max_patches=args.max_patches, is_train=False
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bags,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_idx),
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

    # Class weights mapped dynamically from CLASS_MAP
    # ET = 1.5, PV = 3.5, PMF = 1.0
    _weight_map = {"ET": 1.5, "PV": 3.5, "PMF": 1.0}
    num_classes = len(CLASS_MAP)
    _weights = [1.0] * num_classes
    for name, idx in CLASS_MAP.items():
        _weights[idx] = _weight_map.get(name, 1.0)
    class_weights = torch.tensor(_weights, dtype=torch.float32, device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean").to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # SupConLoss for simple/hybrid/dtfd models
    criterion_supcon = None
    supcon_weight = 0.5
    if args.model_type in ("simple", "hybrid", "dtfd"):
        criterion_supcon = SupConLoss(temperature=0.1)
        log(f"\nSupConLoss: enabled (weight={supcon_weight}, tau=0.1)", log_file)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Debug: verify active class weights
    weights_str = "  ".join(
        f"{n}={w:.1f}" for n, w in zip(CLASS_NAMES, class_weights.tolist())
    )
    log(f"\nLoss: {criterion.__class__.__name__}", log_file)
    log(f"Weights: {weights_str}", log_file)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_f2 = 0.0
    best_epoch = 0

    log(f"\n{'=' * 60}", log_file)
    log(f"Start Training ({args.epochs} epochs)", log_file)
    log(f"{'=' * 60}", log_file)

    # Table header
    hdr = "Ep   | Mode  | Loss  | Acc   | F1    | F2    | B.Acc | Recall ( ET / PV / PMF )"
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
            train_f2,
            train_bacc,
            train_recall,
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
            criterion_supcon=criterion_supcon,
            supcon_weight=supcon_weight,
        )

        # Validate
        (
            val_loss,
            val_acc,
            val_f1,
            val_f2,
            val_bacc,
            val_recall,
            val_cm_str,
            val_report_str,
        ) = validate_and_evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # Per-epoch logging (2-line table)
        ep_str = f"{epoch}/{args.epochs}"
        log(
            f"{ep_str:<5}| Train | {train_loss:<5.3f} | {train_acc:<5.1f} "
            f"| {train_f1:<5.3f} | {train_f2:<5.3f} | {train_bacc:<5.1f} | {fmt_recall(train_recall)}",
            log_file,
        )
        log(
            f"     | Val   | {val_loss:<5.3f} | {val_acc:<5.1f} "
            f"| {val_f1:<5.3f} | {val_f2:<5.3f} | {val_bacc:<5.1f} | {fmt_recall(val_recall)}",
            log_file,
        )
        log(sep, log_file)

        scheduler.step()

        # Save best model (based on val F2-score)
        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f2": val_f2,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "test_idx": test_idx,
                "backbone": args.backbone,
                "model_type": args.model_type,
                "args": vars(args),
            }
            torch.save(checkpoint, exp_dir / checkpoint_name)
            log(
                f"     >>> ⭐ New Best Model! Val F2: {val_f2:.3f} | Acc: {val_acc:.1f}%",
                log_file,
            )
            log(f"\n{val_cm_str}", log_file)
            log(f"\n{val_report_str}", log_file)

    # ── Test Phase ────────────────────────────────────────────────────
    log(f"\n{'=' * 60}", log_file)
    log(f"Final Evaluation on TEST SET", log_file)
    log(f"{'=' * 60}", log_file)

    best_ckpt_path = exp_dir / checkpoint_name
    if best_ckpt_path.exists():
        log(f"Loading best checkpoint from: {best_ckpt_path}", log_file)
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        (
            test_loss,
            test_acc,
            test_f1,
            test_f2,
            test_bacc,
            test_recall,
            test_cm_str,
            test_report_str,
        ) = validate_and_evaluate(
            model,
            test_loader,
            criterion,
            device,
            desc="  Test ",
        )

        log(f"\n{test_cm_str}", log_file)
        log(f"\n{test_report_str}", log_file)

        log(f"\nFINAL RESULTS:", log_file)
        log(f"  Best Validation F2:  {best_val_f2:.3f} (Epoch {best_epoch})", log_file)
        log(f"  Test Accuracy:       {test_acc:.2f}%", log_file)
        log(f"  Test F1 (Macro):     {test_f1:.3f}", log_file)
        log(f"  Test F2 (Macro):     {test_f2:.3f}", log_file)
        log(f"  Test Balanced Acc:   {test_bacc:.2f}%", log_file)
        log(f"  Test Loss:           {test_loss:.4f}", log_file)
    else:
        log("ERROR: Best checkpoint not found. Cannot run test phase.", log_file)

    log(f"\n{'=' * 60}", log_file)
    log(f"Experiment Complete.", log_file)


if __name__ == "__main__":
    main()
