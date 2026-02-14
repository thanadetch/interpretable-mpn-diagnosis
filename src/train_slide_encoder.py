"""
Transformer MIL Training for WSI Classification.

Trains a Transformer head on pre-extracted TITAN/CONCHv1.5 features.
Each .pt file is one "bag" (all patches from one source image).

Usage:
    python -m src.train_slide_encoder \
        --features_dir data/features_titan \
        --epochs 50 \
        --lr 1e-4 \
        --seed 42 \
        --device cuda
"""

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
)
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.config import (
    CLASS_MAP,
    CLASS_MAP_INV,
    EXPERIMENTS_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


# =============================================================================
# Logging
# =============================================================================


class TeeLogger:
    """Tee stdout to both terminal and a log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =============================================================================
# Dataset
# =============================================================================


class FeatureBagDataset(Dataset):
    """
    Dataset that loads pre-extracted .pt feature files.

    Each .pt file is a tensor of shape [N_patches, 768] representing
    all patch features from one source image (one "bag").

    Args:
        file_list: List of (pt_file_path, label) tuples.
    """

    def __init__(self, file_list: List[Tuple[Path, int]]) -> None:
        self.samples = file_list

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            Tuple of (features [N, 768], label, file_path_string).
        """
        fpath, label = self.samples[idx]
        features = torch.load(fpath, weights_only=True)
        return features, label, str(fpath)


# =============================================================================
# Model: Transformer MIL
# =============================================================================


class TransformerMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning head.

    Architecture:
        1. Learnable [CLS] token prepended to the patch sequence
        2. nn.TransformerEncoder (multi-head self-attention)
        3. LayerNorm + Linear classifier on [CLS] output

    Args:
        feature_dim: Dimension of input patch features (768 for CONCHv1.5).
        num_classes: Number of output classes.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        ff_dim: Feedforward dimension in Transformer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 3,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch features of shape [1, N_patches, 768].

        Returns:
            Logits of shape [1, num_classes].
        """
        B = x.shape[0]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, D]

        # Transformer
        x = self.transformer(x)

        # Classify from [CLS] token
        cls_out = x[:, 0]  # [B, D]
        cls_out = self.norm(cls_out)
        logits = self.head(cls_out)
        return logits


# =============================================================================
# Data Splitting (Patient-Level)
# =============================================================================


def _extract_patient_id(filepath: Path) -> str:
    """
    Extract patient ID from a .pt filename.

    Filename format: {PatientID}_{ImgID}.pt
    e.g., "ET1_G1_3.pt" → "ET1_G1"

    The patient ID is everything except the last underscore-separated segment.
    """
    stem = filepath.stem  # e.g., "ET1_G1_3"
    parts = stem.rsplit("_", 1)  # ["ET1_G1", "3"]
    return parts[0] if len(parts) > 1 else stem


def get_feature_split(
    features_dir: Path,
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
    Split .pt feature files at the PATIENT level.

    Mirrors the stratification logic from core.utils.get_patient_split
    but operates on the feature file directory structure.

    Args:
        features_dir: Root dir with {Class}/ subdirectories containing .pt files.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for test.
        seed: Random seed.

    Returns:
        Tuple of (train_files, val_files, test_files).
        Each is a list of (pt_file_path, label).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Discover all .pt files and their labels
    # Structure: features_dir/{Class}/*.pt
    all_files: List[Tuple[Path, int]] = []
    for class_name, label in CLASS_MAP.items():
        class_dir = features_dir / class_name
        if not class_dir.exists():
            print(f"Warning: class directory {class_dir} not found, skipping.")
            continue
        for pt_file in sorted(class_dir.glob("*.pt")):
            all_files.append((pt_file, label))

    if not all_files:
        raise ValueError(f"No .pt files found in {features_dir}")

    # Group files by patient
    patient_to_files: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    patient_to_label: Dict[str, int] = {}
    for fpath, label in all_files:
        pid = _extract_patient_id(fpath)
        patient_to_files[pid].append((fpath, label))
        patient_to_label[pid] = label

    # Group patients by class for stratification
    patients_by_class: Dict[int, List[str]] = defaultdict(list)
    for pid, label in patient_to_label.items():
        patients_by_class[label].append(pid)

    random.seed(seed)

    train_patients, val_patients, test_patients = [], [], []

    for label in sorted(patients_by_class.keys()):
        class_patients = sorted(patients_by_class[label])
        random.shuffle(class_patients)
        n = len(class_patients)

        if n == 1:
            train_patients.extend(class_patients)
        elif n == 2:
            train_patients.append(class_patients[0])
            test_patients.append(class_patients[1])
        elif n < 5:
            test_patients.append(class_patients[0])
            train_patients.append(class_patients[1])
            if n >= 3:
                val_patients.append(class_patients[2])
            if n >= 4:
                train_patients.append(class_patients[3])
        else:
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = 1
                n_val = max(0, n - n_test - n_train)
            test_patients.extend(class_patients[:n_test])
            val_patients.extend(class_patients[n_test : n_test + n_val])
            train_patients.extend(class_patients[n_test + n_val :])

    # Flatten patient IDs → file lists
    def patients_to_files(patient_ids: List[str]) -> List[Tuple[Path, int]]:
        result = []
        for pid in patient_ids:
            result.extend(patient_to_files[pid])
        return result

    train_files = patients_to_files(train_patients)
    val_files = patients_to_files(val_patients)
    test_files = patients_to_files(test_patients)

    # Log split info
    print(f"\n{'=' * 60}")
    print("Patient-Level Split (Feature Files)")
    print(f"{'=' * 60}")
    print(f"  Train : {len(train_patients):3d} patients → {len(train_files):4d} bags")
    print(f"  Val   : {len(val_patients):3d} patients → {len(val_files):4d} bags")
    print(f"  Test  : {len(test_patients):3d} patients → {len(test_files):4d} bags")
    print(f"  Total : {len(patient_to_label):3d} patients → {len(all_files):4d} bags")
    print(f"{'=' * 60}\n")

    return train_files, val_files, test_files


# =============================================================================
# Collate (variable-length bags)
# =============================================================================


def collate_bags(batch):
    """
    Custom collate for variable-length bags with batch_size=1.

    Args:
        batch: List of (features [N, 768], label, path) tuples.

    Returns:
        (features [1, N, 768], labels [1], paths).
    """
    features, labels, paths = zip(*batch)
    # With batch_size=1, just unsqueeze
    features = features[0].unsqueeze(0)  # [1, N, 768]
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels, paths


# =============================================================================
# Training & Validation
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels, _ in tqdm(loader, desc="  Train", leave=False):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Validate the model.

    Returns:
        (avg_loss, accuracy, f2_macro, confusion_matrix)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels, _ in tqdm(loader, desc="  Val  ", leave=False):
        features = features.to(device)
        labels = labels.to(device)

        with torch.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
            logits = model(features)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f2_macro = fbeta_score(
        all_labels, all_preds, beta=2, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return avg_loss, accuracy, f2_macro, cm


# =============================================================================
# Main Training Function
# =============================================================================


def train(args: argparse.Namespace) -> Dict:
    """Main training loop."""
    device = torch.device(args.device)
    num_classes = len(CLASS_MAP)

    # ------------------------------------------------------------------
    # Setup experiment directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"titan_mil_{timestamp}"
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = TeeLogger(str(exp_dir / "train.log"))

    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Args: {vars(args)}\n")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    features_dir = Path(args.features_dir)
    train_files, val_files, test_files = get_feature_split(features_dir, seed=args.seed)

    train_dataset = FeatureBagDataset(train_files)
    val_dataset = FeatureBagDataset(val_files)
    test_dataset = FeatureBagDataset(test_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_bags,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_bags,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_bags,
        pin_memory=device.type == "cuda",
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TransformerMIL(
        feature_dim=768,
        num_classes=num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransformerMIL — Trainable parameters: {total_params:,}")
    print(
        f"  Layers: {args.num_layers}, Heads: {args.num_heads}, "
        f"FF dim: {args.ff_dim}, Dropout: {args.dropout}\n"
    )

    # ------------------------------------------------------------------
    # Optimizer, Scheduler, Loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    best_f2 = 0.0
    best_epoch = 0
    patience_counter = 0

    print(
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>7} | {'Val F2':>6} | {'LR':>10}"
    )
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc, val_f2, val_cm = validate(
            model, val_loader, criterion, device, num_classes
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} | "
            f"{val_loss:8.4f} | {val_acc:6.1%} | {val_f2:6.3f} | "
            f"{lr:10.2e}  ({elapsed:.1f}s)"
        )

        # Save best model by F2
        if val_f2 > best_f2:
            best_f2 = val_f2
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f2": val_f2,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                exp_dir / "best_model.pt",
            )
            print(f"  ★ New best F2={val_f2:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ------------------------------------------------------------------
    # Test with best model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Testing with best model from epoch {best_epoch} (F2={best_f2:.4f})")
    print(f"{'=' * 60}")

    checkpoint = torch.load(exp_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f2, test_cm = validate(
        model, test_loader, criterion, device, num_classes
    )

    # Print results
    class_names = [CLASS_MAP_INV[i] for i in range(num_classes)]

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.1%}")
    print(f"Test F2  : {test_f2:.4f}")

    print(f"\nConfusion Matrix:")
    header = "         " + "  ".join(f"{name:>5}" for name in class_names)
    print(header)
    for i, row in enumerate(test_cm):
        print(f"  {class_names[i]:>5}  " + "  ".join(f"{v:5d}" for v in row))

    # Classification report
    all_preds, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for features, labels, _ in test_loader:
            features = features.to(device)
            with torch.autocast(
                device.type, torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(features)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        )
    )

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f2": test_f2,
        "best_epoch": best_epoch,
        "exp_dir": str(exp_dir),
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Transformer MIL head on TITAN/CONCHv1.5 features."
    )
    parser.add_argument("--features_dir", type=str, default="data/features_titan")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
