"""
Train Janitor Model (Stage 1): Binary classifier for cleaning patches.

This is part of the Two-Stage Cascaded Framework to fix shortcut learning.
The Janitor model filters out artifacts before downstream tasks.

Supported Tasks:
    - grading: Bone vs Marrow classification (Reticulin stain)
    - subtype: Artifact vs Tissue classification (H&E stain)

Usage:
    python src/train_janitor.py --task grading --epochs 10 --batch_size 32
    python src/train_janitor.py --task subtype --epochs 10 --batch_size 32

Dataset Structure Required:
    For grading (Reticulin):
        data/janitor_train_grading/
        ├── artifact/  # Cortical bone artifact patches
        └── marrow/    # Valid bone marrow ROI patches

    For subtype (H&E):
        data/janitor_train_subtype/
        ├── artifact/  # Artifact patches (e.g., folds, tears, blur)
        └── tissue/    # Valid tissue patches
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Ensure src/ is on sys.path when running directly (e.g., python src/janitor/train_janitor.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import EXPERIMENTS_DIR, PROJECT_ROOT, SEED


# ============================================================================
# TeeLogger: Duplicate stdout to file and terminal
# ============================================================================
class TeeLogger(object):
    """Logger that writes to both terminal and file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ============================================================================
# Task-Specific Configuration
# ============================================================================
TASK_CONFIG = {
    "grading": {
        "data_dir": PROJECT_ROOT / "data" / "janitor_train_grading",
        "model_path": EXPERIMENTS_DIR / "janitor_model_grading.pth",
        "classes": ["artifact", "marrow"],
        "description": "Artifact (Bone/Bg) vs Marrow Classifier",
    },
    "subtype": {
        "data_dir": PROJECT_ROOT / "data" / "janitor_train_subtype",
        "model_path": EXPERIMENTS_DIR / "janitor_model_subtype.pth",
        "classes": ["artifact", "tissue"],
        "description": "Artifact vs Tissue Classifier (H&E)",
    },
}


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms() -> dict:
    """
    Get data augmentation transforms for Janitor training.

    Returns:
        Dictionary with 'train', 'val', and 'test' transforms.
        Only 'train' includes data augmentation.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Val and Test use same transform (no augmentation)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def create_janitor_model(
    num_classes: int = 2, device: torch.device = None
) -> nn.Module:
    """
    Create a lightweight ResNet18 model for binary classification.

    Args:
        num_classes: Number of output classes (default: 2 for bone/marrow)
        device: Device to move model to

    Returns:
        ResNet18 model with modified FC layer
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace the final FC layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if device:
        model = model.to(device)

    return model


def split_dataset(
    dataset: datasets.ImageFolder,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices into train/val/test sets with stratification.

    Args:
        dataset: ImageFolder dataset
        train_ratio: Proportion for training (default: 0.70)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for test (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    # Group indices by class for stratified split
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    rng = random.Random(seed)

    for label, indices in class_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    return train_indices, val_indices, test_indices


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, List[int], List[int]]:
    """
    Evaluate model on test set and return detailed metrics.

    Args:
        model: Trained model
        dataloader: Test DataLoader
        device: Device
        class_names: List of class names

    Returns:
        Tuple of (accuracy, all_labels, all_preds)
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(dataloader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(predicted.cpu().numpy().tolist())

    accuracy = 100.0 * correct / total
    return accuracy, all_labels, all_preds


class TransformSubset(torch.utils.data.Dataset):
    """
    A Dataset that wraps a subset of samples with a specific transform.

    This is more robust than Subset for multiprocessing as it stores
    the samples list directly rather than referencing a parent dataset.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        Args:
            samples: List of (image_path, label) tuples
            transform: Transform to apply to images
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image using PIL (imported at module level)
        img = Image.open(img_path).convert("RGB")

        # Apply transform (must convert to tensor)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def train_janitor(args: argparse.Namespace) -> None:
    """Main training function for Janitor model."""
    set_seed(SEED)

    # Get task-specific configuration
    task_config = TASK_CONFIG[args.task]
    data_dir = task_config["data_dir"]
    janitor_classes = task_config["classes"]
    task_description = task_config["description"]

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"janitor_{args.task}_{timestamp}"
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_path = exp_dir / "train_log.txt"
    sys.stdout = TeeLogger(log_path)
    print(f"📄 Logging training output to: {log_path}")

    # Log training configuration
    print("\n" + "=" * 40)
    print("🚀 Training Configuration:")
    print("=" * 40)
    for key, value in vars(args).items():
        print(f"{key:20}: {value}")
    print("=" * 40 + "\n")

    # Model output path (inside experiment folder)
    output_path = exp_dir / f"janitor_model_{args.task}.pth"

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Check data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Janitor training data not found at: {data_dir}\n"
            f"Please create the following structure:\n"
            f"  {data_dir}/{janitor_classes[0]}/   (class 0 patches)\n"
            f"  {data_dir}/{janitor_classes[1]}/ (class 1 patches)"
        )

    # Ge transforms
    data_transforms = get_transforms()

    # Load base dataset (no transform applied yet, we'll apply per-split)
    base_dataset = datasets.ImageFolder(root=data_dir)

    print(f"\n{'=' * 60}")
    print(f"Janitor Model Training ({task_description})")
    print(f"{'=' * 60}")
    print(f"Task: {args.task}")
    print(f"Dataset: {data_dir}")
    print(f"Classes: {base_dataset.classes}")
    print(f"Total samples: {len(base_dataset)}")

    # Split dataset into 70% Train, 15% Val, 15% Test
    train_indices, val_indices, test_indices = split_dataset(
        base_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=SEED
    )

    # Extract samples for each split
    train_samples = [base_dataset.samples[i] for i in train_indices]
    val_samples = [base_dataset.samples[i] for i in val_indices]
    test_samples = [base_dataset.samples[i] for i in test_indices]

    # Create datasets with appropriate transforms
    train_dataset = TransformSubset(train_samples, transform=data_transforms["train"])
    val_dataset = TransformSubset(val_samples, transform=data_transforms["val"])
    test_dataset = TransformSubset(test_samples, transform=data_transforms["test"])

    print(f"Train samples: {len(train_dataset)} (70%)")
    print(f"Val samples:   {len(val_dataset)} (15%)")
    print(f"Test samples:  {len(test_dataset)} (15%)")
    print(f"{'=' * 60}\n")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    model = create_janitor_model(num_classes=2, device=device)
    print("Model: ResNet18 (Pretrained ImageNet)")
    print(f"Output classes: 2 ({', '.join(janitor_classes)})")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model will be saved to: {output_path}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "task": args.task,
                    "classes": janitor_classes,
                },
                output_path,
            )
            print(f"  -> Best model saved! (Val Acc: {val_acc:.2f}%)")

        print()

    print(f"{'=' * 60}")
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'=' * 60}")

    # ==========================================
    # 🧪 Test Phase (using Best Model)
    # ==========================================
    print(f"\n{'=' * 60}")
    print("🚀 Starting Test Evaluation")
    print(f"{'=' * 60}")

    # Load the best model weights
    print(f"[*] Loading best model from: {output_path}")
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    test_acc, all_labels, all_preds = evaluate_test(
        model, test_loader, device, janitor_classes
    )

    print(f"\n🏆 Final Test Results:")
    print(f"   Test Accuracy: {test_acc:.2f}%")

    # Classification Report
    print(f"\n📊 Classification Report:")
    print("-" * 60)
    report = classification_report(
        all_labels, all_preds, target_names=janitor_classes, digits=4
    )
    print(report)

    # Confusion Matrix
    print("📊 Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate dynamic column width (min 12 chars)
    col_width = max(12, max(len(c) for c in janitor_classes) + 2)
    row_label_width = len("Actual") + 1 + col_width  # "Actual " + class name

    # Header row
    header_indent = " " * row_label_width
    header_row = "".join(f"{c:>{col_width}}" for c in janitor_classes)
    print(f"{header_indent}Predicted")
    print(f"{header_indent}{header_row}")

    # Data rows
    for i, class_name in enumerate(janitor_classes):
        row_label = f"{'Actual' if i == 0 else '':6} {class_name:>{col_width}}"
        row_data = "".join(
            f"{cm[i][j]:>{col_width}}" for j in range(len(janitor_classes))
        )
        print(f"{row_label}{row_data}")

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"Experiment directory: {exp_dir}")
    print(f"Model saved to: {output_path}")
    print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Janitor Model for cleaning patches (grading or subtype task)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["grading", "subtype"],
        required=True,
        help="Task to train for: 'grading' (Bone vs Marrow) or 'subtype' (Artifact vs Tissue)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_janitor(args)
