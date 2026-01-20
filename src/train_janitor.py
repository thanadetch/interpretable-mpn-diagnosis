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
        ├── bone/      # Cortical bone artifact patches
        └── marrow/    # Valid bone marrow ROI patches

    For subtype (H&E):
        data/janitor_train_subtype/
        ├── artifact/  # Artifact patches (e.g., folds, tears, blur)
        └── tissue/    # Valid tissue patches
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from config import EXPERIMENTS_DIR, PROJECT_ROOT, SEED

# ============================================================================
# Task-Specific Configuration
# ============================================================================
TASK_CONFIG = {
    "grading": {
        "data_dir": PROJECT_ROOT / "data" / "janitor_train_grading",
        "model_path": EXPERIMENTS_DIR / "janitor_model_grading.pth",
        "classes": ["bone", "marrow"],
        "description": "Bone vs Marrow Classifier (Reticulin)",
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
        Dictionary with 'train' and 'val' transforms
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return {"train": train_transform, "val": val_transform}


def create_janitor_model(num_classes: int = 2, device: torch.device = None) -> nn.Module:
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

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
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
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_janitor(args: argparse.Namespace) -> None:
    """Main training function for Janitor model."""
    set_seed(SEED)

    # Get task-specific configuration
    task_config = TASK_CONFIG[args.task]
    data_dir = task_config["data_dir"]
    output_path = task_config["model_path"]
    janitor_classes = task_config["classes"]
    task_description = task_config["description"]

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

    # Get transforms
    data_transforms = get_transforms()

    # Load dataset using ImageFolder
    # Split into train/val (80/20)
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=data_transforms["train"]
    )

    print(f"\n{'='*60}")
    print(f"Janitor Model Training ({task_description})")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Dataset: {data_dir}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Total samples: {len(full_dataset)}")

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Update val_dataset transform (no augmentation)
    val_dataset.dataset.transform = data_transforms["val"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"{'='*60}\n")

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

    # Create model
    model = create_janitor_model(num_classes=2, device=device)
    print(f"Model: ResNet18 (Pretrained ImageNet)")
    print(f"Output classes: 2 (bone, marrow)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_val_acc = 0.0

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
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "task": args.task,
                "classes": janitor_classes,
            }, output_path)
            print(f"  -> Best model saved! (Val Acc: {val_acc:.2f}%)")

        print()

    print(f"{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")


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
        help="Task to train for: 'grading' (Bone vs Marrow) or 'subtype' (Artifact vs Tissue)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_janitor(args)

