"""
Training pipeline for MPN Classification and Fibrosis Grading.
Implements WeightedRandomSampler for handling class imbalance.
Uses F2-Score (Macro) for model selection to minimize False Negatives.
"""
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import fbeta_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    DATA_MODE_CONFIG,
    DEFAULT_DATA_MODE,
    EPOCHS,
    EXPERIMENTS_DIR,
    LR,
    NUM_WORKERS,
    SEED,
    SUPPORTED_MODELS,
)
from dataset import MPNDataset
from model import get_model, print_model_summary
from utils import (
    get_class_weights,
    get_loss_weights,
    get_num_classes,
    get_patient_split,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MPN Classification or Fibrosis Grading model"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "grading"],
        help="Task to train: 'classification' (H&E) or 'grading' (Reticulin)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=list(SUPPORTED_MODELS),
        help="Model architecture to use",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate (default: {LR})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of data loading workers (default: {NUM_WORKERS})",
    )

    parser.add_argument(
        "--data_mode",
        type=str,
        default=DEFAULT_DATA_MODE,
        choices=list(DATA_MODE_CONFIG.keys()),
        help="Data mode: 'resize' (raw .tif, resize to 224) or 'patch' (preprocessed .png patches)",
    )

    return parser.parse_args()


def create_dataloaders(
    task: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    data_dir: Path,
    file_ext: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train/val/test dataloaders with WeightedRandomSampler for training.

    Args:
        task: Either 'classification' or 'grading'
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed
        data_dir: Root data directory containing class folders
        file_ext: File extension to filter (e.g., 'tif', 'png')

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
    """
    # Get patient-level split
    train_files, val_files, test_files = get_patient_split(
        task, data_dir=data_dir, file_ext=file_ext, seed=seed
    )

    # Get number of classes
    num_classes = get_num_classes(task)

    # Create datasets
    train_dataset = MPNDataset(train_files, task=task, is_training=True)
    val_dataset = MPNDataset(val_files, task=task, is_training=False)
    test_dataset = MPNDataset(test_files, task=task, is_training=False)

    # Calculate sample weights for handling class imbalance
    sample_weights = get_class_weights(train_files, num_classes)

    # Create WeightedRandomSampler for training
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    """
    Validate the model.

    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on
        num_classes: Number of classes for F2-Score calculation

    Returns:
        Tuple of (average_loss, accuracy, f2_score_macro)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    pbar = tqdm(val_loader, desc="Validation", leave=False)

    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Collect predictions for F2-Score
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    # Calculate Macro F2-Score (beta=2 emphasizes recall over precision)
    labels_list = list(range(num_classes))
    epoch_f2 = fbeta_score(
        all_labels, all_preds, beta=2, average='macro',
        labels=labels_list, zero_division=0
    )

    return epoch_loss, epoch_acc, epoch_f2


def train(
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    Main training function.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with best metrics
    """
    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Resolve data mode to data_dir and file_ext
    mode_config = DATA_MODE_CONFIG[args.data_mode]
    data_dir = mode_config["data_dir"]
    file_ext = mode_config["extension"]

    print(f"Data mode: {args.data_mode} ({mode_config['description']})")

    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=data_dir,
        file_ext=file_ext,
    )

    # Get train_files for loss weight calculation (Double Force strategy)
    train_files, _, _ = get_patient_split(
        args.task, data_dir=data_dir, file_ext=file_ext, seed=args.seed
    )

    # Calculate per-class loss weights (inverse frequency, normalized)
    loss_weights = get_loss_weights(train_files, num_classes).to(device)

    # Print class weights for transparency
    if args.task == "classification":
        class_names = ["ET", "PV", "PMF"]
    else:
        class_names = ["G0", "G1", "G2", "G3"]

    print(f"\n{'='*60}")
    print(f"Class Weights for Loss Function (Double Force Strategy)")
    print(f"{'='*60}")
    for i, (name, weight) in enumerate(zip(class_names, loss_weights.tolist())):
        print(f"  {name}: {weight:.4f}")
    print(f"{'='*60}")

    # Create model
    model = get_model(args.model, num_classes, device)
    print_model_summary(model, args.model)

    # Loss function with class weights and optimizer
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Create experiment directory with data mode info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.task}_{args.model}_{args.data_mode}_{timestamp}"
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")

    # Training loop
    best_val_f2 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f2": [],
    }

    print(f"\n{'='*60}")
    print(f"Starting training: {args.task} with {args.model}")
    print(f"Model selection metric: F2-Score (Macro, beta=2)")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch + 1}/{args.epochs}]")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate (now returns F2-Score as well)
        val_loss, val_acc, val_f2 = validate(
            model, val_loader, criterion, device, num_classes
        )

        # Step scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f2"].append(val_f2)

        # Print epoch summary
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val F2: {val_f2:.4f}"
        )

        # Save best model based on F2-Score (not accuracy)
        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f2": val_f2,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "args": vars(args),
            }

            torch.save(checkpoint, exp_dir / "best_model.pth")
            print(f"  [*] Saved best model (Val F2: {val_f2:.4f}, Acc: {val_acc:.2f}%)")

        print()

    # Save final model
    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "args": vars(args),
    }
    torch.save(final_checkpoint, exp_dir / "final_model.pth")

    # Save training history
    torch.save(history, exp_dir / "history.pth")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation F2-Score: {best_val_f2:.4f} (Epoch {best_epoch})")
    print(f"Best validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {exp_dir}")
    print(f"{'='*60}\n")

    return {
        "best_val_f2": best_val_f2,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "exp_dir": str(exp_dir),
    }


def main() -> None:
    """Main entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

