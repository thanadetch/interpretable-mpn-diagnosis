"""
Run Janitor Model: Clean the dataset by quarantining bone artifact patches.

This script uses the trained Janitor model to filter out cortical bone patches
from the main grading dataset, keeping only valid bone marrow ROI patches.

Usage:
    python src/run_janitor.py --threshold 0.90
    python src/run_janitor.py --input_dir data/processed_grading --threshold 0.85

Output:
    Bone patches are MOVED to data/quarantine_janitor/ preserving folder structure.
"""
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from config import EXPERIMENTS_DIR, PROCESSED_GRADING_DIR, PROJECT_ROOT

# ============================================================================
# Configuration
# ============================================================================
JANITOR_MODEL_PATH: Path = EXPERIMENTS_DIR / "janitor_model.pth"
QUARANTINE_DIR: Path = PROJECT_ROOT / "data" / "quarantine_janitor"
JANITOR_CLASSES = ["bone", "marrow"]  # Index 0 = bone, Index 1 = marrow


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_janitor_model(model_path: Path, device: torch.device) -> nn.Module:
    """
    Load the trained Janitor model.

    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Janitor model not found at: {model_path}\n"
            f"Please train the Janitor model first using:\n"
            f"  python src/train_janitor.py --epochs 10"
        )

    # Create model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded Janitor model from: {model_path}")
    print(f"  - Trained for {checkpoint['epoch']} epochs")
    print(f"  - Validation Accuracy: {checkpoint['val_acc']:.2f}%")

    return model


def get_inference_transform() -> transforms.Compose:
    """Get the transform for inference (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def predict_single_image(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> tuple:
    """
    Predict if an image is bone or marrow.

    Args:
        model: Janitor model
        image_path: Path to the image
        transform: Image transform
        device: Device for inference

    Returns:
        Tuple of (predicted_class, bone_probability)
        predicted_class: 0 = bone, 1 = marrow
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        bone_prob = probabilities[0, 0].item()  # Index 0 = bone
        predicted_class = outputs.argmax(dim=1).item()

    return predicted_class, bone_prob


def run_janitor(args: argparse.Namespace) -> None:
    """Main function to clean the dataset using the Janitor model."""
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = load_janitor_model(JANITOR_MODEL_PATH, device)
    transform = get_inference_transform()

    # Input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all PNG files
    all_patches = list(input_dir.rglob("*.png"))

    print(f"\n{'='*60}")
    print("Janitor Model - Dataset Cleaning")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Total patches found: {len(all_patches)}")
    print(f"Bone probability threshold: {args.threshold}")
    print(f"Quarantine directory: {QUARANTINE_DIR}")
    print(f"{'='*60}\n")

    if len(all_patches) == 0:
        print("No patches found! Exiting.")
        return

    # Create quarantine directory
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    # Process patches
    bone_count = 0
    marrow_count = 0
    moved_files = []

    pbar = tqdm(all_patches, desc="Scanning patches", unit="patch")
    for patch_path in pbar:
        predicted_class, bone_prob = predict_single_image(
            model, patch_path, transform, device
        )

        # If bone with high confidence, quarantine it
        if predicted_class == 0 and bone_prob >= args.threshold:
            bone_count += 1

            # Preserve relative folder structure
            relative_path = patch_path.relative_to(input_dir)
            quarantine_path = QUARANTINE_DIR / relative_path
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            if not args.dry_run:
                shutil.move(str(patch_path), str(quarantine_path))
                moved_files.append((patch_path, quarantine_path, bone_prob))
            else:
                moved_files.append((patch_path, quarantine_path, bone_prob))
        else:
            marrow_count += 1

        pbar.set_postfix({
            "bone": bone_count,
            "marrow": marrow_count,
            "last_prob": f"{bone_prob:.2f}"
        })

    # Summary
    print(f"\n{'='*60}")
    print("Janitor Cleaning Complete!")
    print(f"{'='*60}")
    print(f"Total patches scanned: {len(all_patches)}")
    print(f"Bone artifacts detected: {bone_count} ({100*bone_count/len(all_patches):.1f}%)")
    print(f"Valid marrow patches: {marrow_count} ({100*marrow_count/len(all_patches):.1f}%)")

    if args.dry_run:
        print(f"\n[DRY RUN] No files were actually moved.")
        print(f"[DRY RUN] {bone_count} files would be quarantined.")
    else:
        print(f"\nMoved {bone_count} bone patches to: {QUARANTINE_DIR}")

    # Show some examples
    if moved_files and args.verbose:
        print(f"\nExample quarantined files (top 10):")
        for src, dst, prob in moved_files[:10]:
            print(f"  [{prob:.2%}] {src.name}")

    print(f"{'='*60}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Janitor Model to clean dataset by quarantining bone patches"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROCESSED_GRADING_DIR),
        help=f"Input directory with patches (default: {PROCESSED_GRADING_DIR})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Bone probability threshold for quarantine (default: 0.90)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan only, don't actually move files"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show example quarantined files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_janitor(args)

