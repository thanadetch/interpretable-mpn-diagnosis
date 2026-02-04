#!/usr/bin/env python3
"""
Sanity check script to visualize the effect of Stain Normalization.
Uses the actual production StainNormLayer from src/stain_norm.py.
Run from project root: python tools/check_stain.py --num_samples 5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src/ to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # Headless mode
import matplotlib.pyplot as plt
import torch

from config import DATA_MODE_CONFIG, RESULTS_DIR
from dataset import MPNDataset
from stain_norm import StainNormLayer
from utils import get_patient_split


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sanity check for Stain Normalization layer"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of images to check (default: 5)",
    )
    return parser.parse_args()


def main():
    """Run stain normalization sanity check using production StainNormLayer."""
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print("Testing _StainNormTorch worker (GPU path)")
    else:
        print("Testing _StainNormNumpy worker (CPU path)")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"stain_check_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize production StainNormLayer
    stain_layer = StainNormLayer()
    stain_layer = stain_layer.to(device)
    stain_layer.eval()

    # Get dataset configuration
    mode_config = DATA_MODE_CONFIG["subtype_patch_clean"]
    data_dir = mode_config["data_dir"]
    file_ext = mode_config["extension"]

    print(f"Loading data from: {data_dir}")

    # Get training files
    train_files, _, _ = get_patient_split(
        task="classification",
        data_dir=data_dir,
        file_ext=file_ext,
        seed=42,
    )

    # Create dataset (training mode)
    dataset = MPNDataset(train_files, task="classification", is_training=True)
    print(f"Dataset size: {len(dataset)} images")

    # Limit samples to available dataset size
    num_samples = min(args.num_samples, len(dataset))
    print(f"Processing {num_samples} samples...\n")

    # Process each sample
    with torch.no_grad():
        for i in range(num_samples):
            image, label, file_path = dataset[i]
            sample_name = Path(file_path).stem

            print(f"[{i + 1}/{num_samples}] {sample_name}")
            print(f"  Label: {label}")
            print(f"  Input range: [{image.min():.3f}, {image.max():.3f}]")

            # Add batch dimension and move to device: [C, H, W] -> [1, C, H, W]
            image_batch = image.unsqueeze(0).to(device)

            # Apply stain normalization using production layer
            normalized_batch = stain_layer(image_batch)

            # Move results to CPU for plotting
            original = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            normalized = normalized_batch[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]

            print(f"  Output range: [{normalized.min():.3f}, {normalized.max():.3f}]")

            # Clip values to [0, 1] for display
            original = original.clip(0, 1)
            normalized = normalized.clip(0, 1)

            # Plot comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(original)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(normalized)
            axes[1].set_title("Stain Normalized")
            axes[1].axis("off")

            plt.suptitle(f"Sample {i + 1}: {sample_name}\nLabel: {label}", fontsize=12)
            plt.tight_layout()

            # Save figure
            fig_path = output_dir / f"sample_{i + 1}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()

    print(f"\n✅ Saved {num_samples} comparison images to: {output_dir}")


if __name__ == "__main__":
    main()
