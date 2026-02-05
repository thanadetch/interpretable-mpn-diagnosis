#!/usr/bin/env python3
"""
Sanity check script to visualize the effect of Stain Normalization.
Uses Numpy backend for compatibility with all CPU architectures (including Mac M1/M2).
Run from project root: python tools/check_stain.py --num_samples 5
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src/ to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")  # Headless mode
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchstain.base.normalizers import MacenkoNormalizer

from config import DATA_MODE_CONFIG, RESULTS_DIR
from dataset import MPNDataset
from utils import get_patient_split


class NumpyStainNorm:
    """
    Stain normalization helper using Numpy backend for CPU compatibility.
    Used for visualization purposes only.
    """

    def __init__(self, target_path: str = "data/templates/template_he.png"):
        self.target_path = target_path
        self.normalizer = MacenkoNormalizer(backend="numpy")
        self._fitted = False

        self._fit_to_target()

    def _fit_to_target(self) -> None:
        """Fit normalizer to reference H&E image."""
        if not os.path.exists(self.target_path):
            print(f"[NumpyStainNorm] WARNING: Target not found: {self.target_path}")
            return

        # Load reference image (BGR -> RGB)
        target_bgr = cv2.imread(self.target_path)
        if target_bgr is None:
            print(f"[NumpyStainNorm] WARNING: Failed to load: {self.target_path}")
            return

        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

        try:
            self.normalizer.fit(target_rgb)
            self._fitted = True
            print(f"[NumpyStainNorm] Fitted to target: {self.target_path}")
        except Exception as e:
            print(f"[NumpyStainNorm] Fit error: {e}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor using Numpy backend.

        Args:
            x: Input tensor [1, C, H, W] in range [0, 1]

        Returns:
            Normalized tensor [1, C, H, W] in range [0, 1]
        """
        if not self._fitted:
            return x

        # Convert tensor to numpy: [1, C, H, W] -> [H, W, C]
        img_np = x[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C] float [0, 1]
        img_np = (img_np * 255).astype(np.uint8)  # [0, 255] uint8

        try:
            # Normalize using numpy backend
            norm_np, _, _ = self.normalizer.normalize(I=img_np, stains=True)
        except Exception as e:
            print(f"[NumpyStainNorm] Normalize error: {e}")
            norm_np = img_np

        # Convert back to tensor: [H, W, C] -> [1, C, H, W]
        norm_tensor = torch.from_numpy(norm_np).float() / 255.0
        norm_tensor = norm_tensor.permute(2, 0, 1).unsqueeze(0)

        return norm_tensor


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
    """Run stain normalization sanity check."""
    args = parse_args()

    # Setup device (display only, we use numpy backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Display device: {device}")
    print("Using Numpy backend for stain normalization (CPU compatible)\n")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"stain_check_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize NumpyStainNorm (uses numpy backend for compatibility)
    stain_layer = NumpyStainNorm()

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
    for i in range(num_samples):
        image, label, file_path = dataset[i]
        sample_name = Path(file_path).stem

        print(f"[{i + 1}/{num_samples}] {sample_name}")
        print(f"  Label: {label}")
        print(f"  Input range: [{image.min():.3f}, {image.max():.3f}]")

        # Add batch dimension: [C, H, W] -> [1, C, H, W]
        image_batch = image.unsqueeze(0)

        # Apply stain normalization (numpy backend, CPU)
        normalized_batch = stain_layer(image_batch)

        # Get results for plotting
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
