"""
Explainability module for MPN Classification and Fibrosis Grading.
Implements Grad-CAM visualization for model interpretability.

Usage:
    python src/explain.py --checkpoint experiments/xxx/best_model.pth --data_mode resize --num_samples 5
    python src/explain.py --checkpoint experiments/xxx/best_model.pth --data_mode patch --image_path "data/processed/ET/ET1 G1/1_r0c0.png"
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import (
    DATA_MODE_CONFIG,
    DEFAULT_DATA_MODE,
    IMAGE_SIZE,
    RESULTS_DIR,
    SEED,
    CLASS_MAP_INV,
    GRADE_MAP_INV,
)
from dataset import MPNDataset
from model import get_model, get_target_layer
from utils import get_num_classes, get_patient_split, set_seed

# Try to import pytorch_grad_cam, provide helpful message if not available
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed.")
    print("Install with: pip install grad-cam")
    print("Or: pip install git+https://github.com/jacobgil/pytorch-grad-cam.git")

# Ensure figures directory exists
FIGURES_DIR = RESULTS_DIR / "figures"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for MPN models"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )

    parser.add_argument(
        "--data_mode",
        type=str,
        default=DEFAULT_DATA_MODE,
        choices=list(DATA_MODE_CONFIG.keys()),
        help="Data mode: 'resize' (raw .tif) or 'patch' (preprocessed .png patches)",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to specific image to visualize (optional)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to visualize (default: 5)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory for figures (default: {FIGURES_DIR}/<experiment_name>)",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]

    # Get number of classes based on task
    num_classes = get_num_classes(args["task"])

    # Create model
    model = get_model(args["model"], num_classes, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for Grad-CAM visualization.

    Returns:
        Tuple of (preprocess_transform, display_transform)
    """
    # Transform for model input
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Transform for display (just resize, no normalization)
    display = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    return preprocess, display


def generate_gradcam(
    model: torch.nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for an image.

    Args:
        model: PyTorch model
        model_name: Name of model architecture
        image_tensor: Preprocessed image tensor (1, C, H, W)
        target_class: Target class for Grad-CAM (None for predicted class)
        device: Device

    Returns:
        Grad-CAM heatmap as numpy array
    """
    if not GRADCAM_AVAILABLE:
        raise ImportError("pytorch-grad-cam is not installed")

    # Get target layer based on model architecture
    target_layer = get_target_layer(model, model_name)

    # Create Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate CAM
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    else:
        targets = None

    grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Get first image in batch

    return grayscale_cam


def visualize_gradcam(
    image_path: str,
    model: torch.nn.Module,
    model_name: str,
    task: str,
    device: torch.device,
    save_path: Path,
    true_label: Optional[int] = None,
) -> None:
    """
    Generate and save Grad-CAM visualization for a single image.

    Args:
        image_path: Path to image
        model: PyTorch model
        model_name: Model architecture name
        task: Task ('classification' or 'grading')
        device: Device
        save_path: Path to save figure
        true_label: Ground truth label (optional)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Get transforms
    preprocess, display = get_transforms()

    # Preprocess for model
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    # Generate Grad-CAM
    grayscale_cam = generate_gradcam(
        model, model_name, input_tensor, target_class=pred_class, device=device
    )

    # Get display image (normalized to 0-1)
    display_tensor = display(image)
    rgb_img = display_tensor.permute(1, 2, 0).numpy()

    # Overlay CAM on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Get class names
    if task == "classification":
        class_names = CLASS_MAP_INV
    else:
        class_names = GRADE_MAP_INV

    pred_name = class_names[pred_class]
    true_name = class_names[true_label] if true_label is not None else "Unknown"

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Grad-CAM heatmap
    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(visualization)
    axes[2].set_title(f"Pred: {pred_name} ({confidence:.2%})\nTrue: {true_name}")
    axes[2].axis("off")

    # Add image path as subtitle
    image_name = Path(image_path).name
    fig.suptitle(f"Grad-CAM Visualization: {image_name}", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def explain(args: argparse.Namespace) -> None:
    """
    Main explanation function.

    Args:
        args: Command line arguments
    """
    if not GRADCAM_AVAILABLE:
        print("\nError: pytorch-grad-cam is required for this module.")
        print("Install with: pip install git+https://github.com/jacobgil/pytorch-grad-cam.git")
        return

    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    task = checkpoint["args"]["task"]
    model_name = checkpoint["args"]["model"]

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Task: {task}, Model: {model_name}")

    # Resolve data mode
    mode_config = DATA_MODE_CONFIG[args.data_mode]
    data_dir = mode_config["data_dir"]
    file_ext = mode_config["extension"]

    print(f"Data mode: {args.data_mode} ({mode_config['description']})")

    # Determine output directory
    if args.output_dir is not None:
        # User provided custom output directory
        output_dir = Path(args.output_dir)
    else:
        # Use experiment-specific directory based on checkpoint path
        experiment_name = checkpoint_path.parent.name
        output_dir = FIGURES_DIR / experiment_name

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get images to visualize
    if args.image_path:
        # Single image specified
        image_paths = [args.image_path]
        labels = [None]
    else:
        # Get test samples
        _, _, test_files = get_patient_split(
            task, data_dir=data_dir, file_ext=file_ext, seed=args.seed
        )

        # Create dataset to apply task-specific filtering
        test_dataset = MPNDataset(test_files, task=task, is_training=False)

        # Random sample
        num_samples = min(args.num_samples, len(test_dataset))
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)

        image_paths = []
        labels = []
        for idx in indices:
            _, label, path = test_dataset[idx]
            image_paths.append(path)
            labels.append(label)

    print(f"\nGenerating Grad-CAM for {len(image_paths)} images...")

    # Generate visualizations
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        # Include patient folder name to avoid duplicate filenames
        path_obj = Path(image_path)
        patient_folder = path_obj.parent.name
        image_name = path_obj.stem
        # Sanitize patient folder name (replace spaces with underscores)
        patient_folder_safe = patient_folder.replace(" ", "_")
        save_name = f"gradcam_{task}_{model_name}_{args.data_mode}_{patient_folder_safe}_{image_name}.png"
        save_path = output_dir / save_name

        try:
            visualize_gradcam(
                image_path=image_path,
                model=model,
                model_name=model_name,
                task=task,
                device=device,
                save_path=save_path,
                true_label=label,
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"\nGrad-CAM visualizations saved to: {output_dir}")

    # Print summary based on data mode
    if args.data_mode == "patch":
        print("\nNote: In 'patch' mode, Grad-CAM shows attention on individual patches.")
        print("This helps identify which morphological features the model focuses on.")
    else:
        print("\nNote: In 'resize' mode, Grad-CAM shows attention on the whole image.")
        print("Fine details may be lost due to resizing from original resolution.")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    explain(args)


if __name__ == "__main__":
    main()

