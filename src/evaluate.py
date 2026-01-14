"""
Evaluation pipeline for MPN Classification and Fibrosis Grading.
Implements image-level and patient-level aggregation strategies for patch-based predictions.

Usage:
    python src/evaluate.py --checkpoint experiments/xxx/best_model.pth --data_mode resize
    python src/evaluate.py --checkpoint experiments/xxx/best_model.pth --data_mode patch --aggregation mean --level image
    python src/evaluate.py --checkpoint experiments/xxx/best_model.pth --data_mode patch --aggregation vote --level patient
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    BATCH_SIZE,
    CLASS_MAP_INV,
    DATA_MODE_CONFIG,
    DEFAULT_DATA_MODE,
    GRADE_MAP_INV,
    NUM_WORKERS,
    RESULTS_DIR,
    SEED,
)
from dataset import MPNDataset
from model import get_model
from utils import get_num_classes, get_patient_split, set_seed


# Ensure reports directory exists
REPORTS_DIR = RESULTS_DIR / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MPN Classification or Fibrosis Grading model"
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
        "--aggregation",
        type=str,
        default="mean",
        choices=["vote", "mean", "max", "clinical"],
        help="Aggregation strategy for patch mode: 'vote', 'mean', 'max', or 'clinical' (default: mean)",
    )

    parser.add_argument(
        "--level",
        type=str,
        default="image",
        choices=["image", "patient"],
        help="Aggregation level for patch mode: 'image' (per original image) or 'patient' (per patient folder)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for evaluation (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
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


def get_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[np.ndarray], List[str]]:
    """
    Get predictions for all samples.

    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device

    Returns:
        Tuple of (predictions, labels, probabilities, file_paths)
    """
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    model.eval()
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    return all_preds, all_labels, all_probs, all_paths


def _extract_original_image_name(patch_path: str) -> str:
    """
    Extract original image name from patch filename.

    Patch naming convention: {original_name}_r{row}c{col}.png
    Example: "1_r0c0.png" -> "1"
             "reti1_r2c1.png" -> "reti1"

    Args:
        patch_path: Path to patch file

    Returns:
        Original image name (without _rXcX suffix)
    """
    filename = Path(patch_path).stem  # Remove extension
    # Remove _rXcX suffix (e.g., _r0c0, _r1c2)
    original_name = re.sub(r'_r\d+c\d+$', '', filename)
    return original_name


def _get_group_key(file_path: str, level: str) -> str:
    """
    Get grouping key based on aggregation level.

    Args:
        file_path: Path to file
        level: 'image' or 'patient'

    Returns:
        Group key string
    """
    path = Path(file_path)

    if level == "patient":
        # Group by patient folder name
        return path.parent.name
    else:  # level == "image"
        # Group by original image name (strip _rXcX suffix)
        # Include patient folder to ensure uniqueness across patients
        patient_id = path.parent.name
        original_name = _extract_original_image_name(file_path)
        return f"{patient_id}/{original_name}"


def aggregate_predictions(
    predictions: List[int],
    probabilities: List[np.ndarray],
    file_paths: List[str],
    labels: List[int],
    aggregation: str,
    level: str = "image",
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, np.ndarray]]:
    """
    Aggregate patch-level predictions to image-level or patient-level.

    Args:
        predictions: List of patch predictions
        probabilities: List of patch probabilities
        file_paths: List of file paths
        labels: List of ground truth labels
        aggregation: Aggregation strategy ('vote', 'mean', 'max')
        level: Aggregation level ('image' or 'patient')

    Returns:
        Tuple of (aggregated_preds, aggregated_labels, aggregated_probs)
    """
    # Group by specified level
    grouped_data = defaultdict(lambda: {"preds": [], "probs": [], "labels": []})

    for pred, prob, path, label in zip(predictions, probabilities, file_paths, labels):
        group_key = _get_group_key(path, level)
        grouped_data[group_key]["preds"].append(pred)
        grouped_data[group_key]["probs"].append(prob)
        grouped_data[group_key]["labels"].append(label)

    aggregated_preds = {}
    aggregated_labels = {}
    aggregated_probs = {}

    for group_key, data in grouped_data.items():
        preds = np.array(data["preds"])
        probs = np.array(data["probs"])

        # Ground truth (should be same for all patches of the same group)
        aggregated_labels[group_key] = data["labels"][0]

        if aggregation == "vote":
            # Majority voting
            unique, counts = np.unique(preds, return_counts=True)
            aggregated_preds[group_key] = unique[np.argmax(counts)]
            aggregated_probs[group_key] = np.mean(probs, axis=0)

        elif aggregation == "mean":
            # Mean probability
            mean_probs = np.mean(probs, axis=0)
            aggregated_preds[group_key] = np.argmax(mean_probs)
            aggregated_probs[group_key] = mean_probs

        elif aggregation == "max":
            # Max confidence
            max_conf_idx = np.argmax(np.max(probs, axis=1))
            aggregated_preds[group_key] = preds[max_conf_idx]
            aggregated_probs[group_key] = probs[max_conf_idx]

        elif aggregation == "clinical":
            # Clinical grading rules for fibrosis
            # Priority check: G3 -> G2 -> G1 -> G0 with BALANCED thresholds
            counts = np.bincount(preds, minlength=4)  # Ensure we cover G0-G3
            total = len(preds)
            ratios = counts / total

            # Uniform Threshold (30% as requested)
            threshold = 0.30

            # 1. G3: Is G3 >= 30%?
            if ratios[3] >= threshold:
                aggregated_preds[group_key] = 3

            # 2. G2: Is G2 >= 30%? (Checked only if G3 < 30%)
            elif ratios[2] >= threshold:
                aggregated_preds[group_key] = 2

            # 3. G1: Is G1 >= 30%? (Checked only if G3 and G2 < 30%)
            elif ratios[1] >= threshold:
                aggregated_preds[group_key] = 1

            # 4. Fallback: G0 (Normal)
            else:
                aggregated_preds[group_key] = 0

            # Use mean probability for reporting
            aggregated_probs[group_key] = np.mean(probs, axis=0)

    return aggregated_preds, aggregated_labels, aggregated_probs


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    # Define all possible labels based on class_names
    labels = list(range(len(class_names)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "f2_macro": fbeta_score(y_true, y_pred, beta=2, average="macro", labels=labels, zero_division=0),
        "f2_weighted": fbeta_score(y_true, y_pred, beta=2, average="weighted", labels=labels, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, labels=labels, output_dict=True, zero_division=0
        ),
    }

    return metrics


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to: {save_path}")


def plot_per_class_metrics(
    report: Dict,
    save_path: Path,
    title: str = "Per-Class Performance Metrics",
) -> None:
    """
    Plot grouped bar chart showing Precision, Recall, F1-Score, and F2-Score for each class.

    Args:
        report: Dictionary from sklearn's classification_report (output_dict=True)
        save_path: Path to save figure
        title: Plot title
    """
    import pandas as pd
    from sklearn.metrics import fbeta_score

    sns.set_theme(style="whitegrid")

    # Filter out summary keys, keep only class metrics
    summary_keys = {'accuracy', 'macro avg', 'weighted avg'}
    class_data = []

    for class_name, metrics in report.items():
        if class_name not in summary_keys and isinstance(metrics, dict):
            # Calculate F2-Score per class from precision and recall
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            # F2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
            beta = 2
            if precision + recall > 0:
                f2_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            else:
                f2_score = 0.0

            class_data.append({
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': metrics.get('f1-score', 0),
                'F2-Score': f2_score,
            })

    if not class_data:
        print(f"Warning: No class data found in report, skipping per-class plot.")
        return

    # Convert to long-form DataFrame for seaborn
    df = pd.DataFrame(class_data)
    df_long = pd.melt(
        df,
        id_vars=['Class'],
        value_vars=['Precision', 'Recall', 'F1-Score', 'F2-Score'],
        var_name='Metric',
        value_name='Score'
    )

    # Create the grouped bar chart
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_long,
        x='Class',
        y='Score',
        hue='Metric',
        palette='Set2'
    )

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)

    plt.ylim(0, 1.15)  # Leave room for labels
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title='Metric', loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Per-class metrics plot saved to: {save_path}")


def plot_overall_metrics(
    metrics: Dict,
    save_path: Path,
    title: str = "Overall Performance Metrics",
) -> None:
    """
    Plot bar chart showing overall performance metrics (Accuracy, Precision, Recall, F1, F2).

    Args:
        metrics: Dictionary containing 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f2_macro'
        save_path: Path to save figure
        title: Plot title
    """
    import pandas as pd

    sns.set_theme(style="whitegrid")

    # Extract overall metrics (including F2-Score)
    overall_data = {
        'Accuracy': metrics.get('accuracy', 0),
        'Precision (Macro)': metrics.get('precision_macro', 0),
        'Recall (Macro)': metrics.get('recall_macro', 0),
        'F1-Score (Macro)': metrics.get('f1_macro', 0),
        'F2-Score (Macro)': metrics.get('f2_macro', 0),
    }

    # Create DataFrame
    df = pd.DataFrame({
        'Metric': list(overall_data.keys()),
        'Score': list(overall_data.values())
    })

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x='Metric',
        y='Score',
        palette='viridis',
        hue='Metric',
        legend=False
    )

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)

    plt.ylim(0, 1.15)  # Leave room for labels
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Overall metrics plot saved to: {save_path}")


def evaluate(args: argparse.Namespace) -> Dict:
    """
    Main evaluation function.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with evaluation results
    """
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
    # Handle both old (val_acc only) and new (val_f2) checkpoint formats
    if "val_f2" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}, Val F2: {checkpoint['val_f2']:.4f}, Val Acc: {checkpoint['val_acc']:.2f}%")
    else:
        print(f"Checkpoint epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%")

    # Resolve data mode
    mode_config = DATA_MODE_CONFIG[args.data_mode]
    data_dir = mode_config["data_dir"]
    file_ext = mode_config["extension"]

    print(f"Data mode: {args.data_mode} ({mode_config['description']})")

    # Get test data
    _, _, test_files = get_patient_split(
        task, data_dir=data_dir, file_ext=file_ext, seed=args.seed
    )

    test_dataset = MPNDataset(test_files, task=task, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Get predictions
    predictions, labels, probabilities, file_paths = get_predictions(
        model, test_loader, device
    )

    # Get class names
    if task == "classification":
        class_names = list(CLASS_MAP_INV.values())
    else:
        class_names = list(GRADE_MAP_INV.values())

    # Compute metrics based on data mode
    if args.data_mode == "patch":
        print(f"\nAggregation strategy: {args.aggregation}")
        print(f"Aggregation level: {args.level}")

        # Aggregate predictions based on level (image or patient)
        aggregated_preds, aggregated_labels, _ = aggregate_predictions(
            predictions, probabilities, file_paths, labels,
            args.aggregation, level=args.level
        )

        y_true = list(aggregated_labels.values())
        y_pred = list(aggregated_preds.values())

        level_name = "image" if args.level == "image" else "patient"
        print(f"Unique {level_name}s in test set: {len(y_true)}")
        level = args.level
    else:
        # Image-level evaluation (resize mode)
        y_true = labels
        y_pred = predictions
        level = "image"

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, class_names)

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({level}-level)")
    print(f"{'='*60}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro F2-Score:  {metrics['f2_macro']:.4f}  <-- Primary Metric")
    print(f"Macro F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:    {metrics['recall_macro']:.4f}")
    print(f"{'='*60}\n")

    # Extract experiment name from checkpoint path
    experiment_name = checkpoint_path.parent.name

    # Create experiment-specific output directories
    exp_figures_dir = FIGURES_DIR / experiment_name
    exp_reports_dir = REPORTS_DIR / experiment_name
    exp_figures_dir.mkdir(parents=True, exist_ok=True)
    exp_reports_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"eval_{task}_{model_name}_{args.data_mode}_{args.level}_{timestamp}"

    # Save metrics as JSON
    report_path = exp_reports_dir / f"{report_name}.json"
    results = {
        "checkpoint": str(checkpoint_path),
        "task": task,
        "model": model_name,
        "data_mode": args.data_mode,
        "aggregation": args.aggregation if args.data_mode == "patch" else None,
        "level": level,
        "num_samples": len(y_true),
        "metrics": metrics,
    }

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Report saved to: {report_path}")

    # Plot confusion matrix
    cm_title = f"Confusion Matrix ({task}, {args.data_mode})"
    if args.data_mode == "patch":
        cm_title += f" - {args.aggregation} aggregation"

    cm_path = exp_figures_dir / f"{report_name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, cm_title)

    # Plot per-class detailed metrics (Precision, Recall, F1-Score)
    per_class_title = f"Per-Class Performance ({task}, {args.data_mode})"
    if args.data_mode == "patch":
        per_class_title += f" - {args.aggregation} aggregation"
    per_class_path = exp_figures_dir / f"{report_name}_per_class_detailed.png"
    plot_per_class_metrics(metrics["classification_report"], per_class_path, per_class_title)

    # Plot overall metrics
    overall_title = f"Overall Performance ({task}, {args.data_mode})"
    if args.data_mode == "patch":
        overall_title += f" - {args.aggregation} aggregation"
    overall_path = exp_figures_dir / f"{report_name}_overall_metrics.png"
    plot_overall_metrics(metrics, overall_path, overall_title)

    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

