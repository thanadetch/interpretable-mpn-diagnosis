"""
Evaluation pipeline for MIL Reticulin Fibrosis Grading (4-class: G0–G3).

Loads a trained MIL checkpoint, automatically infers model configuration
(backbone, model_type, topk, formulation) from the saved checkpoint,
and evaluates on the exact held-out test split stored in the checkpoint.

Usage:
    python -m src.eval_grading_reti \
        --checkpoint experiments/reti_simple_titan_20260414_1000/best_reti_simple_titan.pth
"""

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import (
    BACKBONE_CONFIG,
    CLASS_NAMES,
    collate_bags,
    validate_and_evaluate,
)
from models.simple_mil import SimpleGatedMIL
from models.hybrid_mil import HybridMIL
from models.dual_stream_mil import DualStreamMIL
from models.multi_branch_mil import MultiBranchMIL
from models.mean_pool_mil import MeanPoolMIL


# ── helpers ───────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _make_log_fn(log_path: Path):
    """Return a ``log(msg)`` closure that prints AND appends to *log_path*."""

    def log(msg: str = "") -> None:
        print(msg)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    return log


# ── argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation.

    Only runtime knobs are exposed here; model-related settings
    (backbone, model_type, topk, formulation) are inferred from
    the checkpoint automatically.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MIL model for Reticulin Fibrosis Grading (G0–G3). "
                    "Model configuration is inferred from the checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained .pth model checkpoint.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--map_scale",
        action="store_true",
        help="Map scalebar_micron from --scalebar_csv onto the predictions CSV (disabled by default).",
    )
    parser.add_argument(
        "--scalebar_csv",
        type=str,
        default="results/scalebar_results.csv",
        help="Path to scalebar_results.csv (used only when --map_scale is set).",
    )
    return parser.parse_args()


# ── model construction ────────────────────────────────────────────────────

def build_model(
    model_type: str,
    backbone: str,
    num_classes: int,
    topk: int,
    device: torch.device,
) -> nn.Module:
    """
    Instantiate the MIL model matching the training configuration.

    Args:
        model_type: One of 'simple', 'hybrid', 'dual_stream', 'multi_branch', 'mean_pool'.
        backbone: Backbone key used to look up input dimension.
        num_classes: Number of output classes (4 for classification, 1 for regression).
        topk: Top-k pooling parameter.
        device: Target device.

    Returns:
        The model placed on *device*.
    """
    cfg = BACKBONE_CONFIG[backbone]
    input_dim = cfg["dim"]

    if model_type == "simple":
        model = SimpleGatedMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=topk,
        )
    elif model_type == "hybrid":
        k = topk if topk > 0 else 5
        model = HybridMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=k,
        )
    elif model_type == "dual_stream":
        k = topk if topk > 0 else 5
        model = DualStreamMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=k,
        )
    elif model_type == "multi_branch":
        model = MultiBranchMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk_focal=5,
        )
    elif model_type == "mean_pool":
        model = MeanPoolMIL(
            vision_dim=input_dim,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


# ── confusion-matrix visualisation ────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and save a high-quality confusion matrix using seaborn.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.
        save_path: Destination path for the PNG file.
        title: Plot title.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    labels = list(range(len(class_names)))
    cm = sk_cm(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0,
        cbar_kws={"shrink": 0.8},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── quick inference loop (for confusion-matrix data) ──────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    formulation: str = "classification",
) -> tuple:
    """
    Run inference and collect per-sample predictions and labels.

    Returns:
        all_labels: List[int] of ground-truth labels.
        all_preds:  List[int] of predicted labels.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for features_list, labels, slide_ids in tqdm(loader, desc="  Infer", leave=False):
        labels = labels.to(device)

        for i, features in enumerate(features_list):
            features = features.to(device)
            label = labels[i].item()

            if isinstance(model, SimpleGatedMIL):
                logits, _, _ = model(features, return_attention=False)
            else:
                logits, _, _ = model(features)

            if formulation == "regression":
                pred = int(max(0, min(3, round(logits.view(-1).item()))))
            else:
                pred = logits.argmax().item()

            all_preds.append(pred)
            all_labels.append(label)

    return all_labels, all_preds


# ── predictions CSV ───────────────────────────────────────────────────────

GRADE_NAMES = ("G0", "G1", "G2", "G3")


def _extract_grade_from_folder(folder_name: str) -> Optional[str]:
    """Return the grade substring (e.g. 'G2') found in *folder_name*, or None."""
    for grade in GRADE_NAMES:
        if grade in folder_name:
            return grade
    return None


def _load_scalebar_lookup(scalebar_csv: Path) -> dict:
    """Read scalebar_results.csv into a (subtype, patient, filename) -> micron dict."""
    lookup: dict = {}
    with open(scalebar_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["subtype"], row["patient"], row["filename"])
            lookup[key] = row["scalebar_micron"]
    return lookup


def write_predictions_csv(
    full_dataset: GradingBagDatasetFull,
    test_idx: List[int],
    all_labels: List[int],
    all_preds: List[int],
    class_names: List[str],
    save_path: Path,
    scalebar_csv: Optional[Path] = None,
) -> None:
    """
    Write per-sample predictions to ``save_path``.

    Columns: subtype, patient, grade, filename, slide_id,
             true_label, true_grade, pred_label, pred_grade, correct
    When *scalebar_csv* is provided, an additional ``scalebar_micron`` column is
    populated by matching on (subtype, patient, filename).
    """
    scalebar_lookup = _load_scalebar_lookup(scalebar_csv) if scalebar_csv else None

    fieldnames = [
        "subtype",
        "patient",
        "grade",
        "filename",
        "slide_id",
        "true_label",
        "true_grade",
        "pred_label",
        "pred_grade",
        "correct",
    ]
    if scalebar_lookup is not None:
        fieldnames.append("scalebar_micron")

    with open(save_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, idx in enumerate(test_idx):
            pt_path = full_dataset.get_slide_path(idx)
            slide_id = pt_path.stem
            patient = pt_path.parent.name
            subtype = pt_path.parent.parent.name
            filename = f"{slide_id}.tif"
            grade = _extract_grade_from_folder(patient)

            true_label = int(all_labels[i])
            pred_label = int(all_preds[i])

            record = {
                "subtype": subtype,
                "patient": patient,
                "grade": grade or "",
                "filename": filename,
                "slide_id": slide_id,
                "true_label": true_label,
                "true_grade": class_names[true_label],
                "pred_label": pred_label,
                "pred_grade": class_names[pred_label],
                "correct": int(true_label == pred_label),
            }
            if scalebar_lookup is not None:
                record["scalebar_micron"] = scalebar_lookup.get(
                    (subtype, patient, filename), ""
                )
            writer.writerow(record)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint early & extract config ────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer training config from the saved checkpoint
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt_args.get("backbone") or ckpt.get("backbone")
    model_type = ckpt_args.get("model_type") or ckpt.get("model_type", "simple")
    topk = ckpt_args.get("topk", 0)
    formulation = ckpt_args.get("formulation", "classification")
    seed = ckpt_args.get("seed", 42)
    saved_epoch = ckpt.get("epoch", "?")

    if backbone is None:
        raise ValueError(
            "Cannot infer 'backbone' from checkpoint. "
            "Ensure the checkpoint was saved with an 'args' dict or a top-level 'backbone' key."
        )

    set_seed(seed)

    # ── Output directory ──────────────────────────────────────────────
    experiment_name = checkpoint_path.parent.name
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / "eval" / experiment_name / f"run_{eval_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────
    log_path = out_dir / "eval.log"
    log = _make_log_fn(log_path)

    log(f"Checkpoint: {checkpoint_path}")
    log(f"{'=' * 60}")
    log(f"Evaluation – Reticulin Fibrosis Grading (G0–G3)")
    log(f"{'=' * 60}")
    log(f"\nOutput directory: {out_dir}")
    log(f"Device: {device}")

    log("\nCLI Arguments:")
    for key, value in vars(args).items():
        log(f"  --{key}: {value}")

    log(f"\nInferred from checkpoint:")
    log(f"  Backbone:     {backbone}")
    log(f"  Model type:   {model_type}")
    log(f"  Top-k:        {topk}")
    log(f"  Formulation:  {formulation}")
    log(f"  Seed:         {seed}")
    log(f"  Epoch saved:  {saved_epoch}")

    # ── Backbone config ───────────────────────────────────────────────
    cfg = BACKBONE_CONFIG[backbone]
    features_dir = Path(args.data_root) / cfg["feature_dir"]
    num_classes = 1 if formulation == "regression" else 4

    log(f"\nBackbone: {cfg['display_name']}  (dim={cfg['dim']})")
    log(f"Feature directory: {features_dir}")
    log(f"Formulation: {formulation}  (num_classes={num_classes})")

    # ── Dataset & split (use exact saved test indices) ────────────────
    log(f"\nLoading dataset from: {features_dir}")
    full_dataset = GradingBagDatasetFull(features_dir)
    log(f"  Total bags: {len(full_dataset)}")

    test_idx = ckpt["test_idx"]
    log(f"  Test split (from checkpoint): {len(test_idx)} bags")

    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bags,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(
        model_type=model_type,
        backbone=backbone,
        num_classes=num_classes,
        topk=topk,
        device=device,
    )

    # Load trained weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f"\nLoaded checkpoint (epoch {saved_epoch})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model: {model_type.upper()}")
    log(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # ── Criterion ─────────────────────────────────────────────────────
    if formulation == "regression":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Evaluate ──────────────────────────────────────────────────────
    log(f"\n{'=' * 60}")
    log("Running evaluation on TEST set …")
    log(f"{'=' * 60}")

    (
        test_loss,
        test_acc,
        test_f1,
        test_recall,
        test_macro_recall,
        test_qwk,
        test_mae,
        test_cm_str,
        test_report_str,
    ) = validate_and_evaluate(
        model,
        test_loader,
        criterion,
        device,
        desc="  Test ",
        formulation=formulation,
    )

    # ── Log results ───────────────────────────────────────────────────
    log(f"\n{'─' * 60}")
    log("TEST RESULTS")
    log(f"{'─' * 60}")
    log(f"  Loss:          {test_loss:.4f}")
    log(f"  Accuracy:      {test_acc:.2f}%")
    log(f"  F1 (Macro):    {test_f1:.3f}")
    log(f"  Macro Recall:  {test_macro_recall:.2f}%")
    log(f"  QWK:           {test_qwk:.3f}")
    log(f"  MAE:           {test_mae:.3f}")

    log(f"\n{test_cm_str}")
    log(f"\n{test_report_str}")

    # ── Confusion-matrix plot ─────────────────────────────────────────
    all_labels, all_preds = collect_predictions(
        model, test_loader, device, formulation=formulation
    )

    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        all_labels,
        all_preds,
        CLASS_NAMES,
        cm_path,
        title=f"Reticulin Grading – {model_type.upper()} / {cfg['display_name']}",
    )
    log(f"\nConfusion-matrix plot saved to: {cm_path}")

    # ── Predictions CSV ───────────────────────────────────────────────
    scalebar_path: Optional[Path] = None
    if args.map_scale:
        scalebar_path = Path(args.scalebar_csv)
        if not scalebar_path.exists():
            raise FileNotFoundError(f"Scalebar CSV not found: {scalebar_path}")

    predictions_csv = out_dir / "predictions.csv"
    write_predictions_csv(
        full_dataset=full_dataset,
        test_idx=test_idx,
        all_labels=all_labels,
        all_preds=all_preds,
        class_names=CLASS_NAMES,
        save_path=predictions_csv,
        scalebar_csv=scalebar_path,
    )
    log(f"Predictions CSV saved to: {predictions_csv}")
    if scalebar_path is not None:
        log(f"  Scalebar mapped from: {scalebar_path}")

    log(f"\n{'=' * 60}")
    log("Evaluation complete.")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()

