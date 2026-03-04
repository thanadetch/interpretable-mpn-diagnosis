"""
Analyze MIL Model Shortcuts (Space vs Tissue).

This script reads pre-extracted features and raw patches, runs the MIL model
to get attention weights, and calculates whether the model is focusing on
"space" (shortcut) or "tissue" (morphology) using CV-based heuristics.

Output:
    results/shortcut_analysis/{exp_name}/shortcut_analysis.csv

Usage:
    python src/tools/analyze_shortcuts.py \
        --mil_checkpoint experiments/simple_titan.../best_model.pth \
        --patches_dir data/processed_subtype \
        --output_dir results/shortcut_analysis

    # Analyze only test set patients:
    python src/tools/analyze_shortcuts.py \
        --mil_checkpoint experiments/simple_titan.../best_model.pth \
        --split test
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CLASS_MAP, CLASS_MAP_INV

from models.hybrid_mil import HybridMIL
from models.simple_mil import SimpleGatedMIL

CLASS_NAMES = [CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))]

# =============================================================================
# Helper Functions (Doctor-Free Metrics)
# =============================================================================


def tissue_fraction_from_rgb(
    patch_rgb_uint8: np.ndarray, tissue_thr: float = 0.05
) -> float:
    """Calculate tissue fraction using Optical Density (OD)."""
    img = np.clip(patch_rgb_uint8.astype(np.float32), 1.0, 255.0)
    od = -np.log10(img / 255.0)  # (H,W,3)
    mean_od = od.mean(axis=2)  # (H,W)
    tissue_mask = mean_od > tissue_thr
    return float(tissue_mask.mean())


def space_bg_fractions_from_rgb(
    patch_rgb_uint8: np.ndarray,
    v_thr: float = 0.90,
    s_thr: float = 0.15,
    min_comp_area: int = 80,
    max_comp_area_ratio: float = 0.35,
) -> Tuple[float, float]:
    """
    Returns:
      space_frac: internal bright low-sat holes fraction (NOT touching border) / total_area
      bg_blank_frac: border-connected bright low-sat fraction (touching border) / total_area
    """
    H, W, _ = patch_rgb_uint8.shape
    hsv = cv2.cvtColor(patch_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    white = (V > v_thr) & (S < s_thr)
    white_u8 = white.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white_u8, connectivity=8)

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    outside_white = np.zeros((H, W), dtype=bool)
    internal_white = np.zeros((H, W), dtype=bool)

    max_comp_area = int(max_comp_area_ratio * H * W)

    for lab in range(1, n):
        area = stats[lab, cv2.CC_STAT_AREA]
        comp = labels == lab

        if (comp & border).any():
            # Border-connected (Background): Only filter small noise, don't filter max area
            if area >= min_comp_area:
                outside_white |= comp
        else:
            # Internal holes (Space): Filter both small noise and excessively large tears
            if area < min_comp_area or area > max_comp_area:
                continue
            internal_white |= comp

    # Both metrics are now normalized to the total patch area to prevent inflation
    bg_blank_frac = float(outside_white.mean())
    space_frac = float(internal_white.mean())
    return space_frac, bg_blank_frac


def summarize_roi(
    attn: np.ndarray,
    space_fracs: List[float],
    tissue_fracs: List[float],
    bg_blank_fracs: List[float],
) -> dict:
    """Summarize attention behaviors for a single ROI."""
    attn = np.asarray(attn, dtype=np.float32)

    # Defensive check: If attention weights contain negative values or sum is far from 1,
    # it implies they are raw logits. Apply softmax first.
    if (attn < 0).any() or attn.sum() > 1.01 or attn.sum() < 0.99:
        e_x = np.exp(attn - np.max(attn))  # Subtract max for numerical stability
        attn = e_x / (e_x.sum() + 1e-6)
    else:
        # If they are already probabilities, just ensure they sum to 1 precisely
        attn = attn / (attn.sum() + 1e-6)

    space_fracs = np.asarray(space_fracs, dtype=np.float32)
    tissue_fracs = np.asarray(tissue_fracs, dtype=np.float32)
    bg_blank_fracs = np.asarray(bg_blank_fracs, dtype=np.float32)

    # Core weighted sums
    attn_space = float((attn * space_fracs).sum())
    attn_tissue = float((attn * tissue_fracs).sum())
    attn_bg = float((attn * bg_blank_fracs).sum())

    # Thesis-friendly Threshold Metrics (Tau = 20%)
    tau_tissue = 0.20
    tau_bg = 0.20
    tau_space = 0.20
    attn_low_tissue = float((attn * (tissue_fracs < tau_tissue)).sum())
    attn_high_bg = float((attn * (bg_blank_fracs > tau_bg)).sum())
    attn_high_space = float((attn * (space_fracs > tau_space)).sum())

    # Good Tissue Attention (Tissue > 60% and Background < 20%)
    tau_good_tissue = 0.60
    attn_good = float(
        (attn * ((tissue_fracs > tau_good_tissue) & (bg_blank_fracs < tau_bg))).sum()
    )

    # Top-1 specific metrics
    top1_idx = int(np.argmax(attn))
    top1_tissue = float(tissue_fracs[top1_idx])
    top1_bg = float(bg_blank_fracs[top1_idx])
    top1_space = float(space_fracs[top1_idx])
    top1_is_low_tissue = int(top1_tissue < tau_tissue)

    # Sanity check: Correlation between attention and tissue density
    corr = np.corrcoef(attn, tissue_fracs)[0, 1]
    corr_attn_tissue = float(0.0 if np.isnan(corr) else corr)

    # Dynamic Top-K (Top 20% of patches, min 3)
    k = max(3, int(0.2 * len(attn)))
    idx = np.argsort(-attn)[:k]

    topk_space_mean = float(space_fracs[idx].mean())
    all_space_mean = float(space_fracs.mean())
    space_enrichment = topk_space_mean - all_space_mean

    topk_tissue_mean = float(tissue_fracs[idx].mean())
    all_tissue_mean = float(tissue_fracs.mean())
    tissue_enrichment = topk_tissue_mean - all_tissue_mean

    topk_bg_mean = float(bg_blank_fracs[idx].mean())
    all_bg_mean = float(bg_blank_fracs.mean())
    bg_enrichment = topk_bg_mean - all_bg_mean

    return {
        "attn_space": attn_space,
        "attn_tissue": attn_tissue,
        "attn_bg": attn_bg,
        "attn_low_tissue": attn_low_tissue,
        "attn_high_bg": attn_high_bg,
        "attn_high_space": attn_high_space,
        "attn_good": attn_good,
        "top1_tissue": top1_tissue,
        "top1_bg": top1_bg,
        "top1_space": top1_space,
        "top1_is_low_tissue": top1_is_low_tissue,
        "corr_attn_tissue": corr_attn_tissue,
        "topk_space_mean": topk_space_mean,
        "all_space_mean": all_space_mean,
        "space_enrichment": space_enrichment,
        "topk_tissue_mean": topk_tissue_mean,
        "all_tissue_mean": all_tissue_mean,
        "tissue_enrichment": tissue_enrichment,
        "topk_bg_mean": topk_bg_mean,
        "all_bg_mean": all_bg_mean,
        "bg_enrichment": bg_enrichment,
    }


# =============================================================================
# Core Pipeline
# =============================================================================


def parse_patch_filename(filename: str) -> Tuple[str, int, int]:
    """Parse '1_r2c5.png' -> ('1', 2, 5)"""
    match = re.match(r"^(\d+)_r(\d+)c(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Unexpected patch format: {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def load_mil_model(checkpoint_path: Path, device: torch.device):
    """Load MIL model from checkpoint. Returns (model, backbone_name, ckpt_args, checkpoint)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint["args"]
    model_type = ckpt_args["model_type"]
    backbone_name = ckpt_args["backbone"]

    # Simple hardcode for dimensions based on your config
    dims = {"titan": 768, "uni2": 1536, "virchow2": 1280}
    input_dim = dims[backbone_name]
    num_classes = len(CLASS_MAP)

    if model_type == "simple":
        topk = ckpt_args.get("topk", 0)
        model = SimpleGatedMIL(input_dim=input_dim, num_classes=num_classes, topk=topk)
    else:
        model = HybridMIL(input_dim=input_dim, num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, backbone_name, ckpt_args, checkpoint


def generate_dashboard(csv_path: Path, out_path: Path, exp_name: str, split: str):
    """Generate a comprehensive 1-page visualization dashboard."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return

        # Metrics Prep
        y_true = df["gt_class"]
        y_pred = df["pred_class"]
        classes = ["ET", "PV", "PMF"]  # Standard order

        acc = accuracy_score(y_true, y_pred)
        recalls = recall_score(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        recall_dict = dict(zip(classes, recalls))
        pv_recall = recall_dict.get("PV", 0)
        et_recall = recall_dict.get("ET", 0)

        n_roi = len(df)
        n_patients = df["patient_id"].nunique()

        # Figure setup (Increased height and adjusted ratios to prevent stretched wide plots)
        fig = plt.figure(figsize=(18, 18))
        gs = gridspec.GridSpec(
            3, 3, height_ratios=[1, 1.2, 1.2], hspace=0.5, wspace=0.4
        )
        sns.set_theme(style="whitegrid", font_scale=1.1)

        # A) Header
        fig.suptitle(
            f"Shortcut Analysis Dashboard | Exp: {exp_name} | Split: {split.upper()}\n"
            f"Patients: {n_patients} | ROIs: {n_roi} | Acc: {acc:.1%} | Macro Recall: {macro_recall:.1%}\n"
            f"PV Recall: {pv_recall:.1%} | ET Recall: {et_recall:.1%}",
            fontsize=18,
            fontweight="bold",
            y=0.97,
        )

        # B) Panel 1: Confusion Matrix (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax1,
            cbar=False,
        )
        ax1.set_title("Confusion Matrix (ROI-level)", fontweight="bold")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Ground Truth")

        # C) Panel 2: Per-class Recall Bar (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(x=classes, y=recalls, ax=ax2, palette="viridis")
        ax2.set_title("Per-class Recall", fontweight="bold")
        ax2.set_ylim(0, 1.05)
        for i, v in enumerate(recalls):
            ax2.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

        # F) Panel 5: PV Focus (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        pv_df = df[df["gt_class"] == "PV"]
        if not pv_df.empty:
            pv_preds = pv_df["pred_class"].value_counts().reindex(classes).fillna(0)
            sns.barplot(x=pv_preds.index, y=pv_preds.values, ax=ax3, palette="Set2")
            ax3.set_title("PV Ground Truth Breakdown", fontweight="bold")
            ax3.set_ylabel("Predicted Count")
            for i, v in enumerate(pv_preds.values):
                ax3.text(
                    i,
                    v + (pv_preds.max() * 0.02),
                    int(v),
                    ha="center",
                    fontweight="bold",
                )

        # D) Panel 3: Shortcut Indicators (Middle Row, spans all 3 cols)
        ax4 = fig.add_subplot(gs[1, :])
        df["Prediction"] = df["is_correct"].map({1: "Correct", 0: "Incorrect"})

        if all(col in df.columns for col in ["attn_high_bg", "attn_good"]):
            df_melt = df.melt(
                id_vars=["Prediction", "gt_class"],
                value_vars=["attn_high_bg", "attn_good"],
                var_name="Metric",
                value_name="Value",
            )
            df_melt["Metric"] = df_melt["Metric"].map(
                {
                    "attn_high_bg": "Attn on High Background\n(Shortcut: Lower is better)",
                    "attn_good": "Attn on Quality Tissue\n(Morphology: Higher is better)",
                }
            )
            sns.boxplot(
                data=df_melt,
                x="Metric",
                y="Value",
                hue="Prediction",
                palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
                ax=ax4,
                showfliers=False,
            )
            ax4.set_title(
                "Shortcut Mechanism Evidence (Correct vs Incorrect)", fontweight="bold"
            )
            ax4.set_ylabel("Attention Fraction")
            ax4.set_xlabel("")

        # E) Panel 4: ROI Quality Effect (Bottom Row, spans all 3 cols)
        ax5 = fig.add_subplot(gs[2, :])
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        df["bg_bin"] = pd.cut(
            df["all_bg_mean"], bins=bins, labels=labels, include_lowest=True
        )

        bg_stats = (
            df.groupby("bg_bin", observed=False)
            .agg(Accuracy=("is_correct", "mean"), Count=("is_correct", "count"))
            .reset_index()
        )

        sns.barplot(data=bg_stats, x="bg_bin", y="Accuracy", color="#3498db", ax=ax5)
        ax5.set_title(
            "ROI Quality Effect (Accuracy vs Background Fraction)", fontweight="bold"
        )
        ax5.set_ylim(0, 1.05)
        ax5.set_xlabel("Background Fraction in ROI (all_bg_mean)")
        ax5.set_ylabel("Accuracy")

        for i, row in bg_stats.iterrows():
            if row["Count"] > 0:
                ax5.text(
                    i,
                    row["Accuracy"] + 0.02,
                    f"{row['Accuracy']:.1%}\n(n={row['Count']})",
                    ha="center",
                    fontweight="bold",
                    fontsize=10,
                )

        # Ignore tight_layout warnings and adjust rect to leave more room for the taller suptitle
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Dashboard successfully saved to: {out_path}")
    except Exception as e:
        print(f"⚠️ Could not generate dashboard: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MIL Model Shortcuts")
    parser.add_argument(
        "--mil_checkpoint",
        type=str,
        required=True,
        help="Path to best MIL model (.pth)",
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        default="data/processed_subtype",
        help="Directory of patches",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Override features dir (auto-inferred if None)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/shortcut_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (auto-inferred from checkpoint if None).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MIL Model from {args.mil_checkpoint}...")
    model, backbone_name, ckpt_args, checkpoint = load_mil_model(
        Path(args.mil_checkpoint), device
    )

    patches_root = Path(args.patches_dir)
    features_root = (
        Path(args.features_dir)
        if args.features_dir
        else patches_root.parent / f"features_{backbone_name}"
    )

    # Derive experiment name from checkpoint path
    exp_name = Path(args.mil_checkpoint).parent.name
    base_out_dir = Path(args.output_dir) / exp_name
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target patients for the chosen split
    seed = args.seed if args.seed is not None else ckpt_args.get("seed", 42)

    if args.split != "all":
        from data.bag_dataset import MPNBagDatasetFull

        dataset = MPNBagDatasetFull(
            features_root, max_patches=ckpt_args.get("max_patches")
        )

        # 1. Best Practice: Explicitly load target split directly from checkpoint
        if args.split == "train" and "train_idx" in checkpoint:
            print("✅ Loading 'train' patients directly from checkpoint...")
            indices = checkpoint["train_idx"]

        elif args.split == "val" and "val_idx" in checkpoint:
            print("✅ Loading 'val' patients directly from checkpoint...")
            indices = checkpoint["val_idx"]

        elif args.split == "test" and "test_idx" in checkpoint:
            print("✅ Loading 'test' patients directly from checkpoint...")
            indices = checkpoint["test_idx"]

        # 2. Fallback: If not found (older models), fallback to the original seed-based split logic
        else:
            print(
                f"⚠️ '{args.split}_idx' not found in checkpoint! Falling back to original seed logic..."
            )
            from train_mil import patient_split

            train_idx, val_idx, test_idx = patient_split(dataset, seed=seed)
            if args.split == "train":
                indices = train_idx
            elif args.split == "val":
                indices = val_idx
            else:
                indices = test_idx

        target_patients = set(dataset.samples[i][0].parent.name for i in indices)
        print(f"Found {len(target_patients)} patients in the '{args.split}' set.")

    else:
        target_patients = None

    results = []
    bad_patch_count = 0

    print(f"Scanning directories...")
    print(f"Output: {base_out_dir}")
    # Find all patient directories (e.g., data/processed_subtype/PV/PV1 G2)
    for class_dir in patches_root.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        gt_class = class_dir.name

        for patient_dir in class_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name

            # Skip patients not in the target split
            if target_patients is not None and patient_id not in target_patients:
                continue

            # Group patches by Image ID
            img_groups = defaultdict(list)
            for patch_file in patient_dir.glob("*.png"):
                try:
                    img_id, r, c = parse_patch_filename(patch_file.name)
                    img_groups[img_id].append((patch_file, r, c))
                except ValueError:
                    bad_patch_count += 1

            # Process each image (ROI)
            for img_id, patches in tqdm(
                img_groups.items(), desc=f"Analyzing {patient_id}", leave=False
            ):
                patches.sort(key=lambda x: (x[1], x[2]))  # Ensure sorted order

                feature_path = features_root / gt_class / patient_id / f"{img_id}.pt"
                if not feature_path.exists():
                    continue

                # Get MIL Attention
                with torch.inference_mode():
                    data = torch.load(
                        feature_path, map_location=device, weights_only=False
                    )

                    if isinstance(data, dict):
                        features = data.get("feats", data.get("features")).to(device)
                        metrics = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in data.get("metrics", {}).items()
                        }
                    else:
                        features = data.to(device)
                        metrics = {}

                    if ckpt_args.get("model_type", "simple") == "simple":
                        logits, attention, _ = model(
                            features, return_attention=True, metrics=metrics
                        )
                    else:
                        logits, attention, _ = model(features, return_attention=True)

                    pred_class = CLASS_NAMES[logits.argmax().item()]
                    attention = attention.cpu().numpy().flatten()

                # Check patch mismatch
                if len(attention) != len(patches):
                    print(
                        f"Mismatch in {patient_id}/{img_id}: {len(attention)} features vs {len(patches)} patches. Skipping."
                    )
                    continue

                # Calculate Space, BG Blank & Tissue fractions per patch
                space_fracs, bg_blank_fracs, tissue_fracs = [], [], []
                for patch_path, _, _ in patches:
                    img_bgr = cv2.imread(str(patch_path))
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    space_f, bg_f = space_bg_fractions_from_rgb(img_rgb)
                    space_fracs.append(space_f)
                    bg_blank_fracs.append(bg_f)
                    tissue_fracs.append(tissue_fraction_from_rgb(img_rgb))

                # Summarize
                summary = summarize_roi(
                    attention, space_fracs, tissue_fracs, bg_blank_fracs
                )

                # Active Sanity Check Warning
                if summary["corr_attn_tissue"] < -0.5:
                    print(
                        f"⚠️ WARNING: Strong negative correlation ({summary['corr_attn_tissue']:.2f}) in {patient_id}/{img_id}. Possible patch-feature mismatch!"
                    )

                # Append to results
                row = {
                    "patient_id": patient_id,
                    "image_id": img_id,
                    "gt_class": gt_class,
                    "pred_class": pred_class,
                    "is_correct": int(gt_class == pred_class),
                    **summary,
                }
                results.append(row)

    # Save single master CSV
    if not results:
        print("No valid ROI results found. Check your directory paths.")
        return

    out_path = base_out_dir / f"shortcut_analysis_{args.split}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Analysis complete! Saved {len(results)} rows to {out_path}")

    # Generate Visualization Dashboard
    plot_out_path = base_out_dir / f"shortcut_dashboard_{args.split}.png"
    exp_name = Path(args.mil_checkpoint).parent.name
    generate_dashboard(out_path, plot_out_path, exp_name=exp_name, split=args.split)

    if bad_patch_count > 0:
        print(
            f"⚠️ Warning: Skipped {bad_patch_count} files due to invalid patch naming format."
        )


if __name__ == "__main__":
    main()
