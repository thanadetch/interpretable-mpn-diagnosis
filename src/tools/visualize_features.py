"""
Feature Quality Evaluation — PCA + PaCMAP + Quantitative Metrics.

Evaluates pre-extracted patch embeddings (e.g. from TITAN / CONCHv1.5,
Virchow2, UNI2-h) at two aggregation levels and on two class subsets.

Aggregation levels:
    • ROI-level   — mean-pool patches within each .pt file  →  1 vector / ROI.
    • Patient-level — mean-pool ROI vectors per patient      →  1 vector / patient.

Class subsets:
    • 3-class — ET vs PV vs PMF.
    • 2-class — ET vs PV only (hardest pair).

For each of the 4 combinations the script:
    1. Computes quantitative metrics on the *original* high-dim features:
       Silhouette Score, Davies-Bouldin Index, 5-NN accuracy (Stratified
       5-Fold CV), Linear Probe accuracy (Logistic Regression, Strat. 5-Fold).
    2. Reduces to 2-D with PCA and PaCMAP and saves a 1×2 scatter plot.

Outputs:
    {output_dir}/roi_3class.png
    {output_dir}/roi_2class.png
    {output_dir}/patient_3class.png
    {output_dir}/patient_2class.png
    {output_dir}/evaluation_metrics.csv

Usage:
    python -m src.tools.visualize_features
    python -m src.tools.visualize_features --features_dir data/features_virchow2
    python -m src.tools.visualize_features --output_dir results/feature_eval
"""

# Prevent duplicate-OpenMP segfault on macOS (PyTorch vs scipy/sklearn libomp)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pacmap
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# Constants
# =============================================================================
DEFAULT_FEATURES_DIR = "data/features_titan"
DEFAULT_OUTPUT_DIR = "results/feature_eval"
CLASSES_3 = ["ET", "PV", "PMF"]
CLASSES_2 = ["ET", "PV"]  # hardest binary pair
COLORS = {
    "ET": "#e41a1c",
    "PV": "#4daf4a",
    "PMF": "#377eb8",
}
SEED = 42


# =============================================================================
# 1. Feature Loading
# =============================================================================

def load_features(
    features_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load all .pt bags and mean-pool to ROI-level embeddings.

    Returns:
        roi_feats:      np.ndarray [N_rois, dim]
        roi_labels:     np.ndarray [N_rois]  (class name strings)
        roi_patients:   list[str]  patient id per ROI
        roi_img_ids:    list[str]  image id per ROI
    """
    roi_feats: List[np.ndarray] = []
    roi_labels: List[str] = []
    roi_patients: List[str] = []
    roi_img_ids: List[str] = []

    for class_name in CLASSES_3:
        class_dir = features_dir / class_name
        if not class_dir.exists():
            print(f"  ⚠ Class directory not found: {class_dir}")
            continue

        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            for pt_file in sorted(patient_dir.glob("*.pt")):
                data = torch.load(pt_file, map_location="cpu", weights_only=False)
                feats = data["feats"] if isinstance(data, dict) else data
                if feats.ndim != 2 or feats.shape[0] == 0:
                    continue
                roi_feats.append(feats.float().mean(dim=0).clone().numpy())
                roi_labels.append(class_name)
                roi_patients.append(patient_dir.name)
                roi_img_ids.append(pt_file.stem)


    X = np.stack(roi_feats)
    y = np.array(roi_labels)
    print(
        f"  Loaded {len(X)} ROIs  "
        f"(dim={X.shape[1]}; "
        + ", ".join(f"{c}={np.sum(y == c)}" for c in CLASSES_3)
        + ")"
    )
    return X, y, roi_patients, roi_img_ids


def aggregate_to_patient(
    roi_feats: np.ndarray,
    roi_labels: np.ndarray,
    roi_patients: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Mean-pool ROI embeddings per patient.

    Returns:
        patient_feats:   [N_patients, dim]
        patient_labels:  [N_patients]
        patient_ids:     list[str]
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, pid in enumerate(roi_patients):
        groups[pid].append(idx)

    patient_feats, patient_labels, patient_ids = [], [], []
    for pid in sorted(groups):
        idxs = groups[pid]
        patient_feats.append(roi_feats[idxs].mean(axis=0))
        patient_labels.append(roi_labels[idxs[0]])  # same class for all ROIs
        patient_ids.append(pid)

    X = np.stack(patient_feats)
    y = np.array(patient_labels)
    print(
        f"  Aggregated to {len(X)} patients  "
        f"("
        + ", ".join(f"{c}={np.sum(y == c)}" for c in CLASSES_3)
        + ")"
    )
    return X, y, patient_ids


# =============================================================================
# 2. Quantitative Metrics (High-Dimensional)
# =============================================================================

def compute_metrics(
    X: np.ndarray,
    y_str: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Compute clustering and classification metrics on the full-dim features.

    Returns dict with keys:
        silhouette, davies_bouldin, knn_5_acc, linear_probe_acc
    """
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    n_classes = len(np.unique(y))
    n_samples = len(y)
    min_class = min(np.bincount(y))

    results: Dict[str, float] = {}

    # --- Silhouette & DB Index ---
    if n_classes >= 2 and n_samples > n_classes:
        results["silhouette"] = float(silhouette_score(X_sc, y))
        results["davies_bouldin"] = float(davies_bouldin_score(X_sc, y))
    else:
        results["silhouette"] = float("nan")
        results["davies_bouldin"] = float("nan")

    # --- Cross-validated classifiers ---
    effective_folds = min(n_folds, min_class)
    if effective_folds < 2:
        warnings.warn(
            f"Smallest class has {min_class} samples — skipping CV metrics."
        )
        results["knn_5_acc"] = float("nan")
        results["linear_probe_acc"] = float("nan")
        return results

    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=SEED)

    # 5-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn, X_sc, y, cv=cv, scoring="accuracy")
    results["knn_5_acc"] = float(knn_scores.mean())

    # Linear probe (Logistic Regression)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_scores = cross_val_score(lr, X_sc, y, cv=cv, scoring="accuracy")
    results["linear_probe_acc"] = float(lr_scores.mean())

    return results


# =============================================================================
# 3. Dimensionality Reduction & Plotting
# =============================================================================

def plot_embeddings(
    X: np.ndarray,
    y_str: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """
    Create a 1×2 figure (PCA | PaCMAP) coloured by class label.
    """
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    classes = sorted(set(y_str))
    has_pacmap = pacmap is not None

    fig, axes = plt.subplots(
        1, 2 if has_pacmap else 1,
        figsize=(14 if has_pacmap else 7, 6),
    )
    if not has_pacmap:
        axes = [axes]

    # --- PCA ---
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_sc)
    _scatter(axes[0], X_pca, y_str, classes, f"{title}  —  PCA")
    var = pca.explained_variance_ratio_
    axes[0].set_xlabel(f"PC-1 ({var[0]:.1%})")
    axes[0].set_ylabel(f"PC-2 ({var[1]:.1%})")

    # --- PaCMAP ---
    if has_pacmap:
        n_samples = X_sc.shape[0]
        n_neighbors = min(10, n_samples - 1)  # PaCMAP default is 10
        pm = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=SEED)
        X_pm = pm.fit_transform(X_sc)
        _scatter(axes[1], X_pm, y_str, classes, f"{title}  —  PaCMAP")
        axes[1].set_xlabel("PaCMAP-1")
        axes[1].set_ylabel("PaCMAP-2")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved plot → {output_path}")


def _scatter(
    ax: plt.Axes,
    X_2d: np.ndarray,
    y_str: np.ndarray,
    classes: list,
    title: str,
) -> None:
    for cls in classes:
        mask = y_str == cls
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=COLORS[cls],
            label=f"{cls} (n={mask.sum()})",
            alpha=0.7,
            edgecolors="k",
            linewidths=0.4,
            s=40,
        )
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)


# =============================================================================
# 4. Evaluation Runner (one subset)
# =============================================================================

def evaluate_subset(
    X: np.ndarray,
    y: np.ndarray,
    subset_classes: List[str],
    level: str,
    output_dir: Path,
) -> Dict[str, object]:
    """
    Filter to *subset_classes*, run metrics + plotting.

    Args:
        level: "roi" or "patient"

    Returns:
        Row dict for the summary CSV.
    """
    mode = f"{len(subset_classes)}class"
    tag = f"{level}_{mode}"

    # Filter
    mask = np.isin(y, subset_classes)
    X_sub = X[mask]
    y_sub = y[mask]

    print(f"\n── {tag} ({len(X_sub)} samples) " + "─" * 40)

    # Metrics on full-dim features
    metrics = compute_metrics(X_sub, y_sub)
    for k, v in metrics.items():
        print(f"    {k:<20s}: {v:.4f}" if not np.isnan(v) else f"    {k:<20s}: N/A")

    # Plot
    plot_path = output_dir / f"{tag}.png"
    plot_embeddings(X_sub, y_sub, tag.replace("_", " ").title(), plot_path)

    return {"level": level, "mode": mode, **metrics}


# =============================================================================
# 5. Main
# =============================================================================

def run(args: argparse.Namespace) -> None:
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pacmap is None:
        print(
            "⚠ pacmap is not installed — PaCMAP plots will be skipped.\n"
            "  Install with:  pip install pacmap"
        )

    # ── Load features ────────────────────────────────────────────────
    print("Loading ROI-level features...")
    roi_feats, roi_labels, roi_patients, _ = load_features(features_dir)

    print("\nAggregating to patient-level...")
    pat_feats, pat_labels, _ = aggregate_to_patient(
        roi_feats, roi_labels, roi_patients
    )

    # ── Run 4 evaluation subsets ─────────────────────────────────────
    summary_rows: List[Dict] = []

    for subset in [CLASSES_3, CLASSES_2]:
        summary_rows.append(
            evaluate_subset(roi_feats, roi_labels, subset, "roi", output_dir)
        )
        summary_rows.append(
            evaluate_subset(pat_feats, pat_labels, subset, "patient", output_dir)
        )

    # ── Save summary CSV ─────────────────────────────────────────────
    csv_path = output_dir / "evaluation_metrics.csv"
    df = pd.DataFrame(summary_rows)
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved → {csv_path}")

    # ── Pretty-print table ───────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Feature Quality Evaluation Summary")
    print(f"{'=' * 72}")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"{'=' * 72}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate extracted feature quality with PCA, PaCMAP, "
            "Silhouette, DB Index, 5-NN, and Linear Probe."
        ),
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=DEFAULT_FEATURES_DIR,
        help="Root directory with .pt bags ({Class}/{Patient}/{Img}.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plots and metrics CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
