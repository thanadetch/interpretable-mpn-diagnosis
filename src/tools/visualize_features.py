"""
Unified t-SNE visualization for backbone features (TITAN, UNI2-h, Virchow2).

Loads mean-pooled embeddings, applies StandardScaler, and plots a 2D t-SNE
scatter colour-coded by MPN subtype.

Usage:
    python -m src.tools.visualize_features --backbone titan
    python -m src.tools.visualize_features --backbone uni2 --data_root data
    python -m src.tools.visualize_features --backbone virchow2 --output my_plot.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.getcwd())
from src.core.config import CLASS_MAP, CLASS_MAP_INV

# ── backbone configuration ───────────────────────────────────────────────
BACKBONE_CONFIG = {
    "titan": {
        "feature_dir": "features_titan",
        "display_name": "TITAN",
    },
    "uni2": {
        "feature_dir": "features_uni2",
        "display_name": "UNI2-h",
    },
    "virchow2": {
        "feature_dir": "features_virchow2",
        "display_name": "Virchow2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize backbone features with t-SNE."
    )
    parser.add_argument(
        "--backbone",
        required=True,
        choices=list(BACKBONE_CONFIG.keys()),
        help="Backbone whose features to visualize.",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG filename. Defaults to tsne_{backbone}.png.",
    )
    return parser.parse_args()


def visualize_tsne(
    features_dir: Path,
    display_name: str,
    output_file: str,
) -> None:
    features = []
    labels = []
    counts = {}

    # Load all data (rglob handles both flat and nested directory layouts)
    print(f"Loading {display_name} features...")
    for class_name, label in CLASS_MAP.items():
        class_dir = features_dir / class_name
        if not class_dir.exists():
            print(f"  ⚠ Class directory not found: {class_dir}")
            continue

        pt_files = sorted(class_dir.rglob("*.pt"))
        counts[class_name] = len(pt_files)

        for pt_file in pt_files:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                feat = data["feats"]
            else:
                feat = data
            # Mean pooling: [N_patches, Dim] -> [Dim]
            feat_mean = feat.mean(dim=0).numpy()
            features.append(feat_mean)
            labels.append(label)

        print(f"  {class_name}: {counts[class_name]} bags loaded")

    X = np.array(features)
    y = np.array(labels)
    print(f"Total: {len(X)} bags, feature dim: {X.shape[1]}")

    # Standardize features before t-SNE
    print("Applying StandardScaler...")
    X_scaled = StandardScaler().fit_transform(X)

    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42,
        learning_rate="auto",
    )
    X_embedded = tsne.fit_transform(X_scaled)

    # Plot
    plt.figure(figsize=(10, 8))
    colors = ["#e41a1c", "#4daf4a", "#377eb8"]  # ET (Red), PV (Green), PMF (Blue)
    target_names = [CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))]

    for i, name in enumerate(target_names):
        mask = y == i
        n = mask.sum()
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=colors[i],
            label=f"{name} (n={n})",
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
            s=40,
        )

    plt.title(f"t-SNE of {display_name} Features (Mean Pooling)")
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot to {output_file}")


if __name__ == "__main__":
    args = parse_args()

    cfg = BACKBONE_CONFIG[args.backbone]
    features_dir = Path(args.data_root) / cfg["feature_dir"]
    output_file = args.output or f"tsne_{args.backbone}.png"

    visualize_tsne(features_dir, cfg["display_name"], output_file)
