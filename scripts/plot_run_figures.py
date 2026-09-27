"""Plot training curves + confusion matrices for an experiment run.

Usage:
    python scripts/plot_run_figures.py <run_dir> <tag> [cm_title]

Outputs (into results/figures/):
    <tag>_loss_qwk.png
    <tag>_val_cm.png
    <tag>_test_cm.png

If cm_title is provided, it replaces the default CM title. The split
name (val / test) is appended in parentheses.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_qwk(df: pd.DataFrame, out_path: str) -> None:
    c_train = "#d62728"  # red
    c_val = "#1f77b4"    # blue

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], color=c_train, marker="o", ms=4, lw=1.4, label="Train")
    ax.plot(df["epoch"], df["val_loss"],   color=c_val,   marker="s", ms=4, lw=1.4, label="Validation")
    ax.set_title("(a) Loss across epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(df["epoch"], df["train_qwk"], color=c_train, marker="o", ms=4, lw=1.4, label="Train")
    ax.plot(df["epoch"], df["val_qwk"],   color=c_val,   marker="s", ms=4, lw=1.4, label="Validation")
    ax.set_title("(b) QWK across epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Quadratic Weighted Kappa (QWK)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cm(cm_csv: str, title: str, out_path: str) -> None:
    cm_df = pd.read_csv(cm_csv, index_col=0)
    cm = cm_df.values
    labels = list(cm_df.columns)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")

    thresh = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11, pad=10)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    run_dir, tag = sys.argv[1], sys.argv[2]
    cm_title = sys.argv[3] if len(sys.argv) == 4 else f"Reticulin Grading – {tag}"

    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(run_dir, "training_log.csv"))
    plot_loss_qwk(df, os.path.join(out_dir, f"{tag}_loss_qwk.png"))
    print("saved:", os.path.join(out_dir, f"{tag}_loss_qwk.png"))

    for split in ("val", "test"):
        cm_csv = os.path.join(run_dir, f"{split}_confusion_matrix.csv")
        out = os.path.join(out_dir, f"{tag}_{split}_cm.png")
        plot_cm(cm_csv, f"{cm_title}  ({split})", out)
        print("saved:", out)


if __name__ == "__main__":
    main()



