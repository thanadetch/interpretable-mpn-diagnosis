"""
Sanity Check: Informativeness and Patch-level Scores.

Validates the patch-level scores by:
1.  Computing global Z-score normalization stats from a random sample of bags.
2.  Plotting global histograms for ALL score types.
3.  Generating per-ROI visual grids (Top-5 vs Bottom-5 patches) for ALL score types.

Input:  data/features_*/{Class}/{PatientID}/{ImgID}.pt
Output: results/sanity_check/{run_name}/{score_type}/{score_type}_hist.png
        results/sanity_check/{run_name}/{score_type}/rois/ROI_{img_id}.png

Usage:
    python -m src.tools.sanity_check_info \
        --features_dir data/features_virchow2 \
        --subtypes ET PV \
        --num_rois 20
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


# =============================================================================
# Constants
# =============================================================================
METRIC_KEYS = [
    "tissue_frac", 
    "border_white_frac", 
    "nuisance_score",
    "cellular_purple_frac",
    "pink_lowinfo_frac"
]
SCORE_TYPES_TO_CHECK = [
    "info_v1", 
    "cellular_purple_frac", 
    "pink_lowinfo_frac"
]



# =============================================================================
# Utility Functions
# =============================================================================


def discover_pt_files(features_dir: Path, subtypes: List[str]) -> List[Path]:
    """Recursively discover .pt files filtered by subtype."""
    pt_files = []
    for p in sorted(features_dir.rglob("*.pt")):
        try:
            # e.g., features_dir/ET/P001/1.pt -> rel_parts[0] is "ET"
            rel_parts = p.relative_to(features_dir).parts
            if rel_parts and rel_parts[0] in subtypes:
                pt_files.append(p)
        except ValueError:
            pass

    if not pt_files:
        raise FileNotFoundError(
            f"No .pt files found for subtypes {subtypes} under {features_dir}"
        )
    return pt_files


def load_bag(pt_path: Path) -> Dict:
    """Load a .pt bag file and return its dictionary."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return data


def extract_img_id(pt_path: Path) -> str:
    """Extract a human-readable image ID from a .pt path."""
    parts = pt_path.parts
    if len(parts) >= 3:
        return f"{parts[-3]}/{parts[-2]}/{pt_path.stem}"
    return pt_path.stem


# =============================================================================
# Step 1: Global Normalization (Z-score Statistics)
# =============================================================================


def compute_global_stats(
    pt_files: List[Path]
) -> Dict[str, Tuple[float, float]]:
    """
    Compute global mean & std for each metric from ALL discovered bags.
    """
    collectors: Dict[str, List[torch.Tensor]] = {k: [] for k in METRIC_KEYS}

    print(f"  Computing global stats from all {len(pt_files)} bags...")
    for pt_path in pt_files:
        try:
            data = load_bag(pt_path)
            metrics = data["metrics"]
            for key in METRIC_KEYS:
                if key in metrics:
                    collectors[key].append(metrics[key].float())
        except Exception as e:
            print(f"  ⚠ Skipping {pt_path.name}: {e}")

    stats: Dict[str, Tuple[float, float]] = {}
    for key in METRIC_KEYS:
        if not collectors[key]:
            raise ValueError(f"No data collected for metric '{key}'. Please re-extract features.")
        cat = torch.cat(collectors[key], dim=0)
        stats[key] = (float(cat.mean()), float(cat.std()))
        print(f"    {key:>20s}: mean={stats[key][0]:.4f}, std={stats[key][1]:.4f}")

    return stats


# =============================================================================
# Step 2: Score Calculation
# =============================================================================


def z_score(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Element-wise Z-score normalization."""
    return (x - mean) / (std + 1e-8)


def compute_score(
    metrics: Dict[str, torch.Tensor], 
    stats: Dict[str, Tuple[float, float]],
    score_type: str = "info_v1",
) -> torch.Tensor:
    """
    Compute the requested score for ranking patches.
    """
    if score_type == "info_v1":
        z_tissue = z_score(metrics["tissue_frac"].float(), *stats["tissue_frac"])
        z_border = z_score(
            metrics["border_white_frac"].float(), *stats["border_white_frac"]
        )
        z_nuisance = z_score(metrics["nuisance_score"].float(), *stats["nuisance_score"])
        return z_tissue - z_border - z_nuisance
    elif score_type == "cellular_purple_frac":
        return metrics["cellular_purple_frac"].float()
    elif score_type == "pink_lowinfo_frac":
        return metrics["pink_lowinfo_frac"].float()
    else:
        raise ValueError(f"Unknown score_type: {score_type}")


# =============================================================================
# Step 3: Dataset Histogram
# =============================================================================


def plot_histogram(
    pt_files: List[Path],
    stats: Dict[str, Tuple[float, float]],
    output_path: Path,
    subtypes: List[str] = None,
    score_type: str = "info_v1",
    seed: int = 42,
) -> None:
    """
    Compute scores for all patches across all bags and plot the histogram,
    color-coded by subtype.
    """
    print(f"  Computing {score_type} for {len(pt_files)} bags...")

    # Initialize dictionary to hold scores per subtype
    scores_by_subtype = {s: [] for s in (subtypes or ["Unknown"])}
    all_scores_flat = []

    for pt_path in pt_files:
        try:
            # Extract subtype from path (e.g., .../ET/P001/1.pt -> ET)
            parts = pt_path.parts
            subtype = parts[-3] if len(parts) >= 3 else "Unknown"
            if subtypes and subtype not in subtypes:
                subtype = "Unknown"

            data = load_bag(pt_path)
            scores = compute_score(data["metrics"], stats, score_type=score_type)
            
            if subtype not in scores_by_subtype:
                scores_by_subtype[subtype] = []
            scores_by_subtype[subtype].append(scores)
            all_scores_flat.append(scores)

        except Exception as e:
            print(f"  ⚠ Skipping {pt_path.name}: {e}")

    if not all_scores_flat:
        print("  ⚠ No scores to plot. Aborting histogram.")
        return

    # Concatenate flat list for global stats
    all_scores_cat = torch.cat(all_scores_flat, dim=0).numpy()
    total_patches = len(all_scores_cat)

    # Prepare data for stacked histogram
    hist_data = []
    hist_labels = []
    for s in (subtypes or ["Unknown"]):
        if s in scores_by_subtype and scores_by_subtype[s]:
            cat_scores = torch.cat(scores_by_subtype[s], dim=0).numpy()
            hist_data.append(cat_scores)
            hist_labels.append(s)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot stacked histogram
    if hist_data:
        ax.hist(hist_data, bins=100, stacked=True, edgecolor="black", alpha=0.75, label=hist_labels)
    
    ax.set_xlabel(f"{score_type}", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    subtypes_text = f" [{', '.join(hist_labels)}]" if hist_labels else ""
    ax.set_title(
        f"Global {score_type} Distribution{subtypes_text}\n(N = {total_patches:,} patches from {len(pt_files)} bags)",
        fontsize=13,
    )
    
    # Show z=0 line only for info_v1
    if score_type == "info_v1":
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="z=0")
        
    ax.legend(fontsize=10)

    # Add summary stats as text (Global + Per Subtype)
    stats_lines = []
    
    # Global stats
    mean_val = float(all_scores_cat.mean())
    std_val = float(all_scores_cat.std())
    med_val = float(np.median(all_scores_cat))
    stats_lines.append(f"ALL: mean={mean_val:.2f} | std={std_val:.2f} | med={med_val:.2f}")
    
    # Per-subtype stats
    for s in (subtypes or ["Unknown"]):
        if s in scores_by_subtype and scores_by_subtype[s]:
            cat_scores = torch.cat(scores_by_subtype[s], dim=0).numpy()
            s_mean = float(cat_scores.mean())
            s_std = float(cat_scores.std())
            s_med = float(np.median(cat_scores))
            stats_lines.append(f"{s:>3s}: mean={s_mean:.2f} | std={s_std:.2f} | med={s_med:.2f}")
            
    textstr = "\n".join(stats_lines)
    ax.text(
        0.98,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Histogram saved to {output_path}")


# =============================================================================
# Step 4: Visual Inspection — Top 5 vs Bottom 5 Grid
# =============================================================================


def create_roi_grid(
    pt_path: Path,
    stats: Dict[str, Tuple[float, float]],
    output_dir: Path,
    top_k: int = 5,
    score_type: str = "info_v1",
) -> None:
    """
    For one ROI (.pt file), create a 2×5 grid:
    Top row = Top-5 patches (highest score),
    Bottom row = Bottom-5 patches (lowest score).
    """
    data = load_bag(pt_path)
    metrics = data["metrics"]
    patch_paths = data["patch_paths"]  # List[str]

    scores = compute_score(metrics, stats, score_type=score_type)
    n_patches = scores.shape[0]

    if n_patches < top_k * 2:
        return

    # Sort by score descending
    sorted_indices = torch.argsort(scores, descending=True)
    top_indices = sorted_indices[:top_k]
    bottom_indices = sorted_indices[-top_k:]

    # Collect images and scores
    rows_data: List[List[Tuple[np.ndarray, float, str]]] = [
        [],
        [],
    ]  # [top_row, bottom_row]

    for row_idx, indices in enumerate([top_indices, bottom_indices]):
        for idx in indices:
            idx_int = int(idx)
            patch_path_str = patch_paths[idx_int]
            score_val = float(scores[idx_int])
            patch_path = Path(patch_path_str)

            if not patch_path.exists():
                placeholder = np.full((224, 224, 3), fill_value=200, dtype=np.uint8)
                placeholder[:, :, 0] = 255  # Red tint
                rows_data[row_idx].append((placeholder, score_val, "NOT FOUND"))
            else:
                img = np.array(Image.open(patch_path).convert("RGB"))
                rows_data[row_idx].append((img, score_val, ""))

    # Build figure
    img_id = extract_img_id(pt_path)
    fig, axes = plt.subplots(2, top_k, figsize=(3 * top_k, 7))

    row_labels = ["Top (Highest)", "Bottom (Lowest)"]
    for r in range(2):
        for c in range(top_k):
            ax = axes[r, c]
            img, score_val, status = rows_data[r][c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            title_color = "red" if status else "black"
            title_text = (
                f"{score_val:.2f}" if not status else f"{score_val:.2f}\n{status}"
            )
            ax.set_title(title_text, fontsize=9, color=title_color)

            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10, fontweight="bold")

    fig.suptitle(f"ROI: {img_id} | {score_type}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    # Extract class name and patient ID from path
    parts = pt_path.parts
    class_name = parts[-3] if len(parts) >= 3 else "Unknown"
    patient_id = parts[-2] if len(parts) >= 3 else "Unknown"

    # Save into subtype/patient subdirectory
    out_path = output_dir / class_name / patient_id / f"ROI_{pt_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_roi_grids(
    pt_files: List[Path],
    stats: Dict[str, Tuple[float, float]],
    output_dir: Path,
    num_rois: int = 20,
    seed: int = 42,
    score_type: str = "info_v1",
) -> None:
    """Generate Top-5 vs Bottom-5 grids for a random sample of ROIs."""
    rng = random.Random(seed)
    sample = rng.sample(pt_files, min(num_rois, len(pt_files)))

    rois_dir = output_dir / "rois"
    rois_dir.mkdir(parents=True, exist_ok=True)

    print(f"    Generating grids for {len(sample)} ROIs...")
    for pt_path in sample:
        try:
            create_roi_grid(pt_path, stats, rois_dir, top_k=5, score_type=score_type)
        except Exception as e:
            print(f"    ⚠ Error on {pt_path.name}: {e}")
    print(f"    ✅ Saved grids to {rois_dir}/")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check the patch-level informativeness scores (All-in-one).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Path to extracted features directory (e.g., data/features_virchow2).",
    )
    parser.add_argument(
        "--num_rois",
        type=int,
        default=20,
        help="Number of ROIs to visualize (default: 20).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sanity_check",
        help="Output directory for plots (default: results/sanity_check).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--subtypes",
        type=str,
        nargs="+",
        default=["ET", "PV", "PMF"],
        help="List of subtypes to process (e.g., --subtypes ET PV).",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="Optional postfix to append to the root feature folder and histogram name.",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        choices=["info_v1", "cellular_purple_frac", "pink_lowinfo_frac"],
        default="info_v1",
        help="Which metric/score to use for ranking and plotting.",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    seed = args.seed

    # Automatically detect feature type from directory name (e.g., "features_virchow2" -> "virchow2")
    feature_name = features_dir.name
    if feature_name.startswith("features_"):
        feature_name = feature_name.replace("features_", "")

    # Build suffix from subtypes and postfix
    subtypes_str = "_".join(args.subtypes)
    if args.postfix:
        combined_suffix = f"{subtypes_str}_{args.postfix}"
    else:
        combined_suffix = subtypes_str

    # Include score_type in the directory name so they don't overwrite each other
    feature_name = f"{feature_name}_{combined_suffix}_{args.score_type}"
    feature_output_dir = output_dir / feature_name

    print("=" * 60)
    print(f"Sanity Check: {args.score_type}")
    print("=" * 60)
    print(f"  Features dir : {features_dir}")
    print(f"  Feature type : {feature_name.upper()}")
    print(f"  Output dir   : {feature_output_dir}")
    print(f"  Subtypes     : {args.subtypes}")
    print(f"  Score type   : {args.score_type}")
    print(f"  Num ROIs     : {args.num_rois}")
    print(f"  Seed         : {seed}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Discover .pt files
    # ------------------------------------------------------------------
    print(f"\n[1/4] Discovering .pt files for subtypes: {args.subtypes}...")
    pt_files = discover_pt_files(features_dir, args.subtypes)
    print(f"  Found {len(pt_files)} .pt files.\n")

    # ------------------------------------------------------------------
    # Step 1: Global Normalization
    # ------------------------------------------------------------------
    print("[2/4] Computing global Z-score normalization stats...")
    stats = compute_global_stats(pt_files)
    print()

    # ------------------------------------------------------------------
    # Step 3: Histogram
    # ------------------------------------------------------------------
    print(f"[3/4] Plotting global {args.score_type} histogram...")
    hist_filename = f"{args.score_type}_hist_{combined_suffix}.png"
    hist_path = feature_output_dir / hist_filename
    plot_histogram(
        pt_files, stats, hist_path, subtypes=args.subtypes, score_type=args.score_type, seed=seed
    )
    print()

    # ------------------------------------------------------------------
    # Step 4: ROI Grids
    # ------------------------------------------------------------------
    print("[4/4] Generating per-ROI visual grids (Top-5 vs Bottom-5)...")
    generate_roi_grids(
        pt_files, stats, feature_output_dir, num_rois=args.num_rois, seed=seed, score_type=args.score_type
    )
    print()

    print("=" * 60)
    print("✅ Sanity check complete.")
    print(f"   Outputs saved to: {feature_output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
