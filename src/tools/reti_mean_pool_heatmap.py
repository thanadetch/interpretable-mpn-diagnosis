"""
Reticulin Fibrosis Grading — Per-Patch Heatmaps for the Mean-Pooling MIL.

Generates region-level heatmaps that visualise where each patch contributes
to the slide-level fibrosis grade prediction produced by a trained
``MeanPoolMIL`` checkpoint (regression formulation gives a scalar grade in
[0, 3]; classification formulation gives an expected-grade scalar derived
from the soft-max distribution).

Per-patch decomposition (mean-pool admits this exactly):
    bag_pred = classifier(mean(F)) = mean(classifier(F_i))
    so per-patch score s_i = classifier(F_i) and slide_pred = mean(s_i).

The patch grid was produced with edge-anchored sliding-window tiling
(``src/data/preprocess.py``) and is therefore overlapping. This tool
re-derives the exact (px, py) pixel anchor of every patch by replaying the
same tiling logic on the raw image, then averages the per-patch scalar
score across every overlapping region pixel-by-pixel.

OD-filtered (rejected) patches are detected as missing entries in the bag's
``rc`` tensor and excluded from the heatmap. Optional ``--show_rejected_mask``
overlays a translucent gray mask over those regions.

────────────────────────────────────────────────────────────────────────────
Default behaviour: clean banded view (much less noisy than the full map)
────────────────────────────────────────────────────────────────────────────
By default the heatmap is filtered to the **same grade bin as the slide
prediction** so the figure is readable. With the slide predicted G_k, the
visible band is the half-open interval ``[k − 0.5, k + 0.5)``:

    Slide pred            Visible band
    1.21  → G1            [0.5, 1.5)
    1.78  → G2            [1.5, 2.5)
    2.40  → G2            [1.5, 2.5)
    2.81  → G3            [2.5, 3.5)

Pass ``--highlight_band 0`` to disable the filter and see the full map.

A small in-band sanity check is printed under the title: it reports the
mean of the surviving patches and whether that mean rounds to the same
grade as the slide call. When it does (``✓``), the visible patches are a
sufficient evidence subset for the model's decision; when it does not
(``✗``), the band view is purely cosmetic for that slide.

Usage:
    # Default (clean band view, easy to look at):
    python -m src.tools.reti_mean_pool_heatmap \\
        --checkpoint experiments/reti_mean_pool_uni2_.../best.pth \\
        --split test

    # Single-slide debug:
    python -m src.tools.reti_mean_pool_heatmap \\
        --checkpoint .../best.pth --slide "ET/ET1 G1/reti10"

    # Faithful (full, noisy) heatmap:
    python -m src.tools.reti_mean_pool_heatmap \\
        --checkpoint .../best.pth --highlight_band 0
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Ensure src/ is importable when run via -m or directly.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from models.mean_pool_mil import MeanPoolMIL  # noqa: E402
from train_grading_reti import (  # noqa: E402
    BACKBONE_CONFIG,
    CLASS_NAMES,
)


# ── constants ─────────────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 224
# These match the reti preprocessing pipeline used to produce
# data/processed_grading (see README and src/data/preprocess.py):
#   --stain reti --patch_size 224 --step_size 112
#   --crop_top 57 --crop_bottom 40 --use_od_filter --save_rejected
DEFAULT_STEP_SIZE = 112
DEFAULT_CROP_TOP = 57
DEFAULT_CROP_BOTTOM = 40
DEFAULT_OUTPUT_DIR = Path("results") / "reti_mean_pool_heatmaps"

RAW_DATA_DIRNAME = "raw"


# ── edge-anchored tiling (mirrors src/data/preprocess.py) ─────────────────


def compute_grid_positions(
    width: int,
    height: int,
    patch_size: int,
    step_size: int,
) -> Tuple[List[int], List[int]]:
    """Return (y_positions, x_positions) using the same edge-anchored tiling
    as ``src/data/preprocess.py:extract_patches``."""
    y_positions: List[int] = []
    y = 0
    while y + patch_size <= height:
        y_positions.append(y)
        y += step_size
    last_y = height - patch_size
    if last_y >= 0 and (not y_positions or y_positions[-1] != last_y):
        y_positions.append(last_y)

    x_positions: List[int] = []
    x = 0
    while x + patch_size <= width:
        x_positions.append(x)
        x += step_size
    last_x = width - patch_size
    if last_x >= 0 and (not x_positions or x_positions[-1] != last_x):
        x_positions.append(last_x)

    return y_positions, x_positions


def build_patch_grid(
    width: int,
    height: int,
    patch_size: int,
    step_size: int,
) -> List[Tuple[int, int, int, int]]:
    """Return a deduplicated list of (row, col, px, py) for the full grid."""
    y_positions, x_positions = compute_grid_positions(
        width, height, patch_size, step_size
    )
    grid: List[Tuple[int, int, int, int]] = []
    seen: set = set()
    for r, py in enumerate(y_positions):
        for c, px in enumerate(x_positions):
            if (px, py) in seen:
                continue
            seen.add((px, py))
            grid.append((r, c, px, py))
    return grid


# ── checkpoint / model loading ────────────────────────────────────────────


def load_mean_pool_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[MeanPoolMIL, Dict, str, str, int, str]:
    """Load a ``MeanPoolMIL`` checkpoint and return (model, ckpt, backbone,
    formulation, num_classes, model_type)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {}) or {}

    backbone = args.get("backbone") or ckpt.get("backbone")
    model_type = args.get("model_type") or ckpt.get("model_type", "mean_pool")
    formulation = args.get("formulation", "classification")
    if model_type != "mean_pool":
        raise ValueError(
            f"This tool only supports MeanPoolMIL checkpoints, got "
            f"model_type='{model_type}'."
        )
    if backbone is None:
        raise ValueError("Cannot infer 'backbone' from checkpoint args.")

    cfg = BACKBONE_CONFIG[backbone]
    num_classes = 1 if formulation == "regression" else 4

    model = MeanPoolMIL(
        vision_dim=cfg["dim"],
        num_classes=num_classes,
        dropout=args.get("dropout", 0.5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt, backbone, formulation, num_classes, model_type


# ── per-patch scoring ─────────────────────────────────────────────────────


@torch.inference_mode()
def per_patch_scalar(
    features: torch.Tensor,
    model: MeanPoolMIL,
    formulation: str,
) -> Tuple[np.ndarray, float]:
    """Compute a scalar score per patch and the resulting bag prediction."""
    classifier = model.classifier  # nn.Sequential(Dropout, Linear)
    logits = classifier(features)  # [N, num_classes]

    if formulation == "regression":
        per_patch = logits.view(-1).cpu().numpy().astype(np.float32)
        bag_logit = classifier(features.mean(dim=0, keepdim=True)).view(-1)
        bag_pred = float(bag_logit.item())
        return per_patch, bag_pred

    # classification: 4-class softmax → expected grade
    probs = F.softmax(logits, dim=-1)  # [N, 4]
    grades = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype)
    per_patch = (probs * grades).sum(dim=-1).cpu().numpy().astype(np.float32)

    bag_logits = classifier(features.mean(dim=0, keepdim=True)).view(-1)
    bag_probs = F.softmax(bag_logits, dim=-1)
    bag_pred = float((bag_probs * grades).sum().item())
    return per_patch, bag_pred


# ── heatmap construction ──────────────────────────────────────────────────


def aggregate_heatmap(
    grid: Sequence[Tuple[int, int, int, int]],
    rc_to_score: Dict[Tuple[int, int], float],
    image_size: Tuple[int, int],  # (W, H)
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average overlapping per-patch scores into a per-pixel heatmap.

    Returns:
        score_map: float32 [H, W], NaN where no valid patch covered the pixel.
        count_map: int32 [H, W], number of valid patches per pixel.
        rejected_mask: bool [H, W], True where only OD-rejected patches covered.
    """
    width, height = image_size
    sum_map = np.zeros((height, width), dtype=np.float64)
    count_map = np.zeros((height, width), dtype=np.int32)
    rejected_count = np.zeros((height, width), dtype=np.int32)

    for r, c, px, py in grid:
        score = rc_to_score.get((r, c))
        if score is None:
            rejected_count[py : py + patch_size, px : px + patch_size] += 1
            continue
        sum_map[py : py + patch_size, px : px + patch_size] += float(score)
        count_map[py : py + patch_size, px : px + patch_size] += 1

    score_map = np.full_like(sum_map, np.nan, dtype=np.float32)
    valid = count_map > 0
    score_map[valid] = (sum_map[valid] / count_map[valid]).astype(np.float32)

    rejected_only_mask = (count_map == 0) & (rejected_count > 0)
    return score_map, count_map.astype(np.int32), rejected_only_mask


# ── plotting ──────────────────────────────────────────────────────────────


def render_heatmap_figure(
    raw_image: np.ndarray,
    score_map: np.ndarray,
    rejected_mask: np.ndarray,
    save_path: Path,
    *,
    bag_pred: float,
    true_grade: str,
    pred_grade: str,
    true_label: int,
    slide_id: str,
    formulation: str,
    show_rejected: bool,
    highlight_band: float = 0.5,
    highlight_center: str = "pred_grade",
    in_band_n: int = 0,
    in_band_mean: float = float("nan"),
    in_band_grade: str = "",
    in_band_recovers: str = "",
    vmin: float = 0.0,
    vmax: float = 3.0,
) -> None:
    """Render a 3-panel figure: raw / heatmap-only / overlay.

    With ``highlight_band > 0`` the heatmap is band-filtered: every pixel
    whose averaged score lies outside the half-open interval
    ``[ref − band, ref + band)`` is NaN-masked, leaving only the patches
    consistent with the slide call. Heatmap values themselves are unchanged
    — only their visibility.
    """
    # ── Optional band-filter masking (half-open upper bound) ──
    title_suffix = ""
    if highlight_band > 0.0:
        if highlight_center == "gt":
            ref = float(true_label)
            ref_label = f"GT={true_grade}"
        elif highlight_center == "bag_pred":
            ref = float(bag_pred)
            ref_label = f"pred={ref:.2f}"
        else:  # "pred_grade" — round to grade bin
            grade_int = int(round(max(0.0, min(3.0, float(bag_pred)))))
            ref = float(grade_int)
            ref_label = f"pred-grade={CLASS_NAMES[grade_int]}"
        score_map = score_map.copy()
        # Half-open interval: [ref - band, ref + band)
        within = (score_map >= ref - highlight_band) & (score_map < ref + highlight_band)
        score_map[~within & np.isfinite(score_map)] = np.nan
        title_suffix = (
            f"  ·  score ∈ [{ref - highlight_band:.2f}, "
            f"{ref + highlight_band:.2f})  ({ref_label})"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

    # 1. Raw image
    axes[0].imshow(raw_image)
    axes[0].set_title("Raw Reticulin", fontsize=12)
    axes[0].axis("off")

    # 2. Heatmap (NaN-safe) — what the model predicts within the band
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(score_map)
    im = axes[1].imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Per-Patch Score" + title_suffix, fontsize=12)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Predicted grade" if formulation == "regression" else "Expected grade")

    # 3. Overlay
    axes[2].imshow(raw_image)
    overlay = axes[2].imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.55)
    if show_rejected and rejected_mask.any():
        rej = np.zeros((*rejected_mask.shape, 4), dtype=np.float32)
        rej[rejected_mask] = (0.5, 0.5, 0.5, 0.45)
        axes[2].imshow(rej, interpolation="nearest")
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")
    fig.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{slide_id}  —  GT: {true_grade}  |  Pred: {pred_grade} ({bag_pred:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    if highlight_band > 0.0 and in_band_n > 0:
        recover_tag = "✓ same grade" if in_band_recovers == "yes" else "✗ different grade"
        fig.text(
            0.5,
            0.93,
            f"In-band: {in_band_n} patches, mean={in_band_mean:.2f} → "
            f"{in_band_grade}  ({recover_tag} as slide pred {pred_grade})",
            ha="center", va="bottom", fontsize=11, color="#444444",
        )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_band_sweep_figure(
    raw_image: np.ndarray,
    score_map_full: np.ndarray,
    save_path: Path,
    *,
    bag_pred: float,
    true_grade: str,
    pred_grade: str,
    true_label: int,
    per_patch_scores: np.ndarray,
    slide_id: str,
    formulation: str,
    band_widths: Sequence[float],
    highlight_center: str = "pred_grade",
    vmin: float = 0.0,
    vmax: float = 3.0,
) -> None:
    """Render a single side-by-side sensitivity figure for several band widths.

    Layout: [Raw] [Full heatmap] [Band w₁] [Band w₂] ... — one row.
    The full heatmap is shown as the leftmost band-filtered panel for
    contrast; sufficiency information for each band is printed under each
    sub-title.
    """
    n_panels = 2 + len(band_widths)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.2 * n_panels, 5.4), facecolor="white"
    )

    # Panel 0: raw
    axes[0].imshow(raw_image)
    axes[0].set_title("Raw Reticulin", fontsize=11)
    axes[0].axis("off")

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="white")

    # Panel 1: full heatmap (no band)
    masked_full = np.ma.masked_invalid(score_map_full)
    im = axes[1].imshow(masked_full, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Full per-patch heatmap", fontsize=11)
    axes[1].axis("off")

    # Determine reference value for masking
    if highlight_center == "gt":
        ref = float(true_label)
        ref_label = f"GT={true_grade}"
    elif highlight_center == "bag_pred":
        ref = float(bag_pred)
        ref_label = f"pred={ref:.2f}"
    else:
        grade_int = int(round(max(0.0, min(3.0, float(bag_pred)))))
        ref = float(grade_int)
        ref_label = f"pred-grade={CLASS_NAMES[grade_int]}"

    # Panels 2..: band-filtered views
    for i, band in enumerate(band_widths):
        ax = axes[2 + i]
        score_map = score_map_full.copy()
        within = (score_map >= ref - band) & (score_map < ref + band)
        score_map[~within & np.isfinite(score_map)] = np.nan
        ax.imshow(np.ma.masked_invalid(score_map),
                  cmap=cmap, vmin=vmin, vmax=vmax)

        # Sufficiency check for this band
        in_band_scores = [
            s for s in per_patch_scores
            if (s >= ref - band) and (s < ref + band)
        ]
        n_in = len(in_band_scores)
        if n_in > 0:
            mean_in = float(np.mean(in_band_scores))
            grade_in = grade_label_to_name(mean_in, formulation)
            ok = grade_in == pred_grade
            tag = "✓" if ok else "✗"
            sub = (
                f"score ∈ [{ref - band:.2f}, {ref + band:.2f})\n"
                f"n={n_in}, mean={mean_in:.2f}→{grade_in} {tag}"
            )
        else:
            sub = (
                f"score ∈ [{ref - band:.2f}, {ref + band:.2f})\n"
                f"n=0 (no patches in band)"
            )
        ax.set_title(f"Band ±{band:.2f}\n{sub}", fontsize=10)
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(
        f"{slide_id}  —  GT: {true_grade}  |  Pred: {pred_grade} "
        f"({bag_pred:.2f})  ·  centre: {ref_label}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── slide-level pipeline ──────────────────────────────────────────────────


def grade_label_to_name(label, formulation: str) -> str:
    """Convert a (possibly continuous) grade into 'G0'..'G3'."""
    if formulation == "regression":
        idx = int(round(max(0.0, min(3.0, float(label)))))
    else:
        idx = int(label)
    return CLASS_NAMES[idx]


def find_raw_image(
    data_root: Path,
    subtype: str,
    patient: str,
    slide_id: str,
) -> Optional[Path]:
    """Locate the raw .tif/.tiff/.png for the given slide."""
    base = data_root / RAW_DATA_DIRNAME / subtype / patient
    if not base.exists():
        return None
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        candidate = base / f"{slide_id}{ext}"
        if candidate.exists():
            return candidate
    for f in base.iterdir():
        if f.is_file() and f.stem.lower() == slide_id.lower():
            return f
    return None


def process_slide(
    pt_path: Path,
    label: int,
    model: MeanPoolMIL,
    *,
    data_root: Path,
    output_dir: Path,
    patch_size: int,
    step_size: int,
    crop_top: int,
    crop_bottom: int,
    formulation: str,
    show_rejected: bool,
    highlight_band: float = 0.5,
    highlight_center: str = "pred_grade",
    band_widths: Optional[Sequence[float]] = None,
    csv_writer=None,
    device: torch.device = torch.device("cpu"),
) -> Optional[Dict]:
    """Process a single slide and emit a heatmap PNG + CSV rows."""
    patient = pt_path.parent.name
    subtype = pt_path.parent.parent.name
    slide_id = pt_path.stem

    raw_path = find_raw_image(data_root, subtype, patient, slide_id)
    if raw_path is None:
        print(f"  ⚠ Raw image not found for {subtype}/{patient}/{slide_id} — skipping.")
        return None

    data = torch.load(pt_path, map_location=device, weights_only=False)
    if not isinstance(data, dict) or "feats" not in data or "rc" not in data:
        print(f"  ⚠ {pt_path} missing 'feats'/'rc' — skipping.")
        return None
    features: torch.Tensor = data["feats"].to(device)
    rc: torch.Tensor = data["rc"].cpu().to(torch.int64)

    per_patch, bag_pred = per_patch_scalar(features, model, formulation)
    if len(per_patch) != rc.shape[0]:
        print(
            f"  ⚠ feats/rc length mismatch in {pt_path} "
            f"({len(per_patch)} vs {rc.shape[0]}) — skipping."
        )
        return None

    rc_to_score: Dict[Tuple[int, int], float] = {
        (int(rc[i, 0]), int(rc[i, 1])): float(per_patch[i])
        for i in range(rc.shape[0])
    }

    # Load + crop raw image to match what preprocess.py saw
    raw_pil = Image.open(raw_path).convert("RGB")
    w0, h0 = raw_pil.size
    if crop_top > 0 or crop_bottom > 0:
        raw_pil = raw_pil.crop((0, crop_top, w0, h0 - crop_bottom))
    raw_arr = np.array(raw_pil)
    H, W = raw_arr.shape[:2]

    # Re-derive full grid (valid + OD-rejected)
    grid = build_patch_grid(W, H, patch_size, step_size)
    n_total = len(grid)
    n_valid = sum(1 for r, c, *_ in grid if (r, c) in rc_to_score)
    n_rejected_od = n_total - n_valid

    score_map, count_map, rejected_only_mask = aggregate_heatmap(
        grid, rc_to_score, (W, H), patch_size,
    )

    true_grade = grade_label_to_name(label, "classification")
    pred_grade = grade_label_to_name(bag_pred, formulation)

    # ── In-band sanity check ─────────────────────────────────────────────
    # Tests whether the patches that survive the band ALONE would still
    # produce the same slide-level grade. Uses the half-open interval
    # [ref − band, ref + band), matching the visual masking.
    in_band_n = 0
    in_band_mean = float("nan")
    in_band_grade = ""
    in_band_recovers = ""
    if highlight_band > 0.0:
        if highlight_center == "gt":
            ref = float(label)
        elif highlight_center == "bag_pred":
            ref = float(bag_pred)
        else:  # pred_grade
            ref = float(int(round(max(0.0, min(3.0, float(bag_pred))))))
        in_band_scores = [
            s for s in per_patch
            if (s >= ref - highlight_band) and (s < ref + highlight_band)
        ]
        in_band_n = len(in_band_scores)
        if in_band_n > 0:
            in_band_mean = float(np.mean(in_band_scores))
            in_band_grade = grade_label_to_name(in_band_mean, formulation)
            in_band_recovers = "yes" if in_band_grade == pred_grade else "no"

    out_png = output_dir / subtype / patient / f"{slide_id}_heatmap.png"
    render_heatmap_figure(
        raw_image=raw_arr,
        score_map=score_map,
        rejected_mask=rejected_only_mask,
        save_path=out_png,
        bag_pred=bag_pred,
        true_grade=true_grade,
        pred_grade=pred_grade,
        true_label=int(label),
        slide_id=f"{subtype}/{patient}/{slide_id}",
        formulation=formulation,
        show_rejected=show_rejected,
        highlight_band=highlight_band,
        highlight_center=highlight_center,
        in_band_n=in_band_n,
        in_band_mean=in_band_mean,
        in_band_grade=in_band_grade,
        in_band_recovers=in_band_recovers,
    )

    # Optional band-width sensitivity sweep figure (Fig S1 in the paper).
    if band_widths:
        sweep_png = (
            output_dir / subtype / patient / f"{slide_id}_band_sweep.png"
        )
        render_band_sweep_figure(
            raw_image=raw_arr,
            score_map_full=score_map,
            save_path=sweep_png,
            bag_pred=bag_pred,
            true_grade=true_grade,
            pred_grade=pred_grade,
            true_label=int(label),
            per_patch_scores=per_patch,
            slide_id=f"{subtype}/{patient}/{slide_id}",
            formulation=formulation,
            band_widths=band_widths,
            highlight_center=highlight_center,
        )

    if csv_writer is not None:
        for (r, c, px, py) in grid:
            score = rc_to_score.get((r, c))
            csv_writer.writerow(
                [
                    subtype,
                    patient,
                    slide_id,
                    r,
                    c,
                    px,
                    py,
                    "" if score is None else f"{score:.6f}",
                    "rejected_od" if score is None else "valid",
                ]
            )

    band_msg = ""
    if highlight_band > 0.0:
        band_msg = (
            f" | in-band={in_band_n} mean={in_band_mean:.2f}→{in_band_grade}"
            f" ({'recovers' if in_band_recovers == 'yes' else 'differs'})"
        )
    print(
        f"  ✅ {subtype}/{patient}/{slide_id} | "
        f"grid={n_total} valid={n_valid} rejected_od={n_rejected_od} | "
        f"GT={true_grade} pred={pred_grade} ({bag_pred:.2f}){band_msg} → {out_png}"
    )

    return {
        "subtype": subtype,
        "patient": patient,
        "slide_id": slide_id,
        "true_label": int(label),
        "true_grade": true_grade,
        "pred_scalar": bag_pred,
        "pred_grade": pred_grade,
        "n_valid": n_valid,
        "n_rejected_od": n_rejected_od,
        "highlight_band": highlight_band,
        "highlight_center": highlight_center,
        "in_band_n": in_band_n,
        "in_band_mean": in_band_mean,
        "in_band_grade": in_band_grade,
        "in_band_recovers": in_band_recovers,
    }


# ── argument parsing ──────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Per-patch prediction-score heatmaps for Reticulin Fibrosis "
            "Grading using a trained MeanPoolMIL checkpoint. By default the "
            "heatmap is band-filtered to the slide's grade bin for "
            "readability; pass --highlight_band 0 to see the full noisy map."
        )
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained MeanPoolMIL .pth checkpoint.")
    p.add_argument("--data_root", type=str, default="data",
                   help="Root data directory (default: data).")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test", "all"],
                   help="Which split to visualise (default: test).")
    p.add_argument("--slide", type=str, default=None,
                   help="Optional explicit slide spec '<subtype>/<patient>/<slide_id>' "
                        "(overrides --split).")
    p.add_argument("--subtype", type=str, default=None, choices=["ET", "PV", "PMF"],
                   help="Only process slides belonging to this subtype.")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
                   help=f"Patch size (default: {DEFAULT_PATCH_SIZE}).")
    p.add_argument("--step_size", type=int, default=DEFAULT_STEP_SIZE,
                   help=f"Sliding-window step size (default: {DEFAULT_STEP_SIZE}). "
                        "Must match the value used at preprocessing time.")
    p.add_argument("--crop_top", type=int, default=DEFAULT_CROP_TOP,
                   help=f"Pixels cropped from top (default: {DEFAULT_CROP_TOP}).")
    p.add_argument("--crop_bottom", type=int, default=DEFAULT_CROP_BOTTOM,
                   help=f"Pixels cropped from bottom (default: {DEFAULT_CROP_BOTTOM}).")
    p.add_argument("--show_rejected_mask", action="store_true",
                   help="Overlay a gray mask on OD-rejected regions of the heatmap.")
    p.add_argument("--highlight_band", type=float, default=0.5,
                   help=("Half-width of the visible score band around the "
                         "highlight reference. Default 0.5: with a slide pred "
                         "of G2 only patches scoring in [1.5, 2.5) are shown. "
                         "Pass 0 to disable and show the full heatmap."))
    p.add_argument("--highlight_center", type=str, default="pred_grade",
                   choices=["pred_grade", "bag_pred", "gt"],
                   help=("Reference value used by --highlight_band. "
                         "'pred_grade' (default): round slide pred to grade bin "
                         "(G2 → 2.0). 'bag_pred': use the continuous slide pred "
                         "unchanged. 'gt': use the slide GT label (debug only — "
                         "slide GT ≠ patch GT)."))
    p.add_argument("--band_widths", type=str, default=None,
                   help=("Optional comma-separated list of band widths, e.g. "
                         "'0.25,0.5,1.0'. When set, an extra "
                         "<slide>_band_sweep.png is emitted per slide showing "
                         "the heatmap at each width side-by-side (Fig S1 in "
                         "the paper)."))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"),
                   help="Inference device.")
    return p.parse_args()


# ── slide selection helpers ───────────────────────────────────────────────


def select_slide_indices(
    dataset: GradingBagDatasetFull,
    ckpt: Dict,
    split: str,
) -> List[int]:
    if split == "all":
        return list(range(len(dataset)))
    key = f"{split}_idx"
    if key in ckpt:
        return list(ckpt[key])
    raise ValueError(
        f"Checkpoint does not contain '{key}'. Use --split all or retrain "
        f"with split-index logging."
    )


def find_slide_index(
    dataset: GradingBagDatasetFull,
    spec: str,
) -> int:
    """Locate the dataset index for '<subtype>/<patient>/<slide_id>'."""
    target = spec.strip().strip("/")
    for idx in range(len(dataset)):
        pt_path = dataset.get_slide_path(idx)
        key = f"{pt_path.parent.parent.name}/{pt_path.parent.name}/{pt_path.stem}"
        if key == target:
            return idx
    raise ValueError(f"Slide '{spec}' not found in dataset.")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("Reti Mean-Pool — Per-Patch Heatmap")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device:     {device}")

    model, ckpt, backbone, formulation, num_classes, _ = load_mean_pool_checkpoint(
        ckpt_path, device
    )
    cfg = BACKBONE_CONFIG[backbone]
    features_dir = Path(args.data_root) / cfg["feature_dir"]

    print(f"  Backbone:    {cfg['display_name']}  (dim={cfg['dim']})")
    print(f"  Formulation: {formulation}  (num_classes={num_classes})")
    print(f"  Features:    {features_dir}")
    print(f"  Highlight:   band={args.highlight_band} center={args.highlight_center}")

    dataset = GradingBagDatasetFull(features_dir)
    print(f"  Dataset bags: {len(dataset)}")

    band_widths: Optional[List[float]] = None
    if args.band_widths:
        try:
            band_widths = [
                float(x.strip()) for x in args.band_widths.split(",") if x.strip()
            ]
        except ValueError:
            raise ValueError(
                f"--band_widths must be comma-separated floats, got "
                f"{args.band_widths!r}"
            )
        print(f"  Band sweep:   widths={band_widths}")

    if args.slide is not None:
        indices = [find_slide_index(dataset, args.slide)]
        split_tag = "single"
    else:
        indices = select_slide_indices(dataset, ckpt, args.split)
        split_tag = args.split

    if args.subtype is not None:
        indices = [
            i for i in indices
            if dataset.get_slide_path(i).parent.parent.name == args.subtype
        ]

    print(f"  Slides to process: {len(indices)}  ({split_tag})")

    exp_name = ckpt_path.parent.name
    out_root = Path(args.output_dir) / exp_name / split_tag
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "per_patch_scores.csv"
    summary_path = out_root / "slide_summary.csv"
    print(f"  Output: {out_root}")
    print("=" * 60)

    summaries: List[Dict] = []
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "subtype", "patient", "slide_id",
                "row", "col", "px", "py",
                "score", "status",
            ]
        )

        grouped: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            p = dataset.get_slide_path(idx)
            grouped[f"{p.parent.parent.name}/{p.parent.name}"].append(idx)

        for patient_key in tqdm(sorted(grouped.keys()), desc="Patients"):
            for idx in grouped[patient_key]:
                pt_path = dataset.get_slide_path(idx)
                _, label = dataset.samples[idx]
                summary = process_slide(
                    pt_path=pt_path,
                    label=label,
                    model=model,
                    data_root=Path(args.data_root),
                    output_dir=out_root,
                    patch_size=args.patch_size,
                    step_size=args.step_size,
                    crop_top=args.crop_top,
                    crop_bottom=args.crop_bottom,
                    formulation=formulation,
                    show_rejected=args.show_rejected_mask,
                    highlight_band=args.highlight_band,
                    highlight_center=args.highlight_center,
                    band_widths=band_widths,
                    csv_writer=writer,
                    device=device,
                )
                if summary is not None:
                    summaries.append(summary)

    if summaries:
        with open(summary_path, "w", newline="") as fh:
            dwriter = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            dwriter.writeheader()
            for row in summaries:
                dwriter.writerow(row)

    print(f"\n{'=' * 60}")
    print(f"Done. Heatmaps : {out_root}")
    print(f"     Per-patch : {csv_path}")
    print(f"     Summary   : {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()







