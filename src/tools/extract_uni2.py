"""
UNI2-h Feature Extraction — Image-Level Bag Pipeline.

Extracts frozen UNI2-h [CLS] embeddings from pre-patched image tiles,
grouped by source image ID. Each original image produces one .pt file
containing a stacked tensor of shape [N_patches, 1536].

Input:  data/processed_subtype/{Class}/{PatientID}/{ImgID}_r{Row}c{Col}.png
Output: data/features_uni2/{Class}/{PatientID}/{ImgID}.pt

Usage:
    python -m src.tools.extract_uni2 \
        --data_dir data/processed_subtype \
        --output_dir data/features_uni2 \
        --batch_size 64 \
        --device cuda

    # Dry-run (validates grouping without extracting):
    python -m src.tools.extract_uni2 --dry_run
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Ensure src/ is on sys.path when running directly (e.g., python src/tools/extract_uni2.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from core.config import hf_login

# =============================================================================
# Constants
# =============================================================================
DEFAULT_DATA_DIR = "data/processed_subtype"
DEFAULT_OUTPUT_DIR = "data/features_uni2"
FEATURE_DIM = 1536
CLASSES = ["ET", "PV", "PMF"]


# =============================================================================
# Patch Grouping Utilities
# =============================================================================


def parse_patch_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse a patch filename into (image_id, row, col).

    Example: "3_r2c5.png" → ("3", 2, 5)
    """
    match = re.match(r"^(\d+)_r(\d+)c(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Unexpected patch filename format: {filename}")
    img_id = match.group(1)
    row = int(match.group(2))
    col = int(match.group(3))
    return img_id, row, col


def group_patches_by_image(
    patient_dir: Path,
) -> Dict[str, List[Tuple[Path, int, int]]]:
    """
    Group all patches in a patient directory by source image ID.

    Returns:
        Dict mapping image_id → list of (patch_path, row, col),
        sorted by (row, col) within each group for spatial ordering.
    """
    groups: Dict[str, List[Tuple[Path, int, int]]] = defaultdict(list)

    for patch_path in patient_dir.iterdir():
        if not patch_path.suffix == ".png":
            continue
        try:
            img_id, row, col = parse_patch_filename(patch_path.name)
            groups[img_id].append((patch_path, row, col))
        except ValueError:
            print(f"  ⚠ Skipping unrecognised file: {patch_path.name}")

    # Sort each group by (row, col) for spatial ordering
    for img_id in groups:
        groups[img_id].sort(key=lambda x: (x[1], x[2]))

    return dict(groups)


# =============================================================================
# Patch-Level Metric Helpers (matches analyze_shortcuts.py CV2 logic)
# =============================================================================


def tissue_fraction_from_rgb(
    patch_rgb_uint8: np.ndarray, tissue_thr: float = 0.05
) -> float:
    img = np.clip(patch_rgb_uint8.astype(np.float32), 1.0, 255.0)
    od = -np.log10(img / 255.0)
    mean_od = od.mean(axis=2)
    return float((mean_od > tissue_thr).mean())


def space_bg_fractions_from_rgb(patch_rgb_uint8: np.ndarray) -> Tuple[float, float]:
    v_thr, s_thr = 0.90, 0.15
    min_comp_area = 80
    H, W, _ = patch_rgb_uint8.shape
    max_comp_area = int(0.35 * H * W)

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

    for lab in range(1, n):
        area = stats[lab, cv2.CC_STAT_AREA]
        comp = labels == lab

        if (comp & border).any():
            if area >= min_comp_area:
                outside_white |= comp
        else:
            if area < min_comp_area or area > max_comp_area:
                continue
            internal_white |= comp

    bg_blank_frac = float(outside_white.mean())
    space_frac = float(internal_white.mean())
    return space_frac, bg_blank_frac


def tissue_mask_from_rgb(patch_rgb_uint8: np.ndarray, tissue_thr: float = 0.05) -> np.ndarray:
    """Returns a boolean mask of tissue pixels based on Optical Density."""
    img = np.clip(patch_rgb_uint8.astype(np.float32), 1.0, 255.0)
    od = -np.log10(img / 255.0)
    mean_od = od.mean(axis=2)
    return mean_od > tissue_thr


def rgb_uint8_to_float01(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = rgb_uint8.astype(np.float32) / 255.0
    return np.clip(rgb, 1e-6, 1.0)


def rgb_to_od(rgb_float: np.ndarray) -> np.ndarray:
    return -np.log(rgb_float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def soft_gt(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    scale = max(scale, 1e-6)
    return sigmoid((x - center) / scale)


def soft_lt(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    scale = max(scale, 1e-6)
    return sigmoid((center - x) / scale)


def compute_patch_color_metrics(
    patch_rgb: np.ndarray,
    core_tissue_mask: np.ndarray,
    min_core_pixels: int = 100,
) -> Tuple[float, float]:
    """
    Computes soft fractions for purple/dark (nuclei-rich) and warm/pink low-info proxy.
    Returns: (cellular_purple_frac, pink_lowinfo_frac)
    """
    if core_tissue_mask.sum() < min_core_pixels:
        return 0.0, 0.0

    rgb = rgb_uint8_to_float01(patch_rgb)
    # cv2.cvtColor on float32 RGB returns standard CIELAB scale (L: 0-100, a/b: ~-127 to 127)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    od = rgb_to_od(rgb)
    od_sum = od.sum(axis=-1)

    a = lab[..., 1][core_tissue_mask]
    b = lab[..., 2][core_tissue_mask]
    ab = a - b
    chroma = np.sqrt(a * a + b * b)
    od_t = od_sum[core_tissue_mask]

    # ---- cellular_purple_frac ----
    purple_score = soft_gt(ab, 8.0, 4.0)
    dark_score = soft_gt(od_t, 1.8, 0.35)
    chroma_score = soft_gt(chroma, 12.0, 5.0)

    cellular_purple_pixel = purple_score * dark_score * chroma_score
    cellular_purple_frac = float(np.mean(cellular_purple_pixel))

    # ---- pink_lowinfo_frac ----
    warm_score = soft_gt(a, 10.0, 5.0) * soft_gt(b, 5.0, 5.0)
    nonpurple_score = soft_lt(ab, 6.0, 4.0)
    lightmid_score = soft_lt(od_t, 1.9, 0.35)

    pink_lowinfo_pixel = warm_score * nonpurple_score * lightmid_score
    raw_pink_frac = float(np.mean(pink_lowinfo_pixel))

    # Soft suppression: prevent penalizing mixed/useful cellular patches
    pink_lowinfo_frac = raw_pink_frac * (1.0 - cellular_purple_frac)

    return cellular_purple_frac, pink_lowinfo_frac


# =============================================================================
# Patch Dataset (for batched feature extraction)
# =============================================================================


class PatchDataset(Dataset):
    """
    Simple dataset for loading patches from a single image group.

    Patches are already 224×224; no resizing is performed.
    The timm transform handles normalization and tensor conversion.

    Args:
        patch_paths: List of paths to patch images.
        transform:   timm eval transform.
    """

    def __init__(self, patch_paths: List[Path], transform) -> None:
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        patch_path = self.patch_paths[idx]
        img = Image.open(patch_path).convert("RGB")
        tensor = self.transform(img)

        # 1. Parse row and col
        match = re.search(r"_r(\d+)c(\d+)\.png$", patch_path.name)
        row = int(match.group(1)) if match else -1
        col = int(match.group(2)) if match else -1

        # 2. Calculate metrics
        img_array = np.array(img, dtype=np.uint8)
        tissue_frac = tissue_fraction_from_rgb(img_array)
        internal_white_frac, border_white_frac = space_bg_fractions_from_rgb(img_array)

        # 3. Pre-compute the nuisance score for Logit Bias
        eps = 1e-6
        space_to_tissue_ratio = float(internal_white_frac / (tissue_frac + eps))

        # sigmoid equivalent: 1 / (1 + exp(-x))
        bad_border = 1.0 / (1.0 + np.exp(-((border_white_frac - 0.2) / 0.15)))
        bad_internal = 1.0 / (1.0 + np.exp(-((space_to_tissue_ratio - 0.8) / 0.15)))
        nuisance_score = float(np.clip(bad_border + bad_internal, 0.0, 1.0))

        # 4. Compute Color Metrics (Soft LAB & OD gating)
        od_tissue_mask = tissue_mask_from_rgb(img_array)
        # Create strict core mask by explicitly excluding bright white/empty spaces
        absolute_white_mask = (img_array > 220).all(axis=-1)
        core_tissue_mask = od_tissue_mask & (~absolute_white_mask)

        cellular_purple_frac, pink_lowinfo_frac = compute_patch_color_metrics(
            img_array, core_tissue_mask=core_tissue_mask
        )

        metrics = {
            "border_white_frac": float(border_white_frac),
            "tissue_frac": float(tissue_frac),
            "internal_white_frac": float(internal_white_frac),
            "nuisance_score": float(nuisance_score),
            "space_to_tissue_ratio": space_to_tissue_ratio,
            "cellular_purple_frac": cellular_purple_frac,
            "pink_lowinfo_frac": pink_lowinfo_frac,
            "row": int(row),
            "col": int(col),
        }
        return tensor, metrics


# =============================================================================
# Feature Extraction
# =============================================================================


@torch.inference_mode()
def extract_features_for_image(
    patch_paths: List[Path],
    model: torch.nn.Module,
    transform,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract UNI2-h [CLS] features for all patches in one image group.

    Args:
        patch_paths: Ordered list of patch file paths.
        model:       Frozen UNI2-h model.
        transform:   timm eval transform.
        batch_size:  Batch size for inference.
        device:      Device to run inference on.

    Returns:
        Tuple of (features [N_patches, 1536], metrics dict, rc [N, 2] int32).
    """
    dataset = PatchDataset(patch_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    use_amp = device.type == "cuda"
    all_features = []
    all_metrics = {
        "border_white_frac": [],
        "tissue_frac": [],
        "internal_white_frac": [],
        "nuisance_score": [],
        "space_to_tissue_ratio": [],
        "cellular_purple_frac": [],
        "pink_lowinfo_frac": [],
        "row": [],
        "col": [],
    }

    for batch_tensor, batch_metrics in loader:
        batch_tensor = batch_tensor.to(device)
        with torch.autocast(device.type, torch.float16, enabled=use_amp):
            output = model(batch_tensor)  # [B, 1536]

        # UNI2-h returns [CLS] token directly as [B, 1536]
        # Move to float32 on CPU to avoid half-precision save issues
        all_features.append(output.float().cpu())
        for key in all_metrics:
            all_metrics[key].append(batch_metrics[key].cpu())

    concatenated_features = torch.cat(all_features, dim=0)  # [N, 1536]

    # Float metrics
    concatenated_metrics = {
        "border_white_frac": torch.cat(all_metrics["border_white_frac"], dim=0).float(),
        "tissue_frac": torch.cat(all_metrics["tissue_frac"], dim=0).float(),
        "internal_white_frac": torch.cat(all_metrics["internal_white_frac"], dim=0).float(),
        "nuisance_score": torch.cat(all_metrics["nuisance_score"], dim=0).float(),
        "space_to_tissue_ratio": torch.cat(all_metrics["space_to_tissue_ratio"], dim=0).float(),
        "cellular_purple_frac": torch.cat(all_metrics["cellular_purple_frac"], dim=0).float(),
        "pink_lowinfo_frac": torch.cat(all_metrics["pink_lowinfo_frac"], dim=0).float(),
    }

    # Integer coordinates tensor [N, 2]
    rc_tensor = torch.stack(
        [
            torch.cat(all_metrics["row"], dim=0),
            torch.cat(all_metrics["col"], dim=0),
        ],
        dim=1,
    ).to(torch.int32)

    return concatenated_features, concatenated_metrics, rc_tensor


# =============================================================================
# Main Pipeline
# =============================================================================


def run_extraction(args: argparse.Namespace) -> None:
    """Main feature extraction pipeline."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Discover all patients and their image groups
    # ------------------------------------------------------------------
    extraction_plan: List[Tuple[str, str, Dict[str, List[Tuple[Path, int, int]]]]] = []
    # Each entry: (class_name, patient_folder_name, image_groups)

    total_images = 0
    total_patches = 0

    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠ Class directory not found: {class_dir}")
            continue

        patient_dirs = sorted(
            [d for d in class_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        for patient_dir in patient_dirs:
            image_groups = group_patches_by_image(patient_dir)
            if not image_groups:
                print(f"  ⚠ No patches found in {patient_dir.name}")
                continue

            extraction_plan.append((class_name, patient_dir.name, image_groups))
            total_images += len(image_groups)
            total_patches += sum(len(v) for v in image_groups.values())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("UNI2-h Feature Extraction Plan")
    print("=" * 60)
    print(f"  {'Data dir':<14}: {data_dir}")
    print(f"  {'Output dir':<14}: {output_dir}")
    print(f"  {'Patients':<14}: {len(extraction_plan)}")
    print(f"  {'Total images':<14}: {total_images}")
    print(f"  {'Total patches':<14}: {total_patches}")
    print(f"  {'Feature dim':<14}: {FEATURE_DIM}")
    print(f"  {'Device':<14}: {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Dry-run mode: print plan and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n🔍 DRY RUN — Printing extraction plan:\n")
        for class_name, patient_name, groups in extraction_plan:
            print(f"  📁 {class_name}/{patient_name}")
            sorted_img_ids = sorted(groups.keys(), key=lambda x: int(x))
            for img_id in sorted_img_ids:
                patches = groups[img_id]
                out_path = output_dir / class_name / patient_name / f"{img_id}.pt"
                print(
                    f"      Image {img_id:>3}: {len(patches):4d} patches → {out_path}"
                )
        print(f"\n✅ Dry run complete. {total_images} .pt files would be created.")
        return

    # ------------------------------------------------------------------
    # Load UNI2-h
    # ------------------------------------------------------------------
    print("\nLoading UNI2-h...")
    hf_login()
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model = model.to(device)
    model.eval()
    print(f"✅ UNI2-h loaded (feature_dim={FEATURE_DIM}).\n")

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------
    processed = 0
    skipped = 0

    for class_name, patient_name, groups in tqdm(
        extraction_plan, desc="Patients", unit="patient"
    ):
        # Mirror input structure: output_dir/{Class}/{PatientID}/
        patient_output_dir = output_dir / class_name / patient_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        sorted_img_ids = sorted(groups.keys(), key=lambda x: int(x))

        for img_id in tqdm(
            sorted_img_ids,
            desc=f"  {patient_name}",
            leave=False,
            unit="img",
        ):
            out_path = patient_output_dir / f"{img_id}.pt"

            # Skip if already extracted
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            patches = groups[img_id]
            patch_paths = [p[0] for p in patches]  # Extract only paths

            features, metrics, rc_tensor = extract_features_for_image(
                patch_paths=patch_paths,
                model=model,
                transform=transform,
                batch_size=args.batch_size,
                device=device,
            )

            # Convert paths to string for saving
            patch_paths_str = [str(p) for p in patch_paths]

            # Save the final standardized dictionary
            torch.save(
                {
                    "feats": features,
                    "metrics": metrics,
                    "rc": rc_tensor,
                    "patch_paths": patch_paths_str,
                },
                out_path,
            )
            processed += 1

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("UNI2-h Feature Extraction Complete")
    print(f"{'=' * 60}")
    print(f"  Extracted : {processed} image bags")
    print(f"  Skipped   : {skipped} (already exist)")
    print(f"  Output    : {output_dir}")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UNI2-h [CLS] features grouped by source image."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Input directory with {Class}/{PatientID}/ structure.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for .pt feature files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for patch inference.",
    )
    # Auto-detect best available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device,
        help="Device for inference (cuda, mps, or cpu).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print extraction plan without running inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
