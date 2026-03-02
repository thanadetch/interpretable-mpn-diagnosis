"""
Sorted Grid Gallery — ViT × MIL Attention Heatmaps.

For a given source image, generates a single grid figure with ALL patches
sorted by MIL attention (highest first). Each cell shows the raw patch
overlaid with ViT CLS attention (COLORMAP_TURBO, dynamic alpha).

Output: results/grid_heatmaps/{Patient}/{ImageID}_grid.png

Usage:
    python src/visualize_heatmap.py \
        --mil_checkpoint experiments/simple_titan_.../best_simple_titan.pth \
        --patient_dir "data/processed_subtype/PV/PV1 G2" \
        --image_id 1

    # All images for a patient:
    python src/visualize_heatmap.py \
        --mil_checkpoint experiments/simple_titan_.../best_simple_titan.pth \
        --patient_dir "data/processed_subtype/PV/PV1 G2"
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from core.config import CLASS_MAP, CLASS_MAP_INV, RESULTS_DIR, hf_login
from models.hybrid_mil import HybridMIL
from models.simple_mil import SimpleGatedMIL

# =============================================================================
# Constants
# =============================================================================

BACKBONE_CONFIG: Dict[str, Dict] = {
    "titan": {
        "dim": 768,
        "feature_dir": "features_titan",
        "display_name": "TITAN",
    },
    "uni2": {
        "dim": 1536,
        "feature_dir": "features_uni2",
        "display_name": "UNI2-h",
    },
    "virchow2": {
        "dim": 1280,
        "feature_dir": "features_virchow2",
        "display_name": "Virchow2",
    },
}

CLASS_NAMES = [CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))]


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
        sorted by (row, col) within each group.
    """
    groups: Dict[str, List[Tuple[Path, int, int]]] = defaultdict(list)

    for patch_path in patient_dir.iterdir():
        if not patch_path.suffix == ".png":
            continue
        try:
            img_id, row, col = parse_patch_filename(patch_path.name)
            groups[img_id].append((patch_path, row, col))
        except ValueError:
            pass  # Silently skip non-patch files

    # Sort by (row, col) for spatial ordering
    for img_id in groups:
        groups[img_id].sort(key=lambda x: (x[1], x[2]))

    return dict(groups)


# =============================================================================
# Backbone Loading — with Attention Extraction
# =============================================================================


class AttentionHook:
    """Captures the attention weights from a ViT attention layer via hook."""

    def __init__(self) -> None:
        self.attention: Optional[torch.Tensor] = None

    def hook_attn_weights(self, module, input, output) -> None:
        """Hook for attn_drop to capture post-softmax attention matrix."""
        self.attention = output.detach()


def _find_last_attn_module_timm(model: nn.Module):
    """
    Find the last transformer block's Attention module in a timm ViT.

    timm ViTs store blocks in `model.blocks` (nn.Sequential).
    Each block has `block.attn` → the Attention module with `attn_drop`.
    """
    blocks = None
    if hasattr(model, "blocks"):
        blocks = model.blocks

    if blocks is None or len(blocks) == 0:
        raise ValueError(
            "Cannot find transformer blocks in the timm model. "
            f"Top-level children: {[n for n, _ in model.named_children()]}"
        )

    last_block = blocks[-1]

    if hasattr(last_block, "attn"):
        return last_block.attn

    raise ValueError(
        f"Last block has no 'attn' attribute. "
        f"Children: {[n for n, _ in last_block.named_children()]}"
    )


def load_backbone(
    backbone_name: str,
    device: torch.device,
) -> Tuple[nn.Module, Callable, Callable]:
    """
    Load a ViT backbone and return (model, transform, attention_extractor).

    The attention_extractor is a callable that takes a [1, 3, H, H] tensor
    and returns the CLS→spatial attention map as a 2D numpy array.

    Architecture-specific handling:
        - TITAN/CONCHv1.5: Uses the native `model.trunk.get_attention(x)`
          method. The model is `EncoderWithAttentionalPooler` wrapping a
          `VisionTransformer` at `.trunk`. Input is 448×448, patch_size=16,
          giving a 28×28 spatial grid.
        - UNI2-h / Virchow2: Uses forward hooks on `attn_drop` in the last
          timm block. Input is 224×224, patch_size=14, giving 16×16 grid.

    Returns:
        model: The frozen ViT backbone.
        transform: Preprocessing transform for patches.
        get_attention: Function(patch_tensor) → np.ndarray [grid_h, grid_w]
    """
    hf_login()

    if backbone_name == "titan":
        from transformers import AutoModel

        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        conch, eval_transform = titan.return_conch()
        # conch is EncoderWithAttentionalPooler; ViT is at conch.trunk
        model = conch.to(device)
        model.eval()

        # CONCHv1.5 VisionTransformer has a native get_attention() method
        # that uses block.forward_with_attention() on the last block.
        # Input: 448×448, patch_size=16 → 28×28 = 784 spatial tokens.
        vit = model.trunk  # the actual VisionTransformer

        def get_attention(patch_tensor: torch.Tensor) -> np.ndarray:
            with torch.inference_mode():
                # get_attention returns [B, num_heads, seq_len, seq_len]
                attn = vit.get_attention(patch_tensor)
            # Average across heads: [B, seq_len, seq_len] → [seq_len, seq_len]
            attn_avg = attn.mean(dim=1)
            # CLS token (index 0) attention to spatial tokens (index 1:)
            cls_attn = attn_avg[0, 0, 1:]  # [S]
            # CONCHv1.5: 448/16 = 28 → 28×28 grid
            grid_size = int(cls_attn.shape[0] ** 0.5)
            cls_attn_2d = cls_attn.reshape(grid_size, grid_size)
            return cls_attn_2d.cpu().float().numpy()

        return model, eval_transform, get_attention

    elif backbone_name in ("uni2", "virchow2"):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

        if backbone_name == "uni2":
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
        else:  # virchow2
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )

        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        model = model.to(device)
        model.eval()

        # Register hook on the last block's attention dropout layer.
        # timm Attention.forward: attn = softmax(q @ k^T); attn = attn_drop(attn)
        # Hooking attn_drop captures the post-softmax attention matrix.
        attn_module = _find_last_attn_module_timm(model)

        # Disable fused/SDPA attention so the manual path through attn_drop
        # is used — otherwise F.scaled_dot_product_attention bypasses the hook.
        if hasattr(attn_module, "fused_attn"):
            attn_module.fused_attn = False

        hook = AttentionHook()
        if hasattr(attn_module, "attn_drop"):
            attn_module.attn_drop.register_forward_hook(hook.hook_attn_weights)
        else:
            raise ValueError(
                "Cannot find 'attn_drop' in the last block's Attention module."
            )

        # Determine spatial grid size
        # UNI2-h: 224/14 = 16×16; Virchow2: 224/14 = 16×16
        grid_size = 224 // 14  # 16

        def get_attention(patch_tensor: torch.Tensor) -> np.ndarray:
            hook.attention = None  # Reset before forward
            with torch.inference_mode():
                model(patch_tensor)
            if hook.attention is None:
                raise RuntimeError(
                    "Attention hook did not capture weights. "
                    "The model may use fused attention (F.scaled_dot_product_attention) "
                    "which bypasses attn_drop. Try running with "
                    "TORCH_CUDNN_SDPA_ENABLED=0 or on CPU."
                )
            attn = hook.attention  # [B, num_heads, seq_len, seq_len]
            if attn.dim() == 3:
                attn = attn.unsqueeze(0)
            attn_avg = attn.mean(dim=1)  # [B, seq_len, seq_len]
            # Handle CLS + register tokens + spatial tokens
            # CLS is at index 0; spatial tokens are the LAST (grid_size^2)
            num_spatial = grid_size * grid_size
            cls_attn = attn_avg[0, 0, -num_spatial:]  # [S]
            cls_attn_2d = cls_attn.reshape(grid_size, grid_size)
            return cls_attn_2d.cpu().float().numpy()

        return model, transform, get_attention

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


# =============================================================================
# MIL Model Loading
# =============================================================================


def load_mil_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, str, str]:
    """
    Load a trained MIL model from checkpoint.

    The checkpoint stores model_type and backbone metadata.

    Returns:
        model: Loaded MIL model in eval mode.
        model_type: 'simple' or 'hybrid'.
        backbone_name: 'titan', 'uni2', or 'virchow2'.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = checkpoint["args"]
    model_type = args["model_type"]
    backbone_name = args["backbone"]
    input_dim = BACKBONE_CONFIG[backbone_name]["dim"]
    num_classes = len(CLASS_MAP)

    if model_type == "simple":
        topk = args.get("topk", 0)
        model = SimpleGatedMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=topk,
        )
    elif model_type == "hybrid":
        topk = args.get("topk", 5)
        model = HybridMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=topk,
        )
    else:
        raise ValueError(
            f"Heatmap visualization supports 'simple' and 'hybrid' MIL models, "
            f"got '{model_type}'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, model_type, backbone_name


# =============================================================================
# MIL Attention Scoring
# =============================================================================


@torch.inference_mode()
def compute_mil_attention(
    features_path: Path,
    mil_model: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Compute MIL attention scores for all patches in a bag.

    Args:
        features_path: Path to .pt file with [N, D] features.
        mil_model: Trained MIL model.
        device: Torch device.

    Returns:
        attention: MIL attention weights [N], normalized to [0, 1].
        pred_class: Predicted class index.
        probs: Class probabilities [C].
    """
    features = torch.load(features_path, map_location=device, weights_only=True)
    features = features.to(device)

    logits, attention, _ = mil_model(features, return_attention=True)
    probs = F.softmax(logits, dim=0).cpu().numpy()
    pred_class = logits.argmax().item()

    # Attention is already softmax-normalized by the MIL model
    attention = attention.cpu().numpy()

    return attention, pred_class, probs


# =============================================================================
# Grid Gallery Generation
# =============================================================================


def create_patch_overlay(
    patch_path: Path,
    mil_score_normalized: float,
    get_vit_attention: Callable,
    backbone_transform: Callable,
    device: torch.device,
    patch_size: int = 224,
) -> np.ndarray:
    """
    Create a ViT attention overlay on a single patch with dynamic alpha.

    Args:
        patch_path: Path to the raw patch PNG.
        mil_score_normalized: MIL score normalized to [0, 1] across all patches.
        get_vit_attention: Callable returning [grid_h, grid_w] attention.
        backbone_transform: Preprocessing transform for the backbone.
        device: Torch device.
        patch_size: Expected patch size.

    Returns:
        Blended overlay image [H, W, 3] (uint8, RGB).
    """
    # Load raw patch
    pil_img = Image.open(patch_path).convert("RGB")
    raw_patch = np.array(pil_img)  # [224, 224, 3] uint8 RGB

    # Extract ViT attention
    patch_tensor = backbone_transform(pil_img).unsqueeze(0).to(device)
    vit_attn_grid = get_vit_attention(patch_tensor)  # [grid, grid]

    # Resize with INTER_CUBIC + GaussianBlur for smoothness
    attn_resized = cv2.resize(
        vit_attn_grid.astype(np.float32),
        (patch_size, patch_size),
        interpolation=cv2.INTER_CUBIC,
    )
    attn_smooth = cv2.GaussianBlur(attn_resized, (11, 11), sigmaX=2.0)

    # Normalize to [0, 1]
    a_min, a_max = attn_smooth.min(), attn_smooth.max()
    if a_max - a_min > 1e-8:
        attn_norm = (attn_smooth - a_min) / (a_max - a_min)
    else:
        attn_norm = np.zeros_like(attn_smooth)

    # Gamma correction: tighten heatmap focus
    attn_norm = attn_norm**0.8

    # Apply COLORMAP_TURBO
    attn_uint8 = (attn_norm * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Dynamic alpha: 0.6 × normalized_attention per-pixel
    # Low-attention areas stay transparent → raw tissue visible
    alpha = 0.6 * attn_norm  # [H, W] in [0, 0.6]
    alpha_3ch = alpha[..., np.newaxis]  # [H, W, 1]

    overlay = (
        (1.0 - alpha_3ch) * raw_patch.astype(np.float64)
        + alpha_3ch * heatmap_rgb.astype(np.float64)
    ).astype(np.uint8)

    return overlay


def generate_grid_gallery(
    patient_dir: Path,
    image_id: str,
    mil_model: nn.Module,
    get_vit_attention: Callable,
    backbone_transform: Callable,
    features_path: Path,
    class_name: str,
    save_path: Path,
    patch_size: int = 224,
    n_cols: int = 8,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Generate a single grid gallery image with all patches sorted by MIL attention.

    Args:
        patient_dir: Patient directory containing patch PNGs.
        image_id: Source image ID.
        mil_model: Trained MIL model.
        get_vit_attention: Callable returning [grid_h, grid_w] attention.
        backbone_transform: Preprocessing transform for the backbone.
        features_path: Path to pre-extracted .pt features.
        class_name: Ground truth class name.
        save_path: Output file path for the grid image.
        patch_size: Patch size in pixels.
        n_cols: Number of columns in the grid (default: 8).
        device: Torch device.
    """
    import math
    import matplotlib.pyplot as plt

    # ── 1. Group patches for this image ─────────────────────────────
    all_groups = group_patches_by_image(patient_dir)
    if image_id not in all_groups:
        raise ValueError(
            f"Image ID '{image_id}' not found in {patient_dir}. "
            f"Available IDs: {sorted(all_groups.keys())}"
        )
    patches = all_groups[image_id]  # [(path, row, col), ...]
    num_patches = len(patches)
    print(f"    Patches: {num_patches}")

    # ── 2. Compute MIL attention ────────────────────────────────────
    mil_attention, pred_class, probs = compute_mil_attention(
        features_path, mil_model, device
    )
    pred_name = CLASS_NAMES[pred_class]
    prob_str = " | ".join(f"{CLASS_NAMES[i]}: {p:.3f}" for i, p in enumerate(probs))
    print(f"    Prediction: {pred_name} (GT: {class_name})")
    print(f"    Probabilities: {prob_str}")

    # ── 3. Sort patches by MIL attention (descending) ───────────────
    scored_patches = []
    for idx, (path, row, col) in enumerate(patches):
        score = mil_attention[idx] if idx < len(mil_attention) else 0.0
        scored_patches.append((path, row, col, score))

    scored_patches.sort(key=lambda x: x[3], reverse=True)

    # Normalize MIL scores to [0, 1] for dynamic alpha
    all_scores = np.array([s[3] for s in scored_patches])
    s_min, s_max = all_scores.min(), all_scores.max()
    if s_max - s_min > 1e-8:
        scores_normalized = (all_scores - s_min) / (s_max - s_min)
    else:
        scores_normalized = np.ones_like(all_scores)

    # ── 4. Create Matplotlib grid ───────────────────────────────────
    n_rows = math.ceil(num_patches / n_cols)
    cell_size = 2.2  # inches per cell
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size + 1.2),
        facecolor="white",
    )
    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Suptitle
    fig.suptitle(
        f"Image {image_id}  —  Pred: {pred_name} (GT: {class_name})\n{prob_str}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # ── 5. Process each patch ───────────────────────────────────────
    for rank, (path, row, col, score) in enumerate(
        tqdm(scored_patches, desc=f"    Grid (Image {image_id})", leave=False)
    ):
        r_idx = rank // n_cols
        c_idx = rank % n_cols
        ax = axes[r_idx, c_idx]

        overlay = create_patch_overlay(
            patch_path=path,
            mil_score_normalized=scores_normalized[rank],
            get_vit_attention=get_vit_attention,
            backbone_transform=backbone_transform,
            device=device,
            patch_size=patch_size,
        )

        ax.imshow(overlay)
        ax.set_title(
            f"Rank {rank + 1} | r{row}c{col}\nMIL: {score:.4f}",
            fontsize=8,
            pad=3,
        )
        ax.axis("off")

    # ── 6. Turn off unused subplots ─────────────────────────────────
    for idx in range(num_patches, n_rows * n_cols):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        axes[r_idx, c_idx].axis("off")

    # ── 7. Save ─────────────────────────────────────────────────────
    plt.subplots_adjust(wspace=0.05, hspace=0.35, top=0.93)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"    ✅ Saved: {save_path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sorted grid gallery of per-patch ViT attention heatmaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image:
  python src/visualize_heatmap.py \\
      --mil_checkpoint experiments/simple_titan/best.pth \\
      --patient_dir "data/processed_subtype/PV/PV1 G2" \\
      --image_id 1

  # All images for a patient:
  python src/visualize_heatmap.py \\
      --mil_checkpoint experiments/simple_titan/best.pth \\
      --patient_dir "data/processed_subtype/PV/PV1 G2"
        """,
    )
    parser.add_argument(
        "--mil_checkpoint",
        type=str,
        required=True,
        help="Path to trained MIL model checkpoint (.pth).",
    )
    parser.add_argument(
        "--patient_dir",
        type=str,
        required=True,
        help="Patient directory containing patch PNGs.",
    )
    parser.add_argument(
        "--image_id",
        type=str,
        default=None,
        help="Source image ID (e.g., '1'). If omitted, processes all images.",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Override features directory. Default: auto-detected from backbone.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: results/grid_heatmaps.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Patch size in pixels (default: 224).",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=8,
        help="Number of columns in the grid (default: 8).",
    )

    # Auto-detect device
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

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ── 1. Load MIL model ───────────────────────────────────────────
    print("=" * 60)
    print("Sorted Grid Gallery — ViT × MIL Attention Heatmaps")
    print("=" * 60)

    mil_checkpoint = Path(args.mil_checkpoint)
    mil_model, model_type, backbone_name = load_mil_model(mil_checkpoint, device)
    print(f"  MIL Model:  {model_type.upper()}")
    print(f"  Backbone:   {BACKBONE_CONFIG[backbone_name]['display_name']}")
    print(f"  Checkpoint: {mil_checkpoint}")

    # ── 2. Load ViT backbone ────────────────────────────────────────
    print(f"\nLoading {BACKBONE_CONFIG[backbone_name]['display_name']} backbone...")
    backbone, backbone_transform, get_vit_attention = load_backbone(
        backbone_name, device
    )
    print(f"  ✅ Backbone loaded.\n")

    # ── 3. Setup paths ──────────────────────────────────────────────
    patient_dir = Path(args.patient_dir)
    patient_name = patient_dir.name
    class_name = patient_dir.parent.name

    # Features directory
    if args.features_dir:
        features_dir = Path(args.features_dir)
    else:
        feature_dir_name = BACKBONE_CONFIG[backbone_name]["feature_dir"]
        data_root = patient_dir.parent.parent.parent
        features_dir = data_root / feature_dir_name

    patient_features_dir = features_dir / class_name / patient_name

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_DIR / "grid_heatmaps"

    print(f"  Patient:    {class_name}/{patient_name}")
    print(f"  Patches:    {patient_dir}")
    print(f"  Features:   {patient_features_dir}")
    print(f"  Output:     {output_dir}")

    # ── 4. Discover images ──────────────────────────────────────────
    all_groups = group_patches_by_image(patient_dir)

    if args.image_id:
        image_ids = [args.image_id]
    else:
        image_ids = sorted(all_groups.keys(), key=lambda x: int(x))

    print(f"  Images:     {len(image_ids)}")
    print("=" * 60)

    # ── 5. Generate grid gallery for each image ─────────────────────
    for img_id in image_ids:
        features_path = patient_features_dir / f"{img_id}.pt"
        if not features_path.exists():
            print(f"  ⚠ Features not found: {features_path} — skipping.")
            continue

        print(f"\n  Image {img_id}...")
        save_path = output_dir / patient_name / f"{img_id}_grid.png"

        generate_grid_gallery(
            patient_dir=patient_dir,
            image_id=img_id,
            mil_model=mil_model,
            get_vit_attention=get_vit_attention,
            backbone_transform=backbone_transform,
            features_path=features_path,
            class_name=class_name,
            save_path=save_path,
            patch_size=args.patch_size,
            n_cols=args.n_cols,
            device=device,
        )

    print(f"\n{'=' * 60}")
    print("Grid Gallery Generation Complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
