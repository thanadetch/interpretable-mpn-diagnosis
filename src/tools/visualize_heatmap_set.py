"""
Batch Heatmap Set Visualization.

Generates attention heatmaps for patients in a specific split (train/val/test/all) 
of a trained MIL experiment. Loads the checkpoint, recovers the specified indices, 
and iterates over every patient/image to produce sorted grid galleries.

Output: results/heatmaps/{ExperimentName}/{split}/{Class}/{PatientID}/

Usage:
    python -m src.tools.visualize_heatmap_set \
        --mil_checkpoint experiments/simple_titan_.../best_simple_titan.pth \
        --split test

    # Custom split and data root:
    python -m src.tools.visualize_heatmap_set \
        --mil_checkpoint experiments/simple_titan_.../best.pth \
        --data_root data \
        --split val \
        --seed 42
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add src/ to path so imports work when run from project root
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import CLASS_MAP, CLASS_MAP_INV, hf_login
from data.bag_dataset import MPNBagDatasetFull
from models.hybrid_mil import HybridMIL
from models.simple_mil import SimpleGatedMIL
from models.dtfd_mil import DTFDMIL
from models.dual_stream_mil import DualStreamMIL
from models.multi_branch_mil import MultiBranchMIL
from models.mean_pool_mil import MeanPoolMIL
from models.dist_pool_mil import DistPoolMIL
from models.mean_std_pool_mil import MeanStdPoolMIL
from models.mean_topk_pool_mil import MeanTopKPoolMIL

CLASS_NAMES = [CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))]

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
) -> Dict[str, list]:
    """
    Group all patches in a patient directory by source image ID.

    Returns:
        Dict mapping image_id → list of (patch_path, row, col),
        sorted by (row, col) within each group.
    """
    groups: Dict[str, list] = defaultdict(list)

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
        num_classes: int = 3,
) -> Tuple[nn.Module, str, str]:
    """
    Load a trained MIL model from checkpoint.

    The checkpoint stores model_type and backbone metadata.

    Returns:
        model: Loaded MIL model in eval mode.
        model_type: 'simple' or 'hybrid'.
        backbone_name: 'titan', 'uni2', or 'virchow2'.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model_type = args["model_type"]
    backbone_name = args["backbone"]
    input_dim = BACKBONE_CONFIG[backbone_name]["dim"]

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
    elif model_type == "dtfd":
        num_pseudo_bags = args.get("num_pseudo_bags", 8)
        dropout = args.get("dropout", 0.25)
        model = DTFDMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            num_pseudo_bags=num_pseudo_bags,
            dropout=dropout,
        )
    elif model_type == "dual_stream":
        topk = args.get("topk", 5)
        model = DualStreamMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=topk,
        )
    elif model_type == "multi_branch":
        model = MultiBranchMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk_focal=args.get("topk_focal", 5),
        )
    elif model_type == "mean_pool":
        dropout = args.get("dropout", 0.5)
        model = MeanPoolMIL(
            vision_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_type == "dist_pool":
        model = DistPoolMIL(
            vision_dim=input_dim,
            num_classes=num_classes,
        )
    elif model_type == "mean_std_pool":
        model = MeanStdPoolMIL(
            vision_dim=input_dim,
            num_classes=num_classes,
        )
    elif model_type == "mean_topk_pool":
        model = MeanTopKPoolMIL(
            vision_dim=input_dim,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Heatmap visualization supports 'simple', 'hybrid', 'dtfd', 'dual_stream', "
            f"'multi_branch', 'mean_pool', 'dist_pool', 'mean_std_pool', and "
            f"'mean_topk_pool' MIL models, got '{model_type}'."
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
        ckpt_args: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], int, np.ndarray]:
    """
    Compute MIL attention scores for all patches in a bag.

    Args:
        features_path: Path to .pt file with [N, D] features (tensor or dict).
        mil_model: Trained MIL model.
        device: Torch device.
        ckpt_args: Checkpoint args dict (to check attention_bias flag).

    Returns:
        attention: MIL attention weights [N], normalized to [0, 1], or None for MeanPoolMIL.
        pred_class: Predicted class index.
        probs: Class probabilities [C].
    """
    data = torch.load(features_path, map_location=device, weights_only=False)

    if isinstance(data, dict):
        features = data["feats"].to(device)
        metrics = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.get("metrics", {}).items()
        }
    else:
        features = data.to(device)
        metrics = {}

    # These pooling models have no spatial attention mechanism
    if isinstance(mil_model, (MeanPoolMIL, DistPoolMIL, MeanStdPoolMIL, MeanTopKPoolMIL)):
        outputs = mil_model(features)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        attention = None
    else:
        # Only pass metrics if the model was trained with --attention_bias
        use_bias = (ckpt_args or {}).get("attention_bias", False)
        if isinstance(mil_model, SimpleGatedMIL):
            logits, attention, _ = mil_model(
                features, return_attention=True, metrics=metrics if use_bias else None
            )
        else:
            logits, attention, _ = mil_model(features, return_attention=True)

        # Attention is already softmax-normalized by the MIL model
        attention = attention.cpu().numpy()

    probs = F.softmax(logits, dim=0).cpu().numpy()
    pred_class = logits.argmax().item()

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
    attn_norm = attn_norm ** 0.8

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
        class_names: List[str],
        patch_size: int = 224,
        n_cols: int = 8,
        device: torch.device = torch.device("cpu"),
        ckpt_args: Optional[dict] = None,
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
        features_path, mil_model, device, ckpt_args=ckpt_args
    )
    pred_name = class_names[pred_class]
    prob_str = " | ".join(f"{class_names[i]}: {p:.3f}" for i, p in enumerate(probs))
    print(f"    Prediction: {pred_name} (GT: {class_name})")
    print(f"    Probabilities: {prob_str}")

    # ── 3. Sort patches by MIL attention (descending) ───────────────
    scored_patches = []
    if mil_attention is not None:
        for idx, (path, row, col) in enumerate(patches):
            score = mil_attention[idx] if idx < len(mil_attention) else 0.0
            scored_patches.append((path, row, col, score))

        scored_patches.sort(key=lambda x: x[3], reverse=True)

        all_scores = np.array([s[3] for s in scored_patches])
        s_min, s_max = all_scores.min(), all_scores.max()
        if s_max - s_min > 1e-8:
            scores_normalized = (all_scores - s_min) / (s_max - s_min)
        else:
            scores_normalized = np.ones_like(all_scores)
    else:
        # Mean Pooling: Keep spatial order, no MIL scores
        for path, row, col in patches:
            scored_patches.append((path, row, col, None))
        scores_normalized = np.ones(len(patches))

    # ── 4. Create Matplotlib grid ───────────────────────────────────
    n_rows = math.ceil(num_patches / n_cols)
    cell_size = 2.2  # inches per cell
    header_height = 1.0  # inches reserved for suptitle
    fig_height = n_rows * cell_size + header_height
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * cell_size, fig_height),
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
        f"Image {image_id}  \u2014  Pred: {pred_name} (GT: {class_name})\n{prob_str}",
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
        if score is not None:
            title_str = f"Rank {rank + 1} | r{row}c{col}\nMIL: {score:.4f}"
        else:
            title_str = f"r{row}c{col}"

        ax.set_title(title_str, fontsize=8, pad=3)
        ax.axis("off")

    # ── 6. Turn off unused subplots ─────────────────────────────────
    for idx in range(num_patches, n_rows * n_cols):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        axes[r_idx, c_idx].axis("off")

    # ── 7. Save ─────────────────────────────────────────────────────
    dynamic_top = 1.0 - (header_height / fig_height)
    plt.subplots_adjust(wspace=0.05, hspace=0.35, top=dynamic_top)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"    ✅ Saved: {save_path}")


# =============================================================================
# Checkpoint Helpers
# =============================================================================


def load_checkpoint_metadata(
        checkpoint_path: Path,
        device: torch.device,
) -> Tuple[dict, str, str, dict]:
    """
    Load checkpoint and extract metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    backbone_name = ckpt_args["backbone"]
    model_type = ckpt_args["model_type"]
    return checkpoint, backbone_name, model_type, ckpt_args


def get_target_patients(
        dataset: MPNBagDatasetFull,
        checkpoint: dict,
        ckpt_args: dict,
        split: str,
        seed_arg: Any,
) -> Dict[str, Dict]:
    """
    Map targeted split indices to unique patients with their metadata.
    """
    seed = seed_arg if seed_arg is not None else ckpt_args.get("seed", 42)
    indices = []

    if split != "all":
        # 1. Best Practice: Explicitly load target split directly from checkpoint
        if split == "train" and "train_idx" in checkpoint:
            print("  ✅ Loading 'train' patients directly from checkpoint...")
            indices = checkpoint["train_idx"]
        elif split == "val" and "val_idx" in checkpoint:
            print("  ✅ Loading 'val' patients directly from checkpoint...")
            indices = checkpoint["val_idx"]
        elif split == "test" and "test_idx" in checkpoint:
            print("  ✅ Loading 'test' patients directly from checkpoint...")
            indices = checkpoint["test_idx"]

        # 2. Fallback: If not found, fallback to original seed-based logic
        else:
            print(
                f"  ⚠️ '{split}_idx' not found in checkpoint! Falling back to original seed logic..."
            )
            try:
                from train_mil import patient_split

                train_idx, val_idx, test_idx = patient_split(dataset, seed=seed)
                if split == "train":
                    indices = train_idx
                elif split == "val":
                    indices = val_idx
                else:
                    indices = test_idx
            except ImportError:
                print(
                    "  ❌ Could not import patient_split from train_mil. Please ensure train_mil is in your src directory."
                )
                sys.exit(1)
    else:
        # 'all' split
        indices = list(range(len(dataset.samples)))

    patients: Dict[str, Dict] = {}
    for idx in indices:
        pt_path, label = dataset.samples[idx]
        patient_id = pt_path.parent.name
        class_name = pt_path.parent.parent.name

        if patient_id not in patients:
            patients[patient_id] = {
                "class_name": class_name,
                "label": label,
                "feature_paths": [],
            }
        patients[patient_id]["feature_paths"].append(pt_path)

    return patients


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate attention heatmaps for patients in a specific split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mil_checkpoint",
        type=str,
        required=True,
        help="Path to trained MIL model checkpoint (.pth).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to visualize (default: test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for splitting if split indices are not in checkpoint.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/heatmaps",
        help="Base output directory (default: results/heatmaps).",
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
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="Optional string to append to the experiment name.",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default=None,
        choices=["ET", "PV", "PMF"],
        help="Specific subtype to visualize. If not provided, visualizes all subtypes.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["et_vs_pv", "pmf_vs_nonpmf", "et_vs_nonet", "multi"],
        default="et_vs_pv",
        help="Task type: et_vs_pv, pmf_vs_nonpmf, et_vs_nonet, or multi (default: et_vs_pv).",
    )

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
    mil_checkpoint = Path(args.mil_checkpoint)

    if args.task == "et_vs_pv":
        class_names = ["ET", "PV"]
        num_classes = 2
    elif args.task == "pmf_vs_nonpmf":
        class_names = ["non-PMF", "PMF"]
        num_classes = 2
    elif args.task == "et_vs_nonet":
        class_names = ["non-ET", "ET"]
        num_classes = 2
    else:
        class_names = [CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))]
        num_classes = len(class_names)

    print("=" * 60)
    print("Batch Heatmap Set Visualization")
    print("=" * 60)

    checkpoint, backbone_name, model_type, ckpt_args = load_checkpoint_metadata(
        mil_checkpoint, device
    )
    cfg = BACKBONE_CONFIG[backbone_name]
    features_dir = Path(args.data_root) / cfg["feature_dir"]
    exp_name = mil_checkpoint.parent.name

    print(f"  Checkpoint:  {mil_checkpoint}")
    print(f"  Experiment:  {exp_name}")
    print(f"  Model:       {model_type.upper()}")
    print(f"  Backbone:    {cfg['display_name']}")
    print(f"  Split:       {args.split.upper()}")
    print(f"  Device:      {device}")

    # Initialise dataset and discover target patients
    dataset = MPNBagDatasetFull(features_dir)
    patients = get_target_patients(
        dataset, checkpoint, ckpt_args, args.split, args.seed
    )

    if args.subtype is not None:
        patients = {
            pid: info
            for pid, info in patients.items()
            if info["class_name"] == args.subtype
        }
        print(f"\n  Filtering to subtype: {args.subtype}")

    class_counts: Dict[str, int] = defaultdict(int)
    for info in patients.values():
        class_counts[info["class_name"]] += 1

    print(f"\n  Target Patients: {len(patients)}")
    for cn in class_names:
        print(f"    {cn}: {class_counts.get(cn, 0)} patients")

    print("\nLoading MIL model...")
    mil_model, _, _ = load_mil_model(mil_checkpoint, device, num_classes=num_classes)
    print("  ✅ MIL model loaded.\n")

    print(f"Loading {cfg['display_name']} backbone...")
    backbone, backbone_transform, get_vit_attention = load_backbone(
        backbone_name, device
    )
    print("  ✅ Backbone loaded.\n")

    # ── Setup output directory (Similar to analyze_shortcuts) ──────
    folder_name = f"{exp_name}_{args.postfix}" if args.postfix else exp_name

    # Structure: base_dir / exp_name / split
    output_root = Path(args.output_dir) / folder_name / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {output_root}")
    print("=" * 60)

    # Locate raw patch directories
    raw_patch_root = Path(args.data_root) / "processed_subtype"

    total_images = 0
    skipped_patients = 0

    for patient_idx, (patient_id, info) in enumerate(sorted(patients.items()), start=1):
        class_name = info["class_name"]
        feature_paths = sorted(info["feature_paths"])

        print(
            f"\n[{patient_idx}/{len(patients)}] "
            f"{class_name}/{patient_id}  ({len(feature_paths)} images)"
        )

        patient_patch_dir = raw_patch_root / class_name / patient_id
        if not patient_patch_dir.exists():
            print(f"  ⚠ Raw patch directory not found: {patient_patch_dir} — skipping.")
            skipped_patients += 1
            continue

        available_groups = group_patches_by_image(patient_patch_dir)

        # Output subdirectory for this patient
        patient_output_dir = output_root / class_name / patient_id

        for feat_path in feature_paths:
            img_id = feat_path.stem

            if img_id not in available_groups:
                print(f"  ⚠ No raw patches for image {img_id} — skipping.")
                continue

            save_path = patient_output_dir / f"{img_id}_grid.png"

            print(f"  Image {img_id}...")
            generate_grid_gallery(
                patient_dir=patient_patch_dir,
                image_id=img_id,
                mil_model=mil_model,
                get_vit_attention=get_vit_attention,
                backbone_transform=backbone_transform,
                features_path=feat_path,
                class_name=class_name,
                save_path=save_path,
                class_names=class_names,
                patch_size=args.patch_size,
                n_cols=args.n_cols,
                device=device,
                ckpt_args=ckpt_args,
            )
            total_images += 1

    print(f"\n{'=' * 60}")
    print(f"Batch Heatmap Generation Complete ({args.split.upper()} Set)")
    print(f"{'=' * 60}")
    print(f"  Patients processed: {len(patients) - skipped_patients}/{len(patients)}")
    print(f"  Images processed:   {total_images}")
    print(f"  Patients skipped:   {skipped_patients}")
    print(f"  Output directory:   {output_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
