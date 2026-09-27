"""
D4 Image-View Feature Extraction — Reticulin-Stained Images (Silver Stain).

Stand-alone companion to ``extract_{virchow2,uni2,titan}_reti.py``. It reuses their
grouping rules, their models and their output schema; the ONLY difference is that a
dihedral (D4) transform is applied to each 224x224 patch BEFORE the backbone's own
transform, and each transform index is written to its own directory. Nothing here
touches or overwrites the existing feature directories.

WHY D4 AND NOTHING ELSE
    Reticulin (silver-impregnation) grading reads FIBRE DENSITY, THICKNESS and
    CONTINUITY per unit area. An image augmentation is admissible only if a
    pathologist would still assign the same MF grade afterwards, which rules out
    scale/zoom (density per area *is* the label), elastic warping (fibre morphology
    is the label) and free-angle rotation (interpolation blurs thin fibres and pads
    the border). The eight dihedral transforms are exact pixel permutations - no
    resampling, no padding, no photometric change - and a fibre network has no
    canonical orientation, so all eight are label-preserving by construction.
    H&E stain machinery (HED jitter, Macenko/Vahadane) is deliberately NOT offered:
    this is a silver stain, not haematoxylin+eosin.

VIEW INDEX (D4)
    0 identity (= the existing features_{backbone}_reti/, not re-extracted by default)
    1 rot90        2 rot180        3 rot270
    4 flip L-R     5 transpose     6 flip T-B      7 transverse

Input:  data/processed_grading/{Class}/{PatientID}/{ImgID}_r{Row}c{Col}.png
Output: {output_root}/features_{backbone}_reti_views/view{k}/{Class}/{PatientID}/{ImgID}.pt

Each .pt holds the same dict as the base extractors: {feats, metrics, rc, patch_paths}.
``feats`` is float16 by default (half the disk, half the Colab download); the training-time
augmentation casts back to float32. Pass --dtype fp32 for byte-comparable precision.

Extraction covers ALL ROIs, not just the training split, so one extraction is reusable
across every fold/seed. Augmentation is still train-only: the trainer never calls an
augmentation on val/test bags.

Usage (one backbone, all 7 non-identity views, model loaded once):
    python -m src.data.extract_reti_views --backbone virchow2 --views 1-7 --device cuda

    # Colab, writing straight to Drive, resumable across sessions:
    python -m src.data.extract_reti_views --backbone uni2 --views 1-3 \
        --output_root /content/drive/MyDrive/mpn --batch_size 128 --device cuda

    # Check the plan without running inference:
    python -m src.data.extract_reti_views --backbone titan --dry_run
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure src/ is on sys.path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.config import hf_login

# =============================================================================
# Constants
# =============================================================================
DEFAULT_DATA_DIR = "data/processed_grading"
DEFAULT_OUTPUT_ROOT = "data"
CLASSES = ["ET", "PV", "PMF"]

BACKBONES: Dict[str, Dict] = {
    "titan": {"dim": 768, "display_name": "TITAN/CONCHv1.5"},
    "uni2": {"dim": 1536, "display_name": "UNI2-h"},
    "virchow2": {"dim": 1280, "display_name": "Virchow2"},
}

# The eight elements of the dihedral group D4, as exact PIL pixel permutations.
# Index 0 is the identity (already covered by features_{backbone}_reti/).
D4_OPS: List[Tuple[str, object]] = [
    ("identity", None),
    ("rot90", Image.Transpose.ROTATE_90),
    ("rot180", Image.Transpose.ROTATE_180),
    ("rot270", Image.Transpose.ROTATE_270),
    ("flip_lr", Image.Transpose.FLIP_LEFT_RIGHT),
    ("transpose", Image.Transpose.TRANSPOSE),
    ("flip_tb", Image.Transpose.FLIP_TOP_BOTTOM),
    ("transverse", Image.Transpose.TRANSVERSE),
]


# =============================================================================
# Patch Grouping Utilities (identical rules to the base reti extractors)
# =============================================================================


def parse_patch_filename(filename: str) -> Tuple[str, int, int]:
    """Parse a patch filename into (image_id, row, col). e.g. "reti1_r2c5.png"."""
    match = re.match(r"^([a-zA-Z0-9_-]+)_r(\d+)c(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Unexpected patch filename format: {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def group_patches_by_image(
    patient_dir: Path,
) -> Dict[str, List[Tuple[Path, int, int]]]:
    """Group patches by source image ID, sorted by (row, col) within each group."""
    groups: Dict[str, List[Tuple[Path, int, int]]] = defaultdict(list)

    for patch_path in patient_dir.iterdir():
        if patch_path.suffix != ".png":
            continue
        try:
            img_id, row, col = parse_patch_filename(patch_path.name)
            groups[img_id].append((patch_path, row, col))
        except ValueError:
            print(f"  ⚠ Skipping unrecognised file: {patch_path.name}")

    for img_id in groups:
        groups[img_id].sort(key=lambda x: (x[1], x[2]))

    return dict(groups)


# =============================================================================
# Patch Dataset — the only place that differs from the base extractors
# =============================================================================


class ViewPatchDataset(Dataset):
    """Loads patches and applies one fixed D4 transform before the backbone transform.

    The transform is FIXED per dataset (not random): randomness lives at training
    time, where the augmentation draws a view per patch from the extracted bank.
    That keeps extraction deterministic and re-runnable.
    """

    def __init__(self, patch_paths: List[Path], transform, d4_op) -> None:
        self.patch_paths = patch_paths
        self.transform = transform
        self.d4_op = d4_op

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        patch_path = self.patch_paths[idx]
        img = Image.open(patch_path).convert("RGB")
        if self.d4_op is not None:
            img = img.transpose(self.d4_op)
        tensor = self.transform(img)

        match = re.search(r"_r(\d+)c(\d+)\.png$", patch_path.name)
        row = int(match.group(1)) if match else -1
        col = int(match.group(2)) if match else -1

        return tensor, row, col


# =============================================================================
# Model loading
# =============================================================================


def load_backbone(backbone: str, device: torch.device):
    """Load a frozen backbone and its eval transform. Mirrors the base extractors."""
    hf_login()

    if backbone == "virchow2":
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

    elif backbone == "uni2":
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

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

    elif backbone == "titan":
        from transformers import AutoModel

        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        model, transform = titan.return_conch()

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model = model.to(device)
    model.eval()
    return model, transform


# =============================================================================
# Feature Extraction
# =============================================================================


@torch.inference_mode()
def extract_features_for_image(
    patch_paths: List[Path],
    model: torch.nn.Module,
    transform,
    d4_op,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract backbone features for one image group under one D4 view."""
    dataset = ViewPatchDataset(patch_paths, transform, d4_op)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    use_amp = device.type == "cuda"
    all_features, all_rows, all_cols = [], [], []

    for batch_tensor, batch_rows, batch_cols in loader:
        batch_tensor = batch_tensor.to(device)
        with torch.autocast(device.type, torch.float16, enabled=use_amp):
            output = model(batch_tensor)

        # Virchow2 returns the full token sequence [B, 261, D] -> take [CLS].
        # UNI2-h and CONCHv1.5 already return [B, D].
        if output.ndim == 3:
            output = output[:, 0, :]

        all_features.append(output.float().cpu())
        all_rows.append(batch_rows)
        all_cols.append(batch_cols)

    feats = torch.cat(all_features, dim=0)
    rc_tensor = torch.stack(
        [torch.cat(all_rows, dim=0), torch.cat(all_cols, dim=0)], dim=1
    ).to(torch.int32)

    return feats, rc_tensor


# =============================================================================
# Main Pipeline
# =============================================================================


def build_extraction_plan(data_dir: Path):
    """Discover (class, patient, image_groups) triples plus totals."""
    plan: List[Tuple[str, str, Dict[str, List[Tuple[Path, int, int]]]]] = []
    total_images = 0
    total_patches = 0

    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠ Class directory not found: {class_dir}")
            continue

        patient_dirs = sorted(
            [d for d in class_dir.iterdir() if d.is_dir()], key=lambda d: d.name
        )
        for patient_dir in patient_dirs:
            image_groups = group_patches_by_image(patient_dir)
            if not image_groups:
                print(f"  ⚠ No patches found in {patient_dir.name}")
                continue
            plan.append((class_name, patient_dir.name, image_groups))
            total_images += len(image_groups)
            total_patches += sum(len(v) for v in image_groups.values())

    return plan, total_images, total_patches


def run_extraction(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    views = parse_views(args.views)
    cfg = BACKBONES[args.backbone]
    device = torch.device(args.device)
    save_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    views_root = (
        Path(args.output_root) / f"features_{args.backbone}_reti_views"
    )

    plan, total_images, total_patches = build_extraction_plan(data_dir)

    print("=" * 64)
    print(f"D4 Image-View Feature Extraction — {cfg['display_name']} (Reticulin)")
    print("=" * 64)
    print(f"  {'Data dir':<14}: {data_dir}")
    print(f"  {'Views root':<14}: {views_root}")
    print(f"  {'Views':<14}: {views} ({[D4_OPS[v][0] for v in views]})")
    print(f"  {'Patients':<14}: {len(plan)}")
    print(f"  {'Images':<14}: {total_images}  (x{len(views)} views = {total_images * len(views)} .pt files)")
    print(f"  {'Patches':<14}: {total_patches}  (x{len(views)} = {total_patches * len(views)} forward passes)")
    print(f"  {'Feature dim':<14}: {cfg['dim']}")
    print(f"  {'Save dtype':<14}: {args.dtype}")
    print(f"  {'Device':<14}: {device}")
    est_gb = total_patches * len(views) * cfg["dim"] * (2 if args.dtype == "fp16" else 4) / 1e9
    print(f"  {'Est. size':<14}: ~{est_gb:.2f} GB")
    print("=" * 64)

    if args.dry_run:
        print("\n🔍 DRY RUN — no model loaded, no inference run.\n")
        for v in views:
            print(f"  view{v} ({D4_OPS[v][0]}) → {views_root / f'view{v}'}/{{Class}}/{{Patient}}/{{ImgID}}.pt")
        print(f"\n✅ Dry run complete. {total_images * len(views)} .pt files would be created.")
        return

    print(f"\nLoading {cfg['display_name']}...")
    model, transform = load_backbone(args.backbone, device)
    print(f"✅ {cfg['display_name']} loaded and frozen (dim={cfg['dim']}).\n")

    processed = 0
    skipped = 0

    # Outer loop over views so that a session can be interrupted and resumed
    # view-by-view; the model is loaded exactly once for all of them.
    for v in views:
        view_name, d4_op = D4_OPS[v]
        view_dir = views_root / f"view{v}"
        print(f"\n── view{v} ({view_name}) → {view_dir}")

        for class_name, patient_name, groups in tqdm(
            plan, desc=f"view{v} patients", unit="patient"
        ):
            patient_output_dir = view_dir / class_name / patient_name
            patient_output_dir.mkdir(parents=True, exist_ok=True)

            for img_id in sorted(groups.keys()):
                out_path = patient_output_dir / f"{img_id}.pt"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                patch_paths = [p[0] for p in groups[img_id]]
                feats, rc_tensor = extract_features_for_image(
                    patch_paths=patch_paths,
                    model=model,
                    transform=transform,
                    d4_op=d4_op,
                    batch_size=args.batch_size,
                    device=device,
                    num_workers=args.num_workers,
                )

                torch.save(
                    {
                        "feats": feats.to(save_dtype),
                        "metrics": {},
                        "rc": rc_tensor,
                        "patch_paths": [str(p) for p in patch_paths],
                        "view": v,
                        "view_name": view_name,
                    },
                    out_path,
                )
                processed += 1

    print(f"\n{'=' * 64}")
    print(f"D4 View Extraction Complete — {cfg['display_name']} (Reticulin)")
    print(f"{'=' * 64}")
    print(f"  Extracted : {processed} bag-views")
    print(f"  Skipped   : {skipped} (already existed)")
    print(f"  Output    : {views_root}")
    print(f"{'=' * 64}")
    print("\nTrain with:  --augmentation image_view --aug_strength 1.0")


# =============================================================================
# CLI
# =============================================================================


def parse_views(spec: str) -> List[int]:
    """Parse "1-7" / "1,2,5" / "all" / "3" into a sorted list of D4 indices."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(8))

    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))

    bad = [v for v in out if not 0 <= v <= 7]
    if bad:
        raise ValueError(f"View indices must be in 0..7, got {bad}")
    return sorted(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen backbone features under D4 image views (Reticulin)."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=sorted(BACKBONES.keys()),
        help="Which frozen foundation model to run.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="1-7",
        help="D4 view indices: '1-7' (default, all non-identity), 'all' (0-7), "
             "or a comma list like '1,2,6'. 0 = identity (already in features_*_reti/).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Input directory with {Class}/{PatientID}/ structure.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root under which features_{backbone}_reti_views/ is created.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Storage precision for feats (default fp16: half the disk/download).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    if torch.cuda.is_available():
        _default_device = "cuda"
    elif torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument("--device", type=str, default=_default_device)

    parser.add_argument("--overwrite", action="store_true", help="Re-extract existing .pt files.")
    parser.add_argument("--dry_run", action="store_true", help="Print the plan and exit.")
    return parser.parse_args()


def main():
    run_extraction(parse_args())


if __name__ == "__main__":
    main()
