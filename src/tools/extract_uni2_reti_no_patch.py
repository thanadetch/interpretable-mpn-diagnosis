"""
UNI2-h Feature Extraction — No-Patch Reticulin (Ablation: no patches).

Each input PNG is a *whole* ROI image already preprocessed (cropped +
resized to 224x224) by ``src/data/preprocess_no_patch.py``. We extract
one UNI2-h [CLS] vector per ROI and save it as a single-instance bag
(``feats`` of shape ``[1, 1536]``), so the existing MIL training script
``src/train_grading_reti.py`` can consume it unchanged.

Input :  data/processed_grading_no_patch/{Class}/{PatientID}/{ImgID}.png
Output:  data/features_uni2_reti_no_patch/{Class}/{PatientID}/{ImgID}.pt

Usage:
    python -m src.tools.extract_uni2_reti_no_patch \
        --data_dir data/processed_grading_no_patch \
        --output_dir data/features_uni2_reti_no_patch \
        --batch_size 64

After extraction, train via:
    python -m src.train_grading_reti \
        --backbone uni2_no_patch --model_type mean_pool \
        --formulation regression
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure src/ is on sys.path when running directly.
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
DEFAULT_DATA_DIR = "data/processed_grading_no_patch"
DEFAULT_OUTPUT_DIR = "data/features_uni2_reti_no_patch"
FEATURE_DIM = 1536
CLASSES = ["ET", "PV", "PMF"]


# =============================================================================
# No-Patch Dataset (one entry per source image)
# =============================================================================


class WholeROIDataset(Dataset):
    """Loads pre-resized no-patch PNGs and applies the timm eval transform."""

    def __init__(self, image_paths: List[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), idx


# =============================================================================
# Discovery
# =============================================================================


def discover_rois(data_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Return (class_name, patient_name, image_path) for every no-patch PNG
    found under ``{data_dir}/{Class}/{PatientID}/*.png``.
    """
    rois: List[Tuple[str, str, Path]] = []
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠ Class directory not found: {class_dir}")
            continue
        for patient_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            for f in sorted(patient_dir.iterdir()):
                if f.suffix.lower() == ".png":
                    rois.append((class_name, patient_dir.name, f))
    return rois


# =============================================================================
# Main pipeline
# =============================================================================


@torch.inference_mode()
def run_extraction(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    rois = discover_rois(data_dir)

    print("=" * 60)
    print("UNI2-h No-Patch Feature Extraction (Reticulin)")
    print("=" * 60)
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  ROIs found  : {len(rois)}")
    print(f"  Feature dim : {FEATURE_DIM}")
    print(f"  Device      : {device}")
    print("=" * 60)

    if args.dry_run:
        for class_name, patient_name, img_path in rois:
            out_path = (
                output_dir / class_name / patient_name / f"{img_path.stem}.pt"
            )
            print(f"  {img_path}  ->  {out_path}")
        print(f"\n✅ Dry run complete. {len(rois)} .pt files would be created.")
        return

    # Filter out already-extracted ROIs (unless --overwrite)
    pending: List[Tuple[str, str, Path, Path]] = []
    skipped = 0
    for class_name, patient_name, img_path in rois:
        out_path = output_dir / class_name / patient_name / f"{img_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        pending.append((class_name, patient_name, img_path, out_path))

    if not pending:
        print(f"\nNothing to do (skipped {skipped} existing files).")
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
    model = model.to(device).eval()
    print(f"✅ UNI2-h loaded (feature_dim={FEATURE_DIM}).\n")

    # ------------------------------------------------------------------
    # Batched feature extraction
    # ------------------------------------------------------------------
    image_paths = [item[2] for item in pending]
    dataset = WholeROIDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    use_amp = device.type == "cuda"
    written = 0
    cursor = 0

    for batch_tensor, _ in tqdm(loader, desc="ROIs", unit="batch"):
        batch_tensor = batch_tensor.to(device)
        with torch.autocast(device.type, torch.float16, enabled=use_amp):
            output = model(batch_tensor)  # [B, 1536]
        feats_cpu = output.float().cpu()  # [B, 1536]

        for i in range(feats_cpu.shape[0]):
            class_name, patient_name, img_path, out_path = pending[cursor]
            cursor += 1

            feats = feats_cpu[i : i + 1]  # [1, 1536]  (single-instance bag)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "feats": feats,
                    "metrics": {},
                    "rc": torch.zeros((1, 2), dtype=torch.int32),
                    "patch_paths": [str(img_path)],
                },
                out_path,
            )
            written += 1

    print(f"\n{'=' * 60}")
    print("UNI2-h No-Patch Extraction Complete")
    print(f"{'=' * 60}")
    print(f"  Extracted : {written} ROIs")
    print(f"  Skipped   : {skipped} (already exist)")
    print(f"  Output    : {output_dir}")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UNI2-h [CLS] features from no-patch Reticulin images."
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=64)

    if torch.cuda.is_available():
        _default_device = "cuda"
    elif torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument("--device", type=str, default=_default_device)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_extraction(parse_args())


if __name__ == "__main__":
    main()

