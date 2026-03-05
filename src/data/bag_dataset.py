"""
Bag-level Dataset for MIL (Multiple Instance Learning).

Loads pre-extracted feature bags (.pt files) produced by backbone extractors
(TITAN, UNI2-h, Virchow2). Each .pt file represents one WSI and contains
a tensor of shape [N_patches, Dim].

Two dataset variants are provided:
    - MPNBagDataset: Applies mean pooling to produce [Dim] vector per WSI.
    - MPNBagDatasetFull: Returns full [N_patches, Dim] tensor (for attention-based MIL).
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from core.config import CLASS_MAP


class MPNBagDataset(Dataset):
    """
    PyTorch Dataset for bag-level MPN classification (mean-pooled).

    Walks ``{features_dir}/{ClassName}/.../*.pt`` using rglob to support
    both flat (Class/*.pt) and nested (Class/SlideID/*.pt) layouts.

    Each sample is mean-pooled from [N_patches, Dim] -> [Dim].

    Args:
        features_dir: Root directory containing per-class feature folders.
    """

    def __init__(self, features_dir: Path) -> None:
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in CLASS_MAP.items():
            class_dir = features_dir / class_name
            if not class_dir.exists():
                continue
            for pt_file in sorted(class_dir.rglob("*.pt")):
                self.samples.append((pt_file, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            feature: Mean-pooled feature vector [Dim].
            label:   Integer class label.
        """
        pt_path, label = self.samples[idx]
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            feat = data["feats"]
        else:
            feat = data
        # Mean pooling: [N_patches, Dim] -> [Dim]
        feat = feat.mean(dim=0)
        return feat, label

    def get_labels(self) -> List[int]:
        """Return all labels (useful for stratified splitting / weighted sampling)."""
        return [label for _, label in self.samples]


class MPNBagDatasetFull(Dataset):
    """
    PyTorch Dataset for attention-based MIL (ABMIL, DTFD-MIL, etc.).

    Returns the FULL sequence of patch features [N_patches, Dim] without
    any pooling. This is required for attention-based aggregation methods.

    Walks ``{features_dir}/{ClassName}/.../*.pt`` using rglob to support
    both flat (Class/*.pt) and nested (Class/SlideID/*.pt) layouts.

    Args:
        features_dir: Root directory containing per-class feature folders.
    """

    def __init__(
        self,
        features_dir: Path,
    ) -> None:
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in CLASS_MAP.items():
            class_dir = features_dir / class_name
            if not class_dir.exists():
                continue
            for pt_file in sorted(class_dir.rglob("*.pt")):
                self.samples.append((pt_file, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, dict]:
        """
        Returns:
            features: Full patch features [N_patches, Dim].
            label:    Integer class label.
            slide_id: Slide identifier (filename without extension).
            metrics:  Dict of patch-level metrics (from extraction).
        """
        pt_path, label = self.samples[idx]

        # Must be False to allow loading lists of strings (patch_paths)
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            feat = data["feats"]
            metrics = data.get("metrics", {})
        else:
            feat = data
            metrics = {}

        slide_id = pt_path.stem
        return feat, label, slide_id, metrics

    def get_labels(self) -> List[int]:
        """Return all labels (useful for stratified splitting / weighted sampling)."""
        return [label for _, label in self.samples]

    def get_slide_path(self, idx: int) -> Path:
        """Return the path to the .pt file for a given index."""
        return self.samples[idx][0]
