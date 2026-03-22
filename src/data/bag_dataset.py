"""
Bag-level Dataset for MIL (Multiple Instance Learning).

Loads pre-extracted feature bags (.pt files) produced by backbone extractors
(TITAN, UNI2-h, Virchow2). Each .pt file represents one WSI and contains
a tensor of shape [N_patches, Dim].

Dataset variants:
    - MPNBagDataset: Applies mean pooling to produce [Dim] vector per WSI.
    - MPNBagDatasetFull: Returns full [N_patches, Dim] tensor (for attention-based MIL).
    - MultiTaskBagDataset: Like MPNBagDatasetFull but also returns auxiliary
      cellularity labels and confidence weights from an expert metrics CSV.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
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


class GradingBagDatasetFull(Dataset):
    """
    PyTorch Dataset for attention-based MIL on Reticulin Fibrosis Grading.

    Extracts fibrosis grade labels (G0, G1, G2, G3) from the parent folder
    name of each .pt file, rather than using the MPN CLASS_MAP.

    Walks ``{features_dir}/**/*.pt`` using rglob and assigns labels based on
    whether the parent folder name contains 'G0', 'G1', 'G2', or 'G3'.

    Returns 3 items per sample: (features, label, slide_id).

    Args:
        features_dir: Root directory containing feature .pt files.
    """

    GRADE_MAP = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}

    def __init__(self, features_dir: Path) -> None:
        self.samples: List[Tuple[Path, int]] = []

        for pt_file in sorted(Path(features_dir).rglob("*.pt")):
            folder_name = pt_file.parent.name
            label = self._extract_grade(folder_name)
            if label is not None:
                self.samples.append((pt_file, label))

    def _extract_grade(self, folder_name: str):
        """Return integer label if folder name contains a grade string."""
        for grade, label in self.GRADE_MAP.items():
            if grade in folder_name:
                return label
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            features: Full patch features [N_patches, Dim].
            label:    Integer class label (0=G0, 1=G1, 2=G2, 3=G3).
            slide_id: Slide identifier (filename without extension).
        """
        pt_path, label = self.samples[idx]
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            feat = data["feats"]
        else:
            feat = data

        slide_id = pt_path.stem
        return feat, label, slide_id

    def get_labels(self) -> List[int]:
        """Return all labels (useful for stratified splitting / weighted sampling)."""
        return [label for _, label in self.samples]

    def get_slide_path(self, idx: int) -> Path:
        """Return the path to the .pt file for a given index."""
        return self.samples[idx][0]


CELLULARITY_MAP: Dict[str, int] = {
    "Hypocellular": 0,
    "Normocellular": 1,
    "Hypercellular / Panmyelosis": 2,
}


class MultiTaskBagDataset(Dataset):
    """
    Bag-level dataset for multi-task MIL with auxiliary cellularity supervision.

    Extends MPNBagDatasetFull by optionally loading expert cellularity labels
    from a CSV file. Each sample returns the feature tensor, the primary
    subtype label, an auxiliary cellularity label, and a confidence weight.

    Args:
        features_dir: Root directory containing per-class feature folders.
        cellularity_csv: Optional path to the expert metrics CSV. If None,
            all samples return aux_label=-1 and w_conf=0.0.
    """

    def __init__(
        self,
        features_dir: Path,
        cellularity_csv: Optional[str] = None,
    ) -> None:
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in CLASS_MAP.items():
            class_dir = features_dir / class_name
            if not class_dir.exists():
                continue
            for pt_file in sorted(class_dir.rglob("*.pt")):
                self.samples.append((pt_file, label))

        # Build cellularity lookup: "Subtype/PatientID/stem" -> (int_label, w_conf)
        self.cell_lookup: Dict[str, Tuple[int, float]] = {}
        if cellularity_csv is not None:
            df = pd.read_csv(cellularity_csv)
            for _, row in df.iterrows():
                full_path = Path(row["Full_Path"])
                # Key: e.g. "ET/ET1 G1/1"
                key = f"{full_path.parts[-3]}/{full_path.parts[-2]}/{full_path.stem}"
                cell_label = CELLULARITY_MAP.get(row["Cellularity"], -1)
                if cell_label == -1:
                    continue
                w_conf = float(row["Cellularity_Confidence"]) / 10.0
                self.cell_lookup[key] = (cell_label, w_conf)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, str, dict, int, float]:
        """
        Returns:
            features: Full patch features [N_patches, Dim].
            label:    Integer class label.
            slide_id: Slide identifier (filename without extension).
            metrics:  Dict of patch-level metrics (from extraction).
            aux_label: Cellularity label (0/1/2) or -1 if unavailable.
            w_conf:   Confidence weight (0.0 if unavailable).
        """
        pt_path, label = self.samples[idx]
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            feat = data["feats"]
            metrics = data.get("metrics", {})
        else:
            feat = data
            metrics = {}

        slide_id = pt_path.stem

        # Lookup auxiliary cellularity label
        # Key: "Subtype/PatientID/stem" derived from the .pt file path
        class_name = pt_path.parts[-3]  # e.g. "ET"
        patient_id = pt_path.parts[-2]  # e.g. "ET1 G1"
        roi_key = f"{class_name}/{patient_id}/{pt_path.stem}"
        aux_label, w_conf = self.cell_lookup.get(roi_key, (-1, 0.0))

        return feat, label, slide_id, metrics, aux_label, w_conf

    def get_labels(self) -> List[int]:
        """Return all labels (useful for stratified splitting / weighted sampling)."""
        return [label for _, label in self.samples]
