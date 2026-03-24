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
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from core.config import CLASS_MAP

# Cellularity string → one-hot index mapping
_CELLULARITY_ONEHOT: Dict[str, List[float]] = {
    "Hypocellular": [1.0, 0.0, 0.0],
    "Normocellular": [0.0, 1.0, 0.0],
    "Hypercellular / Panmyelosis": [0.0, 0.0, 1.0],
}


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


class BagFusionDataset(Dataset):
    """
    PyTorch Dataset for concept-augmented MIL models.

    Extends ``MPNBagDatasetFull`` by adding a binary concept tensor derived
    from an expert-assessed cellularity CSV (``evaluate_cellularity.py``).

    Each sample returns a 2-dimensional concept tensor:
        [is_hyper, confidence_weight]

    - ``is_hyper``: 1.0 if cellularity contains "Hypercellular", else 0.0.
    - ``confidence_weight``: Cellularity_Confidence / 10.0 (normalised to [0, 1]).

    Falls back to ``torch.zeros(2)`` for ROIs not found in the CSV.

    Args:
        features_dir: Root directory containing per-class feature folders.
        cellularity_csv: Path to the expert metrics CSV with columns
            ``Patient_ID``, ``Filename``, ``Cellularity``,
            ``Cellularity_Confidence``.
    """

    def __init__(self, features_dir: Path, cellularity_csv: Path) -> None:
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in CLASS_MAP.items():
            class_dir = features_dir / class_name
            if not class_dir.exists():
                continue
            for pt_file in sorted(class_dir.rglob("*.pt")):
                self.samples.append((pt_file, label))

        # Build lookup: (Patient_ID, filename_stem) → concept tensor [2]
        self.concept_lookup: Dict[Tuple[str, str], torch.Tensor] = {}
        if cellularity_csv is not None and Path(cellularity_csv).exists():
            df = pd.read_csv(cellularity_csv)
            for _, row in df.iterrows():
                patient_id = str(row["Patient_ID"])
                fname_stem = Path(str(row["Filename"])).stem
                cellularity = str(row.get("Cellularity", ""))
                confidence = float(row.get("Cellularity_Confidence", 0))

                is_hyper = 1.0 if "Hypercellular" in cellularity else 0.0
                conf_weight = max(0.0, min(confidence / 10.0, 1.0))
                concept = torch.tensor([is_hyper, conf_weight], dtype=torch.float32)
                self.concept_lookup[(patient_id, fname_stem)] = concept

        # ── Strict mapping validation ────────────────────────────────
        matched = 0
        unmatched_keys: List[Tuple[str, str]] = []
        for pt_path, _ in self.samples:
            key = (pt_path.parent.name, pt_path.stem)
            if key in self.concept_lookup:
                matched += 1
            else:
                unmatched_keys.append(key)
        unmatched = len(unmatched_keys)

        print(f"\n{'─' * 50}")
        print(f"  BagFusionDataset Mapping Validation")
        print(f"{'─' * 50}")
        print(f"  Total Samples:   {len(self.samples)}")
        print(f"  CSV Entries:     {len(self.concept_lookup)}")
        print(f"  Matched:         {matched}")
        print(f"  Unmatched:       {unmatched}")
        if unmatched > 0:
            print(f"  ⚠️ First {min(10, unmatched)} unmatched (patient_id, slide_id):")
            for pid, sid in unmatched_keys[:10]:
                print(f"     ({pid}, {sid})")
        else:
            print(f"  ✅ All samples matched!")
        print(f"{'─' * 50}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, torch.Tensor, str, dict]:
        """
        Returns:
            features: Full patch features [N_patches, Dim].
            label:    Integer class label.
            concept_tensor: Binary concept vector [2] = [is_hyper, confidence].
            slide_id: Slide identifier (filename without extension).
            metrics:  Dict of patch-level metrics (from extraction).
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
        patient_id = pt_path.parent.name

        # Look up concept tensor; fall back to zeros if not found
        key = (patient_id, slide_id)
        concept_tensor = self.concept_lookup.get(key, torch.zeros(2))

        return feat, label, concept_tensor, slide_id, metrics

    def get_labels(self) -> List[int]:
        """Return all labels (useful for stratified splitting / weighted sampling)."""
        return [label for _, label in self.samples]

    def get_slide_path(self, idx: int) -> Path:
        """Return the path to the .pt file for a given index."""
        return self.samples[idx][0]
