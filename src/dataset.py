"""
Custom Dataset for MPN Classification and Fibrosis Grading.
Implements dual-task data loading with H&E and Reticulin image filtering.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE


class MPNDataset(Dataset):
    """
    PyTorch Dataset for MPN bone marrow pathology images.

    Supports two tasks:
    - Classification: Uses H&E images (excludes 'reti' files)
    - Grading: Uses Reticulin images (only 'reti' files)

    Args:
        file_list: List of (file_path, label) tuples
        task: Either 'classification' or 'grading'
        transform: Optional torchvision transforms
        is_training: Whether this is for training (affects default transforms)
    """

    def __init__(
        self,
        file_list: List[Tuple[Path, int]],
        task: str,
        transform: Optional[Callable] = None,
        is_training: bool = True,
    ) -> None:
        """
        Initialize the MPN Dataset.

        Args:
            file_list: List of (file_path, label) tuples (already filtered)
            task: Either 'classification' or 'grading'
            transform: Optional custom transforms
            is_training: If True, applies data augmentation
        """
        self.task = task
        self.is_training = is_training

        # Validate task
        if task not in ("classification", "grading"):
            raise ValueError(f"Unknown task: {task}. Use 'classification' or 'grading'")

        # Filter files based on task (additional safety check)
        self.samples: List[Tuple[Path, int]] = []
        for file_path, label in file_list:
            filename_lower = file_path.name.lower()

            if task == "classification":
                # H&E images only: exclude 'reti' files
                if "reti" not in filename_lower:
                    self.samples.append((file_path, label))
            elif task == "grading":
                # Reticulin images only: keep ONLY 'reti' files
                if "reti" in filename_lower:
                    self.samples.append((file_path, label))

        # Set transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(is_training)

    def _get_default_transforms(self, is_training: bool) -> transforms.Compose:
        """
        Get default transforms based on training/evaluation mode.

        Args:
            is_training: Whether to apply training augmentations

        Returns:
            Composed transforms
        """
        if is_training:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label, file_path_string)
        """
        file_path, label = self.samples[idx]

        # Load image
        image = Image.open(file_path).convert("RGB")

        # Apply transforms
        image = self.transform(image)

        return image, label, str(file_path)

    def get_labels(self) -> List[int]:
        """
        Get all labels in the dataset.

        Returns:
            List of integer labels
        """
        return [label for _, label in self.samples]

    def get_patient_id(self, file_path: str) -> str:
        """
        Extract patient ID from file path.

        Args:
            file_path: Path to image file

        Returns:
            Patient ID string (folder name)
        """
        return Path(file_path).parent.name

