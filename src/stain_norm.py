"""
Universal Stain Normalization Layer (CPU/GPU Compatible).
Uses Standard H&E Reference Values (Macenko).
Classes are fully decoupled for independent testing.
"""

import numpy as np
import torch
import torch.nn as nn
from torchstain.base.normalizers import MacenkoNormalizer

# Threshold for detecting background patches (0-255 scale)
BACKGROUND_THRESHOLD = 235


def _is_background_torch(img: torch.Tensor, threshold: int = BACKGROUND_THRESHOLD) -> bool:
    return img.float().mean().item() > threshold


class _StainNormNumpy(nn.Module):
    """
    Standard CPU-based stain normalization.
    Uses Numpy backend with Hardcoded Standard Values.
    """

    def __init__(self, target_path: str = None) -> None:
        super().__init__()
        # Initialize with Numpy backend
        self.normalizer = MacenkoNormalizer(backend="numpy")

        # ✅ Set Standard H&E Reference Values (Macenko et al.)
        self.normalizer.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        # Max Concentrations
        self.normalizer.maxC = np.array([1.9705, 1.0308])

        print(f"[_StainNormNumpy] Initialized with Standard Values.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        dtype = x.dtype
        device = x.device

        # Prepare input: [B, C, H, W] -> [B, H, W, C] numpy uint8
        x_255 = (x * 255).to(torch.uint8)
        x_hwc = x_255.permute(0, 2, 3, 1).cpu().numpy()

        normalized_list = []
        for i in range(batch_size):
            img = x_hwc[i]
            try:
                # Normalize using Numpy
                norm_img, _, _ = self.normalizer.normalize(I=img, stains=True)
                normalized_list.append(norm_img)
            except Exception:
                normalized_list.append(img)

        # Reconstruct
        x_normalized = np.stack(normalized_list, axis=0)
        x_tensor = torch.from_numpy(x_normalized).to(device)
        return x_tensor.permute(0, 3, 1, 2).to(dtype) / 255.0


class _StainNormTorch(nn.Module):
    """
    Standard GPU-based stain normalization.
    Uses Torch backend with Hardcoded Standard Values.
    Run purely on GPU (No CPU offloading).
    """

    def __init__(self, target_path: str = None) -> None:
        super().__init__()
        # Initialize with Torch backend
        self.normalizer = MacenkoNormalizer(backend="torch")

        # ✅ Set Standard H&E Reference Values (Converted to Float Tensors)
        HERef_np = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        maxC_np = np.array([1.9705, 1.0308])

        self.normalizer.HERef = torch.from_numpy(HERef_np).float()
        self.normalizer.maxC = torch.from_numpy(maxC_np).float()

        print(f"[_StainNormTorch] Initialized with Standard Values.")

    def _sync_to_device(self, device: torch.device) -> None:
        """Ensure reference vectors are on the correct GPU."""
        try:
            if self.normalizer.HERef.device != device:
                self.normalizer.HERef = self.normalizer.HERef.to(device)
            if self.normalizer.maxC.device != device:
                self.normalizer.maxC = self.normalizer.maxC.to(device)
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        # Sync constants to current GPU
        self._sync_to_device(device)

        # Prepare input: [B, C, H, W] -> [B, H, W, C]
        x_255 = (x * 255).to(torch.uint8)
        x_hwc = x_255.permute(0, 2, 3, 1)

        normalized_list = []

        # Disable AMP for stability
        with torch.amp.autocast(device_type="cuda", enabled=False):
            for i in range(batch_size):
                img = x_hwc[i]

                # Skip background
                if _is_background_torch(img):
                    normalized_list.append(img)
                    continue

                try:
                    # Pure GPU Normalization
                    norm_img = self.normalizer.normalize(I=img, stains=False)
                    normalized_list.append(norm_img)
                except Exception:
                    normalized_list.append(img)

        # Reconstruct
        x_normalized = torch.stack(normalized_list, dim=0)
        x_chw = x_normalized.permute(0, 3, 1, 2)

        return x_chw.to(dtype) / 255.0


class StainNormLayer(nn.Module):
    def __init__(self, target_path: str = None) -> None:
        super().__init__()

        # Initialize workers independently
        self.cpu_worker = _StainNormNumpy()
        self.gpu_worker = _StainNormTorch()

        print(f"[StainNormLayer] Mode: Standard Values (CPU/GPU Separated)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ Clean Forward (No Logging)
        if x.device.type == "cuda":
            return self.gpu_worker(x)
        else:
            return self.cpu_worker(x)
