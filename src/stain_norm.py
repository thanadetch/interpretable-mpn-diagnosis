"""
Universal Stain Normalization Layer (CPU/GPU Compatible).
Uses Strategy Pattern to separate CPU (Numpy) and GPU (Torch) backends.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchstain.base.normalizers import MacenkoNormalizer


class _StainNormNumpy(nn.Module):
    """
    CPU-based stain normalization using Numpy backend.
    This is more stable and reliable for CPU tensors.
    """

    def __init__(self, target_path: str) -> None:
        super().__init__()
        self.target_path = target_path
        self.normalizer = MacenkoNormalizer(backend="numpy")
        self._fitted = False
        self._passthrough = False
        self._fit_to_target()

    def _fit_to_target(self) -> None:
        """Fit normalizer to a reference H&E image."""
        if not os.path.exists(self.target_path):
            print(
                f"[_StainNormNumpy] WARNING: Target image not found: {self.target_path}"
            )
            self._passthrough = True
            return

        target_bgr = cv2.imread(self.target_path)
        if target_bgr is None:
            print(
                f"[_StainNormNumpy] WARNING: Failed to load image: {self.target_path}"
            )
            self._passthrough = True
            return

        # Convert to RGB (numpy array)
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

        try:
            self.normalizer.fit(target_rgb)
            self._fitted = True
            print(f"[_StainNormNumpy] Fitted to target: {self.target_path}")
        except Exception as e:
            print(f"[_StainNormNumpy] Fit Error: {e}")
            self._passthrough = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W] range [0, 1] on CPU
        Returns:
            Normalized tensor [B, C, H, W] range [0, 1]
        """
        if self._passthrough or not self._fitted:
            return x

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Scale [0, 1] -> [0, 255] and convert to uint8 numpy
        x_255 = (x * 255).to(torch.uint8)

        # Permute: [B, C, H, W] -> [B, H, W, C] and convert to numpy
        x_hwc = x_255.permute(0, 2, 3, 1).cpu().numpy()

        normalized_list = []
        for i in range(batch_size):
            img = x_hwc[i]  # [H, W, C] numpy array
            try:
                norm_img, _, _ = self.normalizer.normalize(I=img, stains=True)
                normalized_list.append(norm_img)
            except Exception as e:
                if not hasattr(self, "_logged_error"):
                    print(f"[_StainNormNumpy] Normalization failed: {e}")
                    self._logged_error = True
                normalized_list.append(img)

        # Stack and convert back to tensor
        x_normalized = np.stack(normalized_list, axis=0)  # [B, H, W, C]
        x_tensor = torch.from_numpy(x_normalized).to(device)

        # Permute: [B, H, W, C] -> [B, C, H, W]
        x_chw = x_tensor.permute(0, 3, 1, 2)

        # Scale back to [0, 1]
        x_out = x_chw.to(dtype) / 255.0

        return x_out


class _StainNormTorch(nn.Module):
    """
    GPU-based stain normalization using Torch backend.
    Includes fixes for A100 GPU quirks (autocast disabled, stains=False).
    """

    def __init__(self, target_path: str) -> None:
        super().__init__()
        self.target_path = target_path
        self.normalizer = MacenkoNormalizer(backend="torch")
        self._fitted = False
        self._passthrough = False
        self._fit_to_target()

    def _fit_to_target(self) -> None:
        """Fit normalizer to a reference H&E image on CPU."""
        if not os.path.exists(self.target_path):
            print(
                f"[_StainNormTorch] WARNING: Target image not found: {self.target_path}"
            )
            self._passthrough = True
            return

        target_bgr = cv2.imread(self.target_path)
        if target_bgr is None:
            print(
                f"[_StainNormTorch] WARNING: Failed to load image: {self.target_path}"
            )
            self._passthrough = True
            return

        # Convert to RGB tensor (fit on CPU first)
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        target_tensor = torch.from_numpy(target_rgb).to(torch.uint8)

        try:
            self.normalizer.fit(target_tensor)
            self._fitted = True
            print(f"[_StainNormTorch] Fitted to target: {self.target_path}")
        except Exception as e:
            print(f"[_StainNormTorch] Fit Error: {e}")
            self._passthrough = True

    def _sync_device(self, device: torch.device) -> None:
        """Move normalizer internal statistics to the target device."""
        try:
            if hasattr(self.normalizer, "HERef") and isinstance(
                self.normalizer.HERef, torch.Tensor
            ):
                if self.normalizer.HERef.device != device:
                    self.normalizer.HERef = self.normalizer.HERef.to(device)

            if hasattr(self.normalizer, "maxC") and isinstance(
                self.normalizer.maxC, torch.Tensor
            ):
                if self.normalizer.maxC.device != device:
                    self.normalizer.maxC = self.normalizer.maxC.to(device)
        except Exception as e:
            print(f"[_StainNormTorch] Device sync error: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W] range [0, 1] on CUDA
        Returns:
            Normalized tensor [B, C, H, W] range [0, 1]
        """
        if self._passthrough or not self._fitted:
            return x

        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        # Sync normalizer stats to the same device
        try:
            self._sync_device(device)
        except Exception:
            return x  # Fallback on device sync error

        # Scale [0, 1] -> [0, 255] and convert to uint8
        x_255 = (x * 255).to(torch.uint8)

        # Permute: [B, C, H, W] -> [B, H, W, C]
        x_hwc = x_255.permute(0, 2, 3, 1)

        normalized_list = []

        # Disable autocast to prevent linalg_eigh errors on A100
        with torch.amp.autocast(device_type="cuda", enabled=False):
            for i in range(batch_size):
                img = x_hwc[i]  # [H, W, C] torch tensor
                try:
                    # Use stains=False to prevent device mismatch bugs
                    norm_img = self.normalizer.normalize(I=img, stains=False)
                    normalized_list.append(norm_img)
                except Exception as e:
                    if not hasattr(self, "_logged_error"):
                        print(f"[_StainNormTorch] Normalization failed: {e}")
                        self._logged_error = True
                    normalized_list.append(img)

        # Stack and Reconstruct
        x_normalized = torch.stack(normalized_list, dim=0)

        # Permute: [B, H, W, C] -> [B, C, H, W]
        x_chw = x_normalized.permute(0, 3, 1, 2)

        # Scale back to [0, 1]
        x_out = x_chw.to(dtype) / 255.0

        return x_out


class StainNormLayer(nn.Module):
    """
    Facade for stain normalization that delegates to CPU or GPU workers.
    Automatically selects the appropriate backend based on input device.
    """

    def __init__(self, target_path: str = "data/templates/template_he.png") -> None:
        super().__init__()
        self.target_path = target_path

        # Initialize both workers
        self.cpu_worker = _StainNormNumpy(target_path)
        self.gpu_worker = _StainNormTorch(target_path)

        print(f"[StainNormLayer] Initialized with CPU and GPU workers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W] range [0, 1]
        Returns:
            Stain-normalized tensor [B, C, H, W] range [0, 1]
        """
        if x.device.type == "cuda":
            return self.gpu_worker(x)
        else:
            return self.cpu_worker(x)
