"""
Universal Stain Normalization Layer (CPU/GPU Compatible).
Uses Strategy Pattern to separate CPU (Numpy) and GPU (Torch) backends.

CRITICAL FIX FOR A100: CPU Offload strategy.
UPDATE: Removed explicit background check to force normalization attempt on all patches.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchstain.base.normalizers import MacenkoNormalizer


class _StainNormNumpy(nn.Module):
    """
    Standard CPU-based stain normalization.
    """
    def __init__(self, target_path: str) -> None:
        super().__init__()
        self.target_path = target_path
        self.normalizer = MacenkoNormalizer(backend="numpy")
        self._fitted = False
        self._fit_to_target()

    def _fit_to_target(self) -> None:
        if not os.path.exists(self.target_path):
            print(f"[_StainNormNumpy] Target not found: {self.target_path}")
            return

        target_bgr = cv2.imread(self.target_path)
        if target_bgr is None:
            return

        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        try:
            self.normalizer.fit(target_rgb)
            self._fitted = True
        except Exception as e:
            print(f"[_StainNormNumpy] Fit Error: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            return x

        batch_size = x.shape[0]
        dtype = x.dtype
        device = x.device

        # [B, C, H, W] -> [B, H, W, C] numpy uint8
        x_255 = (x * 255).to(torch.uint8)
        x_hwc = x_255.permute(0, 2, 3, 1).cpu().numpy()

        normalized_list = []
        for i in range(batch_size):
            img = x_hwc[i]

            # REMOVED: Background check
            # We try to normalize EVERYTHING now.

            try:
                norm_img, _, _ = self.normalizer.normalize(I=img, stains=True)
                normalized_list.append(norm_img)
            except Exception:
                # If math fails (e.g. pure white image), fallback to original
                normalized_list.append(img)

        # Reconstruct
        x_normalized = np.stack(normalized_list, axis=0)
        x_tensor = torch.from_numpy(x_normalized).to(device)

        return x_tensor.permute(0, 3, 1, 2).to(dtype) / 255.0


class _StainNormTorch(nn.Module):
    """
    GPU Wrapper that OFFLOADS math to CPU.
    """

    def __init__(self, target_path: str) -> None:
        super().__init__()
        self.target_path = target_path
        # Use NUMPY backend even for the GPU worker
        self.normalizer = MacenkoNormalizer(backend="numpy")
        self._fitted = False
        self._fit_to_target_cpu()

    def _fit_to_target_cpu(self) -> None:
        """Fit using Numpy on CPU (Guaranteed stability)."""
        if not os.path.exists(self.target_path):
            print(f"[_StainNormTorch] Target not found: {self.target_path}")
            return

        target_bgr = cv2.imread(self.target_path)
        if target_bgr is None:
            print(f"[_StainNormTorch] Failed to load: {self.target_path}")
            return

        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        try:
            self.normalizer.fit(target_rgb)
            self._fitted = True
            print(f"[_StainNormTorch] Successfully fitted (CPU Backend)")
        except Exception as e:
            print(f"[_StainNormTorch] Fit Error: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If fit failed, pass through
        if not self._fitted:
            return x

        batch_size = x.shape[0]
        dtype = x.dtype
        device = x.device

        # 1. OFFLOAD: Move batch to CPU & Numpy
        x_255 = (x * 255).to(torch.uint8)
        x_hwc = x_255.permute(0, 2, 3, 1).cpu().numpy()

        normalized_list = []

        for i in range(batch_size):
            img = x_hwc[i]

            # REMOVED: Background check
            # Normalized everything. If it crashes, catch block handles it.

            try:
                norm_img, _, _ = self.normalizer.normalize(I=img, stains=True)
                normalized_list.append(norm_img)
            except Exception as e:
                # Fail silently for math errors on empty patches
                normalized_list.append(img)

        # 4. UPLOAD: Stack and move back to GPU
        x_normalized = np.stack(normalized_list, axis=0)
        x_tensor = torch.from_numpy(x_normalized).to(device)

        return x_tensor.permute(0, 3, 1, 2).to(dtype) / 255.0


class StainNormLayer(nn.Module):
    def __init__(self, target_path: str = "data/templates/template_he.png") -> None:
        super().__init__()
        self.target_path = target_path
        self.cpu_worker = _StainNormNumpy(target_path)
        self.gpu_worker = _StainNormTorch(target_path)
        print(f"[StainNormLayer] Initialized. Target: {target_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            return self.gpu_worker(x)
        else:
            return self.cpu_worker(x)