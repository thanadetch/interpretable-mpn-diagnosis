"""
Universal Stain Normalization Layer (CPU/GPU Compatible).
Uses Strategy Pattern to separate CPU (Numpy) and GPU (Torch) backends.

PERFORMANCE FIX:
Implements 'Optimistic GPU Execution with CPU Fallback'.
1. Tries to normalize on GPU first (Fastest).
2. If A100 instability occurs (linalg error), falls back to CPU for that specific patch.
"""

import os
import cv2
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
    Standard CPU-based stain normalization using Numpy backend.
    """

    def __init__(self, target_path: str) -> None:
        super().__init__()
        self.target_path = target_path
        self.normalizer = MacenkoNormalizer(backend="numpy")
        self._fitted = False
        self._fit_to_target()

    def _fit_to_target(self) -> None:
        if not os.path.exists(self.target_path):
            # print(f"[_StainNormNumpy] Target not found: {self.target_path}")
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
            # No explicit background check here to keep logic consistent
            try:
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
    High-Performance GPU Stain Normalization with Fallback.
    """

    def __init__(self, target_path: str) -> None:
        super().__init__()
        self.target_path = target_path

        # Initialize TWO normalizers
        self.gpu_normalizer = MacenkoNormalizer(backend="torch")  # Primary (Fast)
        self.cpu_normalizer = MacenkoNormalizer(backend="numpy")  # Fallback (Stable)

        self._fitted = False
        self._fit_to_target()

    def _fit_to_target(self) -> None:
        """Fit on CPU (stable), then sync stats to GPU normalizer."""
        if not os.path.exists(self.target_path):
            print(f"[_StainNormTorch] Target not found: {self.target_path}")
            return

        target_bgr = cv2.imread(self.target_path)
        if target_bgr is None:
            print(f"[_StainNormTorch] Failed to load: {self.target_path}")
            return

        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

        try:
            # 1. Fit CPU Normalizer (Numpy) - Guaranteed to work
            self.cpu_normalizer.fit(target_rgb)

            # 2. Fit GPU Normalizer (Torch) - using data from CPU fit to be safe
            # Convert target to tensor for torch backend
            target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1)  # C,H,W
            self.gpu_normalizer.HERef = torch.from_numpy(self.cpu_normalizer.HERef).float()
            self.gpu_normalizer.maxC = torch.from_numpy(self.cpu_normalizer.maxC).float()

            self._fitted = True
            print(f"[_StainNormTorch] Successfully initialized (Hybrid Mode)")

        except Exception as e:
            print(f"[_StainNormTorch] Fit Error: {e}")

    def _sync_to_device(self, device: torch.device) -> None:
        """Ensure GPU normalizer stats are on the correct device."""
        if not self._fitted:
            return
        try:
            if self.gpu_normalizer.HERef.device != device:
                self.gpu_normalizer.HERef = self.gpu_normalizer.HERef.to(device)
            if self.gpu_normalizer.maxC.device != device:
                self.gpu_normalizer.maxC = self.gpu_normalizer.maxC.to(device)
        except Exception:
            pass

    def _normalize_on_cpu_fallback(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Fallback function: Execute on CPU using Numpy backend."""
        try:
            # Move to CPU numpy
            img_np = img_tensor.cpu().numpy().astype(np.uint8)
            # Normalize
            norm_np, _, _ = self.cpu_normalizer.normalize(I=img_np, stains=True)
            # Move back to GPU
            return torch.from_numpy(norm_np).to(img_tensor.device)
        except Exception:
            # If even CPU fails (e.g. empty white patch), return original
            return img_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            return x

        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        # Sync stats to current GPU
        self._sync_to_device(device)

        # Prepare input: [B, H, W, C]
        x_255 = (x * 255).to(torch.uint8)
        x_hwc = x_255.permute(0, 2, 3, 1)

        normalized_list = []

        # Disable AMP for stability during SVD/Eigh on GPU
        with torch.amp.autocast(device_type="cuda", enabled=False):
            for i in range(batch_size):
                img = x_hwc[i]

                # Background check (Optional: Skip calculation if white)
                if _is_background_torch(img):
                    normalized_list.append(img)
                    continue

                # --- 1. Try GPU (Fast Path) ---
                try:
                    # stains=False returns tensor [H,W,C]
                    norm_img = self.gpu_normalizer.normalize(I=img, stains=False)
                    normalized_list.append(norm_img)

                # --- 2. Catch A100/Math Errors ---
                except (RuntimeError, ValueError) as e:
                    # e.g. "linalg.eigh failed", "kthvalue", etc.
                    # Fallback to CPU for this single image
                    norm_img = self._normalize_on_cpu_fallback(img)
                    normalized_list.append(norm_img)

                except Exception:
                    # Any other error -> use original
                    normalized_list.append(img)

        # Reconstruct
        x_normalized = torch.stack(normalized_list, dim=0)
        x_chw = x_normalized.permute(0, 3, 1, 2)

        return x_chw.to(dtype) / 255.0


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
