"""
Universal Stain Normalization Layer (CPU/GPU Compatible).
Uses torchstain library with automatic device handling.
"""

import os
import cv2
import torch
import torch.nn as nn
from torchstain.base.normalizers import MacenkoNormalizer


class StainNormLayer(nn.Module):
    """
    Macenko stain normalization layer that automatically adapts to CPU or GPU.
    """

    def __init__(self, target_path: str = "data/templates/template_he.png") -> None:
        super().__init__()
        self.target_path = target_path
        # ใช้ backend='torch' เพื่อให้รองรับ Tensor และ CUDA
        self.normalizer = MacenkoNormalizer(backend="torch")
        self._fitted = False
        self._passthrough = False

        # Attempt to fit on initialization
        self._fit_to_target()

    def _fit_to_target(self) -> None:
        """Fit normalizer to a reference H&E image."""
        if not os.path.exists(self.target_path):
            print(
                f"[StainNormLayer] WARNING: Target image not found: {self.target_path}"
            )
            self._passthrough = True
            return

        # Load reference image
        target_bgr = cv2.imread(self.target_path)
        if target_bgr is None:
            print(f"[StainNormLayer] WARNING: Failed to load image: {self.target_path}")
            self._passthrough = True
            return

        # Convert to RGB
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

        # เริ่มต้น Fit บน CPU ไปก่อน (เดี๋ยวตอนใช้จริงค่อยย้ายไป GPU)
        target_tensor = torch.from_numpy(target_rgb).to(torch.uint8)

        try:
            self.normalizer.fit(target_tensor)
            self._fitted = True
            print(f"[StainNormLayer] Fitted to target: {self.target_path}")
        except Exception as e:
            print(f"[StainNormLayer] Fit Error: {e}")
            self._passthrough = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W] range [0, 1]
        """
        # Passthrough conditions
        if self._passthrough or not self._fitted:
            return x

        # 1. เช็คว่า Input อยู่ที่ไหน (CPU หรือ GPU?)
        device = x.device

        # 2. ย้ายค่าสถิติ (Matrix) ของ Normalizer ไปหาอุปกรณ์นั้นๆ
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
            # ถ้า Error เรื่องย้าย Device ให้ปริ้นบอกแล้วคืนค่าเดิม
            print(f"[StainNormLayer] Device sync error: {e}")
            return x

        batch_size = x.shape[0]

        # Scale [0, 1] -> [0, 255] and convert to uint8
        x_255 = (x * 255).to(torch.uint8)

        # Permute: [B, C, H, W] -> [B, H, W, C]
        x_hwc = x_255.permute(0, 2, 3, 1)

        normalized_list = []
        for i in range(batch_size):
            img = x_hwc[i]
            try:
                # Normalize (ตอนนี้ทั้ง img และ matrix อยู่ Device เดียวกันแล้ว)
                norm_img, _, _ = self.normalizer.normalize(I=img, stains=True)
                normalized_list.append(norm_img)
            except Exception as e:
                # ปริ้น Error แค่ครั้งแรกครั้งเดียว
                if not hasattr(self, "_logged_error"):
                    print(f"[StainNormLayer] Normalization failed: {e}")
                    self._logged_error = True
                normalized_list.append(img)

        # Stack and Reconstruct
        x_normalized = torch.stack(normalized_list, dim=0)
        x_chw = x_normalized.permute(0, 3, 1, 2)

        # Scale back to [0, 1]
        x_out = x_chw.float() / 255.0

        return x_out
