"""a145 — Hybrid-Normalized Bone-Aware Diffuse Density (per-bag CENTER + GLOBAL scale).

Third arm of the normalization ablation (a143 per_bag / a144 global / a145 hybrid). Rationale
(agent recommendation): bone-ness is an ABSOLUTE tissue property (dark/dense), so the
suppression should keep the absolute magnitude — but per-ROI stain/magnification can shift
the whole score distribution. Hybrid subtracts each ROI's own mean (removes the acquisition
offset, leakage-safe) but divides by the GLOBAL std (keeps absolute bone magnitude so
bone-heavy ROIs are still recognised + avoids the per-bag tiny-std blow-up):

  bone_z = (bone − μ_bag_bone) / σ_global_bone ,  fib_z = (fib − μ_bag_fib) / σ_global_fib

Everything else identical to a143/a144 (warm fibrosis axis, λ bone-suppression, w_s/c_s
calibration). Reads data_bonefib.
"""
from __future__ import annotations
from .a143_perbag_boneaware_density import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, norm_mode="hybrid", warm_start=True)
