"""a131 — bone-aware diffuse fibrosis pool with a CALIBRATED affine readout.

a129 underperformed a96 because its affine head was neutral-initialised (w_s=1,
c_s=1.5). a131 = a129's bone-aware diffuse fibrosis pool but with the affine head
warm-calibrated from the per-grade anchor span (like a96): w_s=3/(a_max-a_min),
c_s=-w_s*a_min, where a_g=<P_g, axis>. This fairly tests whether bone-avoidance
can be ADDED to the seed-2-winning a96 mechanism WITHOUT losing grade.

Ablation a132 = bone_aware=False (= calibrated diffuse fibrosis projection ~ a96).
Reads 1282-d [virchow2 | bone | fibrosis]. forward reads ONLY 'features'.
"""
from __future__ import annotations
from pathlib import Path
import torch
from .a129_boneaware_fibrosis_pool import Model as _Base

_PROTO = Path(__file__).resolve().parents[3] / "data" / "prototypes_virchow2_reti_train_seed2.pt"


class Model(_Base):
    def __init__(self, input_dim=1280, num_classes=1, bone_aware=True, warm_start=True):
        super().__init__(input_dim=input_dim, num_classes=num_classes, bone_aware=bone_aware, warm_start=warm_start)
        # calibrate affine from per-grade anchor span <P_g, v> (init-only, train-only)
        try:
            blob = torch.load(_PROTO, map_location="cpu", weights_only=False)
            v0 = blob["axis"].float().view(-1)
            if v0.numel() == input_dim:
                v0 = v0 / v0.norm().clamp(min=1e-8)
                protos = blob["prototypes"]
                anchors = torch.tensor([float(protos[g].float().view(-1) @ v0) for g in range(4)])
                a_min, a_max = float(anchors.min()), float(anchors.max())
                span = max(a_max - a_min, 1e-6)
                with torch.no_grad():
                    self.w_s.fill_(3.0 / span)
                    self.c_s.fill_(-(3.0 / span) * a_min)
        except Exception:
            pass


KWARGS = dict(input_dim=1280, num_classes=1, bone_aware=True, warm_start=True)
