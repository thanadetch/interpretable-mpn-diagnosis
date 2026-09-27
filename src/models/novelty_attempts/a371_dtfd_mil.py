"""a371 — DTFD-MIL (Zhang et al., CVPR 2022), "DTFD-MIL: Double-Tier Feature Distillation
Multiple Instance Learning for Histopathology Whole Slide Image Classification".

Thin adapter that runs the project's existing `src/models/standard_dtfd.py` implementation through
the `--novelty_id` plugin path. The fourth standard published MIL baseline with zero runs on the
grading task until now.

MECHANISM
  The bag is split into pseudo-bags; each pseudo-bag is distilled by keeping only its top-k and
  bottom-k scoring instances (MaxMin distillation); a Tier-1 gated attention aggregates each
  distilled pseudo-bag; a Tier-2 gated attention aggregates the pseudo-bag representations.

WHY THE RESULT IS INFORMATIVE EITHER WAY
  DTFD-MIL's pseudo-bag construction exists to make thousands of instances tractable and to
  manufacture extra bag-level supervision. ROI bags here hold ~44 instances (min 13), so 3
  pseudo-bags of ~15 instances distilled to top-1 + bottom-1 keeps **6 of 44 patches**. If it
  loses to ABMIL, that is the same theme this project already measured for ReMix and PseMix —
  slide-scale machinery does not transfer to ROI scale — and it is a citable comparison rather
  than an omission.

DEVIATIONS (honest)
  - REGRESSION HEAD: num_classes=1 with this project's ordinal target, not the paper's
    cross-entropy. The two-tier structure and MaxMin distillation are unchanged.
  - The paper's Tier-1 pseudo-bag auxiliary loss cannot be applied: the trainer's contract calls
    `model(features)` with a single regression loss and has no auxiliary-loss hook. Only the
    forward architecture is compared. Report accordingly.
  - `num_pseudo_bags=3, distill_k=1` are the defaults already in `standard_dtfd.py`.

Deterministic at eval.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..standard_dtfd import StandardDTFDMIL


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, proj_dim: int = 512,
                 hidden_dim: int = 128, dropout: float = 0.25,
                 num_pseudo_bags: int = 3, distill_k: int = 1):
        super().__init__()
        self.net = StandardDTFDMIL(
            input_dim=input_dim, num_classes=num_classes, proj_dim=proj_dim,
            hidden_dim=hidden_dim, dropout=dropout,
            num_pseudo_bags=num_pseudo_bags, distill_k=distill_k)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        out = self.net(features, return_attention)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.view(-1), None, None


KWARGS = dict(input_dim=1280, num_classes=1)
