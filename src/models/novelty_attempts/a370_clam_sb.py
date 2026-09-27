"""a370 — CLAM-SB (Lu et al., Nature Biomedical Engineering 2021), "Data-efficient and weakly
supervised computational pathology on whole-slide images".

Thin adapter that runs the project's existing `src/models/clam.py` implementation through the
`--novelty_id` plugin path, so the most-cited MIL baseline in computational pathology finally gets
a number on the grading task alongside ABMIL and ASGAP.

DEVIATION — READ BEFORE CITING THIS ROW
  CLAM has two parts: (a) gated-attention pooling over a 512-d projection, and (b) an INSTANCE-
  LEVEL CLUSTERING loss that supervises the top-k / bottom-k attended patches with pseudo-labels.
  This trainer's contract calls `model(features)` and applies a single regression loss, with no
  hook for an auxiliary term, so **only (a) is active here**. `clam.py` does implement (b) in
  `forward_training`, but it is defined for classification and is never invoked on this path.

  Report this row as "CLAM-SB (attention branch)", not as full CLAM. It is still the right
  architectural comparison — CLAM's pooling versus ABMIL's versus ASGAP's — but the instance
  clustering regulariser is absent, which is the part that helps CLAM most on small cohorts.

Deterministic at eval, permutation- and size-invariant.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..clam import CLAM_SB


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1,
                 proj_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        self.net = CLAM_SB(input_dim=input_dim, num_classes=num_classes,
                           proj_dim=proj_dim, dropout=dropout)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        logits, attn, _ = self.net(features, return_attention=return_attention)
        return logits.view(-1), attn, None


KWARGS = dict(input_dim=1280, num_classes=1)
