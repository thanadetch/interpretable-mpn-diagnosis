"""bdq_mil — Bulk-Deflated Query MIL (BDQ; base module for a390-a394).

THE MECHANISM WAS DERIVED FROM THIS PROJECT'S OWN FORENSICS (2026-08-08), not imported
from a paper. The FC-MIL control family left exactly one surviving fact:

    a347  query = bag centroid              -> WORST cell in the family (G1 61 vs 82)
    a342  query = conditioned, non-centroid -> dual-gate pass
    a348  query = WRONG ROI's field         -> dual-gate pass anyway
    a361  query = pure noise, non-centroid  -> passes on uni2
    a360  query = one learned static vector -> dual-gate pass on uni2 (+.020 test, above noise)

Conditioning was falsified; what mattered every time was that the scoring direction is NOT
the bag's bulk direction. BDQ turns that observation into an explicit, per-bag mechanism:

    h_i  = Bottleneck(x_i)                          # [N, H], as ASGAP
    c    = mean_i h_i ;  c_hat = c / ||c||          # the bag's BULK direction
    P_b  = I - eps * c_hat c_hat^T                  # per-bag deflation projector
    e_i  = < q , P_b h_i > / sqrt(H)                # scoring: bulk component REMOVED
    a    = entmax_1.5(e)                            # sparse pooling kept (G1 = sparse evidence)
    z    = sum_i a_i h_i                            # VALUES are the ORIGINAL h_i
    y    = Linear(z)

Why the bulk direction is nuisance here (all measured in this project): per-patch feature
magnitude tracks tissue-vs-background/bone, not grade (Spearman ~0, the a81 diagnostic); the
bag mean is dominated by whatever tissue is most ABUNDANT, while the WHO criterion is carried
by fibres that may occupy few patches; and anisotropic noise ONLY in the subspace orthogonal
to grade directions helped titan (subspace_noise) — same geometry, applied as augmentation.

Because entmax/softmax are shift-invariant, SUBTRACTING the centroid from the scores is a
no-op (<q, h_i - c> shifts every logit equally). Deflation is the smallest operation that
actually changes the attention: it removes the bulk COMPONENT of each instance, which differs
per instance. This is also why the mechanism is not "centering" and has no direct prior art;
the statistical roots are Schur-complement deflation (sparse PCA) and nuisance projection
(PeDecURe), neither per-bag nor in MIL.

DESIGN GUARDS (each tied to a measured failure mode):
  * scoring is deflated, VALUES are not     -> diffuse G3 bags keep their representation
  * no instance-instance mixing             -> the ctx-family's G1 collapse (75->47, 55->12)
                                               cannot recur by construction
  * sparse entmax kept                      -> the ASGAP G1/G2 advantage is retained
  * eps=0 recovers the plain static-query model (a394 control, identical architecture)

Fewer parameters than ASGAP (no gated V/U/W attention MLPs): ~164K vs 197K at D=1280.

HONEST RISK: in a uniformly fibrotic G3 bag the fibrosis signal may BE the bulk direction,
so full deflation could misdirect attention there; a391 (eps=0.5) hedges, and per-grade
recalls decide. Judged against BOTH the ASGAP/ABMIL baselines AND the a360/a361 controls.

Permutation-invariant, bag-size-invariant, deterministic at eval, MPS-safe (matmul/norm only).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect

DEFLATE_MODES = ("mean", "rank2")
POOLS = ("entmax", "softmax")


def _pc1(hc: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """Top principal direction of the CENTERED instances via deterministic power iteration."""
    v = torch.ones(hc.shape[1], device=hc.device, dtype=hc.dtype)
    v = v / v.norm().clamp(min=1e-8)
    for _ in range(iters):
        v = hc.t() @ (hc @ v)
        v = v / v.norm().clamp(min=1e-8)
    return v


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, eps: float = 1.0, deflate: str = "mean",
                 pool: str = "entmax", alpha: float = 1.5):
        super().__init__()
        if deflate not in DEFLATE_MODES:
            raise ValueError(f"deflate must be one of {DEFLATE_MODES}")
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}")
        self.eps = float(eps)
        self.deflate = deflate
        self.pool = pool
        self.alpha = float(alpha)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        h = self.bottleneck(features)                              # [N, H]

        hs = h
        if self.eps > 0.0:
            c = h.mean(dim=0)
            cn = c.norm().clamp(min=1e-8)
            c_hat = c / cn
            # remove the bulk component of every instance from the SCORING copy only
            hs = h - self.eps * torch.outer(h @ c_hat, c_hat)
            if self.deflate == "rank2":
                v = _pc1(h - c.unsqueeze(0))
                hs = hs - self.eps * torch.outer(hs @ v, v)

        e = (hs @ self.query) / (self.hidden_dim ** 0.5)           # [N]
        a = torch.softmax(e, dim=0) if self.pool == "softmax" else entmax_bisect(e, self.alpha)
        z = torch.mv(h.t(), a)                                     # values: ORIGINAL h
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, eps=1.0, deflate="mean", pool="entmax")
