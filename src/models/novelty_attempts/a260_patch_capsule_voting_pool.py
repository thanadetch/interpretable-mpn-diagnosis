"""a260 - patch capsule voting pool. NEW family: CAPSULE DYNAMIC ROUTING (routing-by-agreement,
Sabour 2017). NOT an attention-shape/temperature/sparsity/de-concentration pool, NOT VLAD/Fisher,
NOT covariance/spectral, NOT Sinkhorn/OT, NOT modern-Hopfield, NOT Perceiver/multi-query, NOT
power-mean, NOT MMD/histogram/median/quantile, NOT DPP.

WHY THIS IS A CATEGORICALLY-DISTINCT FAMILY. The exhausted lanes either (a) produce ONE relevance
weight vector over patches and take a relevance-weighted first moment (all the attention-shape /
temperature / sparsity / James-Stein / variance-reg / robust-MAD / content-gate / mean-blend
variants), or (b) build distribution descriptors against fixed-but-learned prototypes/centres in a
SINGLE feed-forward pass (VLAD / Fisher / Hopfield / OT / Perceiver readout). a260 is neither: it is
an ITERATIVE AGREEMENT process. Each patch emits a "vote" toward several output capsules; the
coupling between a patch and a capsule is then refined over a few routing iterations so that patches
whose votes AGREE (large inner product) with an emerging capsule output get their coupling increased,
while disagreeing votes get it decreased. The bag descriptor is the set of converged output capsules.
The defining mechanism -- routing-by-agreement, where the pooling weights are not produced by a head
but EMERGE from iterative vote/output agreement -- has not been tried (it is distinct from a258
modern-Hopfield, which is a fixed-point retrieval against stored patterns via energy/softmax, and from
a259 multi-query cross-attention, where queries are learned parameters that attend ONCE; here the
coupling is recomputed from the agreement between each patch's own vote and the capsule it helped
form, with no learned query and no stored-pattern memory).

MECHANISM (matmul + softmax + elementwise only; NO eigh/svd/solve/cdist/median).
  1. bottleneck:  h = Dropout(ReLU(Linear(input_dim, hidden_dim)))(features)                  -> [N, H]
  2. vote production:  a single Linear(H -> M*H_c) produces, per patch, M votes (one per output
     capsule), reshaped to votes V = [N, M, H_c]; each vote is passed through squash() along its
     H_c axis so vote magnitudes are bounded in [0,1).                                         -> [N, M, H_c]
        squash(x) = (||x||^2 / (1 + ||x||^2)) * (x / (||x|| + eps))   (length-bounding nonlinearity)
  3. iterative routing (n_iter, e.g. 3); coupling C = [N, M], softmax OVER THE M CAPSULES per patch:
        init logits b = 0  ->  C = softmax(b, dim=capsules) = 1/M uniform.
        repeat:
          (a) coupling-weighted MEAN of votes per capsule (size-invariant; magnitudes do not grow
              with N):  s_j = (Σ_i C_ij V_ij) / (Σ_i C_ij + eps)                                -> [M, H_c]
          (b) capsule output:  u_j = squash(s_j)                                                -> [M, H_c]
          (c) agreement of each patch's (detached) vote with each capsule output:
                  agree_ij = < V_ij.detach(), u_j >    (sum over H_c)                           -> [N, M]
          (d) update logits and recouple:  b <- b + agree;  C = softmax(b, dim=capsules).
     The .detach() on the vote in the agreement term is the standard Sabour-2017 routing recipe
     (routing logits are treated as a non-differentiated agreement statistic; gradients still flow to
     the votes through the weighted-mean s_j of the FINAL routing step below).
  4. final capsules:  u_j = squash( (Σ_i C_ij V_ij) / (Σ_i C_ij + eps) )                        -> [M, H_c]
  5. bag descriptor = concat of the M capsule vectors:  bag = u.reshape(M*H_c)                  -> [M*H_c]
  6. y = Linear(M*H_c, num_classes)(bag);  y.view(-1)                                           -> (1,)

HONEST NOTES / APPROXIMATIONS.
  - Size invariance is achieved by using a coupling-weighted MEAN (divide by Σ_i C_ij + eps) rather
    than the original paper's coupling-weighted SUM. The sum makes capsule magnitudes grow with the
    number of patches (a hard failure on variable bag size); the mean removes that dependence. The
    squash nonlinearity then maps each capsule into the unit ball regardless of N. This is a
    deliberate, plainly-stated deviation from Sabour 2017 for bag-size invariance; it does not change
    the routing-by-agreement mechanism.
  - The agreement term detaches the vote (standard routing). Gradients to the bottleneck and the vote
    Linear still flow through the FINAL squash(weighted-mean) in step 4 (and through every routing
    iteration's s_j, since only the vote inside the inner product is detached, not the vote inside the
    weighted mean). So the learnable content -- bottleneck matrix, vote matrix [H -> M*H_c], classifier
    -- carries real, non-flat gradients. There is NO lone learnable shape-scalar (the a215/a237/a238
    inert-scalar failure mode is avoided): all learnable parameters live in vectors/matrices/heads.
  - The number of routing iterations and the capsule count M are fixed hyperparameters, not learned.
    Few iterations (3) and few capsules (M=4) keep capacity modest for the 214-ROI cohort.
  - This is a pooled-descriptor model: the head reads the concatenated capsule VECTORS (a valid pooled
    bag representation), NOT a median/histogram/quantile-only read (the a245/246/247 collapse mode is
    avoided).
  - Concept-free: no zero-shot text prompts, no bone/fibrosis labels. NOT norm-weighted: ||h|| is never
    used as a relevance signal; coupling comes only from vote/output agreement.

CONSTRAINTS satisfied. Self-contained nn.Module; torch / torch.nn / torch.nn.functional only; no
external files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, softmax, matmul (bmm/@),
elementwise mul/add/sub/div, norm (vector L2 via sqrt of sum of squares), reshape -- NO linalg.solve,
NO linalg.eigh, NO linalg.svd, NO cdist, NO torch.median. Permutation-invariant: every per-patch
contribution enters only through SUMS over patches (Σ_i C_ij V_ij and Σ_i C_ij), and the per-patch
softmax over capsules + the agreement dot are computed independently per patch, so reordering patches
leaves the capsules, bag, and y unchanged. Bag-size-invariant: coupling-weighted MEAN (divide by the
coupling mass) + squash into the unit ball; NO division by raw patch count, NO std, NO /(N-1).
Deterministic in eval() (Dropout off; routing is a fixed deterministic recurrence). n=1 SAFE: with one
patch, V is [1, M, H_c], C is [1, M] = softmax over M capsules (valid), the weighted mean is just
V_1j (denominator Σ_i C_ij = C_1j > 0, eps-guarded anyway), squash is elementwise with an eps-guarded
norm, the agreement is a scalar dot per capsule -- no div-by-zero, no std, no /(N-1).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Length-bounding capsule nonlinearity (Sabour 2017).

    squash(x) = (||x||^2 / (1 + ||x||^2)) * (x / (||x|| + eps)), norm taken over `dim`.
    Maps any vector into the open unit ball; short vectors shrink toward 0, long vectors saturate
    toward unit length. Uses an explicit eps-guarded vector norm (no linalg op).
    """
    sq_norm = (x * x).sum(dim=dim, keepdim=True)          # ||x||^2 along dim
    norm = torch.sqrt(sq_norm + eps)                       # ||x|| (eps inside sqrt: safe, no div-by-zero)
    scale = sq_norm / (1.0 + sq_norm)                      # in [0,1)
    return scale * (x / (norm + eps))


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128,
                 num_capsules=4, capsule_dim=16, n_iter=3, dropout=0.5):
        super().__init__()
        assert num_capsules <= 8, "keep the number of capsules modest for this small cohort"
        assert n_iter >= 1
        self.M = num_capsules
        self.Hc = capsule_dim
        self.n_iter = n_iter
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # single Linear producing M votes (each capsule_dim) per patch; reshaped to [N, M, H_c].
        self.vote = nn.Linear(hidden_dim, num_capsules * capsule_dim)
        self.classifier = nn.Linear(num_capsules * capsule_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        eps = 1e-8
        h = self.bottleneck(features)                                  # [N, H]
        N = h.shape[0]
        # vote production: [N, M*H_c] -> [N, M, H_c], then squash each vote along H_c.
        V = self.vote(h).view(N, self.M, self.Hc)                      # [N, M, H_c]
        V = _squash(V, dim=-1, eps=eps)                                # bounded votes
        V_det = V.detach()                                             # for the agreement statistic (Sabour routing)

        # iterative routing-by-agreement. logits b: [N, M]; coupling C = softmax over the M capsules.
        b = torch.zeros(N, self.M, device=h.device, dtype=h.dtype)     # init -> uniform coupling 1/M
        for it in range(self.n_iter):
            C = F.softmax(b, dim=1)                                     # [N, M] per-patch over capsules
            # coupling-weighted MEAN of votes per capsule (size-invariant): s_j = Σ_i C_ij V_ij / Σ_i C_ij
            # weighted sum: [M, H_c] = C^T (over patches) applied to V; use einsum-free matmul via broadcasting.
            weighted = (C.unsqueeze(-1) * V).sum(dim=0)                # [M, H_c] = Σ_i C_ij V_ij
            mass = C.sum(dim=0).unsqueeze(-1)                          # [M, 1]   = Σ_i C_ij
            s = weighted / (mass + eps)                                # [M, H_c] coupling-weighted mean
            u = _squash(s, dim=-1, eps=eps)                            # [M, H_c] capsule outputs
            if it < self.n_iter - 1:
                # agreement of each (detached) vote with each capsule output: Σ_{H_c} V_det_ij * u_j
                agree = (V_det * u.unsqueeze(0)).sum(dim=-1)           # [N, M]
                b = b + agree                                          # recouple

        # final capsules using the last coupling (recompute C from final logits for consistency).
        C = F.softmax(b, dim=1)                                        # [N, M]
        weighted = (C.unsqueeze(-1) * V).sum(dim=0)                    # [M, H_c]
        mass = C.sum(dim=0).unsqueeze(-1)                              # [M, 1]
        u = _squash(weighted / (mass + eps), dim=-1, eps=eps)         # [M, H_c] final capsule outputs

        bag = u.reshape(-1)                                            # [M*H_c] pooled bag descriptor
        y = self.classifier(bag).view(-1)                             # (1,)
        if return_attention:
            # no patch-level attention vector available for capsule routing; return None.
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
