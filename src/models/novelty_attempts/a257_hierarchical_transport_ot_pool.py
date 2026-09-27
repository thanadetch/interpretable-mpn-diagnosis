"""a257 - Hierarchical transport OT pooling. A TWO-STAGE assignment-then-transport bag descriptor:
soft-cluster the patches into K cluster summaries, then entropic-OT (Sinkhorn) transport those K cluster
summaries onto M learned anchors, and read the transport-weighted anchor representation as the pooled bag.

FAMILY: optimal transport (OT). This is a genuinely-DIFFERENT OT formulation from the prior Sinkhorn attempt
(a181, which Sinkhorn-assigned individual PATCHES to learned anchors with a single transport stage). a257 is
HIERARCHICAL: it inserts a soft-clustering stage BETWEEN patches and anchors, so the expensive transport runs
only on a fixed [K,M] cost matrix (K cluster summaries -> M anchors) that is completely independent of bag
size N. The pooled descriptor is the per-anchor transported readout (M*D), NOT a single first-moment
attention-weighted mean -- it is a richer, structured bag representation, but the head stays modest (M*hidden_dim
in, num_classes out; K=M=4, hidden_dim=128).

MECHANISM (exact, matmul/elementwise/softmax/normalize only; MPS-safe).
  STAGE 1 -- soft patch clustering into K cluster summaries.
    (1) h = bottleneck(features) = Dropout(ReLU(Linear(input_dim, hidden_dim)))           -> [N, D]
    (2) s_logits = normalize(h) @ normalize(cluster_centers).t()  (cosine sim to K learned centers) -> [N, K]
    (3) s = softmax(s_logits, dim=1)  (each patch is a soft membership distribution over K clusters) -> [N, K]
    (4) cluster_sum  = s.t() @ h                                                          -> [K, D]
        cluster_mass = s.sum(dim=0, keepdim=True).t()                                     -> [K, 1]
        clusters     = cluster_sum / cluster_mass.clamp(min=1e-8)   (soft-membership-weighted cluster means) -> [K, D]
    The N patches are SUMMED out here, so stage 2 sees only K fixed cluster summaries: size- and order-invariant.
  STAGE 2 -- entropic OT transport of the K clusters onto M learned anchors.
    (5) cost = -(normalize(clusters) @ normalize(anchors).t())   (negative cosine = transport cost) -> [K, M]
    (6) Kmat = exp(-cost / eps)   (Gibbs kernel)                                          -> [K, M]
    (7) symmetric Sinkhorn for n_iters, with uniform marginals (row 1/K, col 1/M):
            u = ones(K)/K
            for _ in range(n_iters):
                v = (1/M) / (Kmat.t() @ u).clamp(min=1e-8)                                -> [M]
                u = (1/K) / (Kmat   @ v).clamp(min=1e-8)                                  -> [K]
    (8) P  = u[:,None] * Kmat * v[None,:]   (entropic-OT transport plan, ~doubly-stochastic) -> [K, M]
    (9) Pn = P / P.sum(dim=0, keepdim=True).clamp(min=1e-8)   (column-normalize: each anchor's incoming mix) -> [K, M]
   (10) z  = (Pn.t() @ clusters).reshape(-1)   (per-anchor transported cluster readout, concatenated) -> [M*D]
   (11) y  = classifier(z).view(-1)   (classifier = Linear(M*hidden_dim, num_classes))    -> shape (1,)
  return_attention returns P.sum(dim=1) -- the per-cluster total transported mass [K] (NOT a per-patch map).

WHY THIS IS A NEW FAMILY (vs the exhausted attention-shape lane). There is no softmax-over-patches attention
distribution and no first-moment weighted mean of patch features being regressed. The bag is represented as a
TRANSPORT PLAN between learned cluster prototypes and learned anchors; the descriptor is the structured
per-anchor readout. It is categorically distinct from entmax / temperature / sparsity / shrinkage / variance-reg
/ robust-MAD / content-gate / mean-blend, all of which reshape one attention vector. It differs from a181 by the
hierarchical cluster stage (transport cost is [K,M], not [N,M]) and from VLAD/covariance families by using an
OT transport plan rather than residual or second-moment pooling.

HONEST NOTES / APPROXIMATIONS (stated plainly).
  - n_iters=3 Sinkhorn iterations is a TRUNCATED (not converged) approximation of the entropic-OT plan; with
    eps=0.5 and a tiny K=M=4 problem this is a coarse, deliberately under-converged transport. P is therefore
    only approximately doubly-stochastic. This is intentional (cheap, stable, low-capacity), not exact OT.
  - cluster_centers and anchors are SEPARATE learned nn.Parameters that can collapse toward each other or toward
    degenerate (near-duplicate) directions; nothing forces them to spread out. If centers collapse, stage-1
    memberships flatten toward uniform and clusters -> the global mean repeated K times; if anchors collapse,
    columns of P become near-identical and z repeats one readout M times. No diversity/orthogonality regularizer
    is imposed, so the effective descriptor rank can be lower than M*D.
  - cosine similarities (L2-normalized) drop feature MAGNITUDE on purpose: per the task, ||h|| is grade-
    uninformative, so both the clustering and the transport cost are direction-only.
  - eps, K, M, n_iters are FIXED hyperparameters (closed-form constants), not learned or theory-derived optima.
  - The only learnable content lives in VECTORS / matrices (bottleneck, cluster_centers[K,D], anchors[M,D],
    classifier) -- there is NO lone learnable shape-scalar (eps is a fixed python float), so nothing can go
    inert the way a215/a237/a238 single scalars did.

CONSTRAINTS satisfied. Single self-contained nn.Module; torch / nn / F only; no new deps. MPS-safe ops only:
matmul (@), elementwise, exp, softmax, F.normalize, sum, clamp, reshape -- NO linalg.solve/eigh, NO cdist, NO
torch.median; squared distances are not even needed (cosine via normalize+matmul). Permutation-invariant
(stage-1 sums over patches; everything after operates on the order-free [K,D] summaries). Bag-size-invariant
(N is summed/normalized out in stage 1; the [K,M] transport is independent of N; no (N-1) term, no std).
Deterministic in eval(). n=1 safe: s is [1,K] (a valid softmax row), cluster_sum=[K,D] gets that single patch's
contribution spread by its memberships, cluster_mass.clamp(min=1e-8) prevents div-by-zero for empty clusters,
and stage 2 is entirely independent of N.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
                 K=4, M=4, n_iters=3, eps=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # Learnable content lives in VECTORS/matrices (non-flat gradients) -> nothing goes inert.
        self.cluster_centers = nn.Parameter(torch.randn(K, hidden_dim))   # [K, D] stage-1 prototypes
        self.anchors = nn.Parameter(torch.randn(M, hidden_dim))           # [M, D] stage-2 OT targets
        self.classifier = nn.Linear(M * hidden_dim, num_classes)
        # FIXED closed-form constants (not nn.Parameter) -> no lone learnable scalar to go inert.
        self.K = int(K)
        self.M = int(M)
        self.n_iters = int(n_iters)
        self.eps = float(eps)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                          # [N, D]

        # STAGE 1: soft patch clustering into K cluster summaries (sums out N -> size/perm invariant).
        s_logits = F.normalize(h, dim=1) @ F.normalize(self.cluster_centers, dim=1).t()  # [N, K] cosine sims
        s = F.softmax(s_logits, dim=1)                                         # [N, K] soft memberships
        cluster_sum = s.t() @ h                                                # [K, D]
        cluster_mass = s.sum(dim=0, keepdim=True).t()                          # [K, 1]
        clusters = cluster_sum / cluster_mass.clamp(min=1e-8)                  # [K, D] cluster means

        # STAGE 2: entropic-OT (Sinkhorn) transport of K clusters onto M anchors. [K,M] is N-independent.
        cost = -(F.normalize(clusters, dim=1) @ F.normalize(self.anchors, dim=1).t())    # [K, M] neg cosine cost
        Kmat = torch.exp(-cost / self.eps)                                     # [K, M] Gibbs kernel
        u = torch.full((self.K,), 1.0 / self.K, device=h.device, dtype=h.dtype)          # [K] uniform row marginal
        for _ in range(self.n_iters):
            v = (1.0 / self.M) / (Kmat.t() @ u).clamp(min=1e-8)                # [M]
            u = (1.0 / self.K) / (Kmat @ v).clamp(min=1e-8)                    # [K]
        P = u.unsqueeze(1) * Kmat * v.unsqueeze(0)                             # [K, M] transport plan
        Pn = P / P.sum(dim=0, keepdim=True).clamp(min=1e-8)                    # [K, M] column-normalized
        z = (Pn.t() @ clusters).reshape(-1)                                    # [M*D] per-anchor readout

        y = self.classifier(z).view(-1)                                        # shape (1,)
        if return_attention:
            return y, P.sum(dim=1), None                                       # [K] per-cluster transported mass
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
