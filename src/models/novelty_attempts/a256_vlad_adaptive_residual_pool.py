"""a256 - VLAD with an adaptive per-cluster diagonal residual scaling. NEW family: prototype/VLAD
soft-assignment residual aggregation (NOT a single attention pool, NOT a mean, NOT an attention-shape/
temperature/sparsity tweak).

WHY THIS IS A CATEGORICALLY-DISTINCT FAMILY. The exhausted lane is first-moment, attention-shaped pools:
one relevance-weighted mean of the patches, with all the tried variations living in HOW that single
weight vector is shaped (entmax, size-temp, James-Stein shrink, entropy-gated rank-cap/redundancy,
variance-reg, robust-MAD, content-gate, mean-blend). a256 abandons the single weighted-mean bag vector
entirely. Instead it builds a VLAD descriptor: it soft-assigns each patch to K learned cluster centres
and aggregates, PER CLUSTER, the SUM OF RESIDUALS (h_i - c_k) of the patches assigned to that cluster.
The bag representation is therefore K separate per-cluster residual vectors describing the DISTRIBUTION
of patches around prototypal centres -- a higher-order descriptor of how patches deviate from learned
prototypes, not a single first-moment average. This is the classic NetVLAD aggregator family.

WHAT a256 ADDS OVER a183 (plain NetVLAD, already tried). a256 inserts a learnable per-cluster diagonal
Lambda_k in R^D (one positive-ish scale per feature per cluster) that rescales the residuals BEFORE
aggregation:  scaled residual = Lambda_k (elementwise) * (h_i - c_k). Lambda is initialised to ALL ONES,
so at init a256 is EXACTLY plain NetVLAD; training can then up/down-weight specific feature dimensions
within each cluster's residual descriptor (an adaptive, learned feature-importance per prototype). This
is genuinely new learnable CONTENT carried in a VECTOR/matrix (Lambda is [K,D]), not a lone learnable
shape-scalar (the a215/a237/a238 inert-scalar failure mode is avoided -- Lambda has K*D entries with
non-flat per-dimension gradients flowing from the residuals).

MECHANISM (matmul-only, no [N,K,D] tensor).
  1. h = bottleneck(features) = Dropout(ReLU(Linear(input_dim, hidden_dim)))                 -> [N,D]
  2. soft-assign  s = softmax(assign(h), dim=1),  assign = Linear(hidden_dim, K)             -> [N,K]
  3. centres  c_k in R^D  = nn.Parameter [K,D], init randn * D**-0.5
  4. per-cluster diagonal  Lambda_k in R^D = nn.Parameter [K,D], init ONES (-> plain NetVLAD at start)
  5. algebraic VLAD identity (the [N,K,D] residual tensor is never materialised):
         V_k = sum_i s_i,k * Lambda_k * (h_i - c_k)
             = Lambda_k (elementwise) * ( (s^T h)_k  -  (sum_i s_i,k) * c_k )
     so V = Lambda * ( s.t() @ h  -  colmass[:,None] * centres ),  colmass_k = sum_i s_i,k        -> [K,D]
  6. intra-normalise each cluster row:  V_k <- F.normalize(V_k, dim=1)  (L2 over D, eps-guarded)
  7. flatten + global L2-normalise:  z = F.normalize(V.reshape(-1), dim=0)                    -> [K*D]
  8. y = classifier(z).view(-1),  classifier = Linear(K*hidden_dim, num_classes)             -> (1,)

HONEST NOTES / APPROXIMATIONS.
  - The diagonal Lambda is a PER-FEATURE per-cluster scaling only; it cannot capture cross-feature
    (off-diagonal) interactions within a cluster's residual. A full per-cluster matrix would, but that
    blows up capacity (K*D*D) on a 214-ROI cohort, so the diagonal is a deliberate low-capacity choice.
  - Lambda is unconstrained (no softplus/abs); after the residual is computed it just elementwise-scales,
    and the subsequent double L2-normalisation makes the overall sign/scale of Lambda partly redundant
    with the centre/classifier, so Lambda mainly reshapes the RELATIVE per-feature weighting within a
    cluster. This is intended, not a bug; it keeps Lambda from being a pure global gain that would go
    inert. (Init at ones means it starts as identity, i.e. plain NetVLAD.)
  - The double L2-normalisation (intra-cluster then global) is what delivers bag-size invariance; it
    removes the dependence on the absolute residual magnitudes / number of patches per cluster. It is a
    heuristic normalisation, standard in NetVLAD, not a calibrated density measure of fibrosis.
  - This is concept-free: no zero-shot text prompts, no bone/fibrosis labels, no norm-weighting
    (||h|| is never used as a relevance signal; assignment is from the learned `assign` head only).

CONSTRAINTS satisfied. Self-contained nn.Module; torch / torch.nn / torch.nn.functional only; no external
files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, softmax, matmul, elementwise mul/sub,
F.normalize, reshape -- NO linalg.solve, NO linalg.eigh, NO cdist, NO torch.median, NO eig/solve.
Permutation-invariant: every per-patch contribution enters through sums over patches (s.t() @ h and
colmass = s.sum over patches), so reordering patches leaves V, z, y unchanged. Bag-size-invariant: the
double L2-normalisation removes absolute magnitude; there is NO division by patch count, NO std, NO
/(N-1). Deterministic in eval() (Dropout off; all other ops deterministic). n=1 SAFE: with one patch,
s is [1,K], V_k = Lambda_k*(s_1,k*(h_1 - c_k)) is a single scaled weighted residual; F.normalize clamps
the L2 norm by eps BEFORE dividing (no div-by-zero), and there is no /N or /(N-1) anywhere. Capacity is
modest: K<=8 clusters, hidden_dim=128 bottleneck; descriptor dim K*hidden_dim, classifier Linear from it.
Only learnable params: bottleneck, assign, centres [K,D], Lambda [K,D], classifier -- all vectors/heads/
matrices with non-flat gradients; no lone learnable shape-scalar.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=8):
        super().__init__()
        assert K <= 8, "keep K modest for this small cohort"
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.assign = nn.Linear(hidden_dim, K)                          # soft-assignment logits -> [N,K]
        self.centres = nn.Parameter(torch.randn(K, hidden_dim) * (hidden_dim ** -0.5))  # [K,D]
        # learnable per-cluster diagonal residual scaling; init ONES => starts as plain NetVLAD.
        self.residual_scale = nn.Parameter(torch.ones(K, hidden_dim))   # Lambda [K,D]
        self.classifier = nn.Linear(K * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                   # [N,D]
        s = F.softmax(self.assign(h), dim=1)                            # [N,K] soft assignment over clusters
        # VLAD residual identity (no [N,K,D] tensor):
        #   V_k = Σ_i s_i,k * Lambda_k * (h_i - c_k) = Lambda_k * ( (s^T h)_k - (Σ_i s_i,k) c_k )
        colmass = s.sum(dim=0, keepdim=True).t()                        # [K,1] = Σ_i s_i,k per cluster
        V = (s.t() @ h) - colmass * self.centres                       # [K,D] raw per-cluster residual sum
        V = self.residual_scale * V                                     # [K,D] adaptive per-cluster diagonal scaling
        V = F.normalize(V, dim=1)                                       # intra-normalise each cluster row (eps-guarded)
        z = F.normalize(V.reshape(-1), dim=0)                           # flatten + global L2-normalise (eps-guarded)
        y = self.classifier(z).view(-1)                                 # shape (1,)
        if return_attention:
            return y, s.sum(dim=1), None                                # per-patch total assignment mass as a proxy attn
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
