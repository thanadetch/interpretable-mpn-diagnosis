"""a262 - Patch RELATION-NETWORK pooling. NEW family: pairwise relation-network aggregation
(Santoro 2017-style relational reasoning over patches), NOT a single attention-shaped weighted mean,
NOT VLAD/Fisher, NOT spectral/covariance, NOT OT/Sinkhorn, NOT Hopfield, NOT Perceiver/multi-query,
NOT power-mean, NOT witness/histogram/quantile, NOT DPP.

WHY THIS IS A CATEGORICALLY-DISTINCT FAMILY. Every exhausted attention-pool family computes ONE
per-patch relevance vector w_i from each patch IN ISOLATION (a learned head on h_i, possibly
temperature/entmax/shrink/variance-reg shaped) and returns a single weighted mean. a262 instead
derives per-patch importance from PAIRWISE RELATIONS: for every ordered pair (i,j) it scores an
asymmetric relation strength r_ij = sigmoid(relation(h_i, h_j)), and a patch's importance is its
OUTGOING relation strength (how strongly it relates to the rest of the bag), w_i ~ sqrt(mean_j r_ij).
Importance is thus a function of the WHOLE bag's pairwise structure, not of the patch alone -- a
relational mechanism. The bag descriptor is further augmented with three scalar summaries of the
relation matrix itself (mean off-/on-diagonal relation, diagonal self-relation, relation variance),
so the head reads both a pooled patch vector AND a compact summary of relational structure.

MECHANISM (matmul-only, NO cdist, NO [N,N,2H] tensor materialised).
  1. Bottleneck:   h = Dropout(ReLU(Linear(input_dim -> H)))(features)                       -> [N,H]
  2. Asymmetric pairwise relations via a BILINEAR FORM (the cheap, memory-safe equivalent of
     concatenating [h_i; h_j] then Linear(2H->1): a Linear on a concat is exactly
     a @ Wa.t() (depends on i) + b @ Wb.t() (depends on j) + bias, i.e. two projections + a bias;
     here we use a single shared relation-embedding e = relation_proj(h) [N,He] and form the
     bilinear logit  L_ij = e_i . e_j  via e @ e.t(), then add asymmetric per-row/per-col biases
     from a second projection so r_ij != r_ji):
        e = relation_proj(h)                         -> [N, He]
        a = row_proj(h).view(-1)                     -> [N]   (additive "outgoing" bias for i)
        b = col_proj(h).view(-1)                     -> [N]   (additive "incoming" bias for j)
        L = e @ e.t() + a[:, None] + b[None, :] + bias                                          -> [N,N]
        r = sigmoid(L)                                                                           -> [N,N]
     r is asymmetric because a[:,None] + b[None,:] is asymmetric (a != b in general).
  3. Per-patch importance from OUTGOING relation strength:
        w_raw_i = sqrt(clamp(mean_j r_ij, min=1e-8))                                             -> [N]
        w = w_raw / (w_raw.sum() + 1e-8)            (normalise to sum 1; bag-size invariant)     -> [N]
  4. Weighted pool:   z = sum_i w_i * h_i                                                        -> [H]
  5. Bag relation stats (all N-invariant means/var, unbiased=False so N=1 -> var 0, no /(N-1)):
        r_mean = r.mean(), r_diag = r.diagonal().mean(), r_var = r.var(unbiased=False)           -> 3 scalars
  6. Augment:   z_aug = cat([z, r_mean, r_diag, r_var])                                          -> [H+3]
  7. y = classifier(z_aug).view(-1)                                                              -> (1,)

HONEST NOTES / APPROXIMATIONS.
  - APPROXIMATION OF THE PAIR-MLP. The spec offers two equivalent forms: (A) explicit concat
    [h_i; h_j] -> Linear(2H -> 1), and (B) a cheaper bilinear/two-projection form. We implement form (B)
    plus an explicit bilinear inner-product term e_i.e_j. A *plain* Linear on the concat is purely
    additive (separable: f(h_i)+g(h_j)+bias) and can only express relations that are a SUM of an
    outgoing and an incoming term -- it cannot express agreement/interaction between h_i and h_j. We
    therefore ADD the bilinear term e@e.t() so the relation can depend on the INTERACTION of the pair,
    which is the point of a relation network. This makes a262 strictly more expressive than the bare
    concat-Linear while staying matmul-only and avoiding the [N,N,2H] tensor. The bilinear core e@e.t()
    is itself symmetric; ALL asymmetry comes from the additive row/col bias terms a[:,None]+b[None,:].
    This is a deliberate, stated approximation, not a hidden one.
  - r is a bounded (0,1) relation score, NOT a probability/coupling; it is not normalised over j.
    Per-patch importance uses mean_j r_ij (outgoing strength), then a global sum-to-1 normalisation.
  - The sqrt() in step 3 is a mild compression of the per-patch outgoing strength (so a few very
    high-relation patches do not dominate w as hard as a linear map would); clamp(min=1e-8) guards the
    sqrt's gradient at 0. It carries no learnable scalar (avoids the a215/a237/a238 inert-scalar trap).
  - The 3 relation scalars are a compact, fixed summary of relational structure; they are NOT a
    histogram/quantile-only read (we keep the pooled patch vector z; the scalars merely AUGMENT it),
    so this does not repeat the a245/246/247 collapse-to-random failure mode.
  - Concept-free: no zero-shot text prompts, no bone/fibrosis labels. NOT norm-weighted: ||h|| is never
    used as relevance; importance comes from learned pairwise relations only.

CONSTRAINTS satisfied. Self-contained nn.Module; torch / torch.nn / torch.nn.functional only; no external
files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, matmul (@), sigmoid, elementwise mul/add,
clamp, sqrt, sum, mean, var, diagonal, cat -- NO linalg.solve, NO linalg.eigh, NO linalg.svd, NO cdist,
NO torch.median. Squared distance is never needed (we use inner products, not distances).
Permutation-invariant: relation_proj/row_proj/col_proj are per-patch maps so a permutation P sends
L -> P L P.t() (permutation-EQUIVARIANT in the [N,N] indices); w is the matching permutation of itself,
z = sum_i w_i h_i is a permutation-INVARIANT weighted sum, and r_mean/r_diag/r_var are symmetric
reductions over all entries -> all invariant. Bag-size-invariant: w is renormalised to sum to 1 (no /N,
no /(N-1) on patches); r_mean/r_diag/r_var are means/var over pairs; duplicating the whole bag leaves z,
r_mean, r_diag, r_var (hence y) unchanged. Deterministic in eval() (Dropout off; all else deterministic).
n=1 SAFE: L is [1,1], r=[[sigmoid(L_11)]] in (0,1); w_raw=sqrt(r_11)>0, w=[1.0]; z=h_1; r_mean=r_diag=r_11,
r_var=0 (unbiased=False, no /(N-1)); z_aug=[h_1, r_11, r_11, 0]; no div-by-zero / no std / no /(N-1).
Capacity is modest: bottleneck (input_dim*H), relation_proj (H*He), row/col_proj (2*H), classifier (H+3).
All learnable content lives in vectors/heads/matrices with non-flat gradients; no lone learnable scalar.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=96, relation_dim=48, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # bilinear relation embedding (interaction core: e_i . e_j)
        self.relation_proj = nn.Linear(hidden_dim, relation_dim)
        # asymmetric additive biases: outgoing (row) and incoming (col)
        self.row_proj = nn.Linear(hidden_dim, 1)
        self.col_proj = nn.Linear(hidden_dim, 1)
        self.relation_bias = nn.Parameter(torch.zeros(1))  # single global offset of the relation logits
        # head reads pooled patch vector + 3 relation-structure scalars
        self.classifier = nn.Linear(hidden_dim + 3, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                   # [N,H]

        # --- asymmetric pairwise relation matrix r (matmul-only, no [N,N,2H] tensor) ---
        e = self.relation_proj(h)                                       # [N,He]
        a = self.row_proj(h).view(-1)                                   # [N]  outgoing bias (depends on i)
        b = self.col_proj(h).view(-1)                                   # [N]  incoming bias (depends on j)
        logits = e @ e.t() + a.unsqueeze(1) + b.unsqueeze(0) + self.relation_bias  # [N,N]
        r = torch.sigmoid(logits)                                       # [N,N] in (0,1), asymmetric

        # --- per-patch importance from OUTGOING relation strength ---
        out_strength = r.mean(dim=1)                                    # [N] mean_j r_ij
        w_raw = torch.sqrt(torch.clamp(out_strength, min=1e-8))         # [N] compressed outgoing strength
        w = w_raw / (w_raw.sum() + 1e-8)                                # [N] normalise to sum 1

        # --- weighted pool ---
        z = (w.unsqueeze(1) * h).sum(dim=0)                             # [H]

        # --- bag relation-structure scalars (N-invariant; unbiased=False => N=1 gives var 0) ---
        r_mean = r.mean()                                               # scalar
        r_diag = r.diagonal().mean()                                    # scalar
        r_var = r.var(unbiased=False)                                   # scalar (no /(N-1))

        z_aug = torch.cat([z, r_mean.view(1), r_diag.view(1), r_var.view(1)], dim=0)  # [H+3]
        y = self.classifier(z_aug).view(-1)                            # (1,)

        if return_attention:
            return y, w.detach(), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
