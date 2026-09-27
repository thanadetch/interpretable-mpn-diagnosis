"""a263 - Mixture-of-Experts POOLING with per-expert classifier heads and a content gate that
mixes PREDICTIONS (not pools). NEW family: MoE pooling (multiple heterogeneous pooling operators,
each with its OWN classifier head, combined by a per-bag content gate over the K expert predictions).

WHY THIS IS A CATEGORICALLY-DISTINCT FAMILY. The exhausted attention lane produces ONE bag vector via a
single relevance-weighted mean and then varies HOW that single weight vector is shaped (entmax, size-temp,
James-Stein, entropy-gated rank-cap/redundancy, variance-reg, robust-MAD, content-gate, mean-blend). a263
does NOT search for one good pooled vector. It runs K=3 STRUCTURALLY DIFFERENT pooling operators in
parallel -- gated-attention pool, mean pool, max pool -- each producing its own pooled vector, and gives
EACH its OWN independent Linear(H->num_classes) head. So there are K independent regressors that read the
bag through K different lenses (a first-moment relevance-weighted view, an unweighted first-moment view,
and an extreme/order-statistic view). A per-bag content gate (softmax over a small MLP read off the bag
mean) then forms a CONVEX COMBINATION of the K scalar PREDICTIONS. This is the mixture-of-experts pooling
family: gating over predictions of heterogeneous pooling experts.

HOW a263 DIFFERS FROM a243 (content-gate, already tried). a243 is a SINGLE pooled vector formed by a
content gate that blends attention-pool vs mean-pool BEFORE the (single) head -- it gates the POOLS and
has one regressor. a263 gates the PREDICTIONS: each of K=3 experts pools independently AND regresses
independently with its own head, and the gate mixes the K resulting scalars. Gating predictions of
independently-headed experts (vs gating two pooled vectors into one shared head) is the defining MoE
distinction, and a263 also adds a third, order-statistic (max) expert that a243 never had.

MECHANISM (matmul / elementwise / softmax only; no [N,K,D] tensor, no forbidden ops).
  1. Bottleneck:  h = Dropout(ReLU(Linear(input_dim -> H)))(features)                          -> [N,H]
  2. K=3 heterogeneous expert pooled vectors:
       (a) gated-attention pool (Ilse 2018 gated attention):
             gate_pre = tanh(attn_V(h)) * sigmoid(attn_U(h))                                    -> [N, A]
             e        = attn_W(gate_pre).squeeze(-1)                                            -> [N]
             a        = softmax(e, dim=0)                                                       -> [N]
             z_attn   = h.t() @ a   (= weighted mean of patch features, a matrix-vector product) -> [H]
       (b) z_mean = h.mean(0)                                                                   -> [H]
       (c) z_max  = h.max(0).values                                                             -> [H]
  3. Each expert has its OWN independent head:  logit_k = head_k(z_k)  (Linear(H -> num_classes))
       stack the K expert logits                                                                -> [K, num_classes]
  4. Gating descriptor = h.mean(0) (bag content, order/size-invariant)                          -> [H]
       gate = softmax(gate_MLP(descriptor), dim=0)   (gate_MLP: H -> Hg -> K)                   -> [K]
  5. Final logit = sum_k gate_k * logit_k   (convex combination of the K expert PREDICTIONS)    -> [num_classes]
  6. y = final_logit.view(-1)  (shape (1,) for num_classes=1).  Trainer rounds+clips [0,3] at
     eval time; a263 returns the RAW scalar logit (no rounding/clipping inside the module).

HONEST NOTES / APPROXIMATIONS.
  - The three experts are not learned to be diverse by any explicit penalty; their diversity comes purely
    from the fixed structural difference of the pooling operators (relevance-weighted mean vs plain mean
    vs coordinate-wise max). The gate can collapse onto one expert if that is best on the (tiny) training
    set; that is a legitimate MoE outcome, not a bug, and it does NOT make any learnable scalar inert
    because every learnable parameter here lives in a VECTOR/MATRIX/head (the gate_MLP, the attention
    V/U/W matrices, the three independent heads, the bottleneck) -- there is NO lone learnable shape-scalar
    (the a215/a237/a238 inert-scalar failure mode is avoided).
  - The gate reads only the bag MEAN (a first-moment summary), so its routing signal is coarse; it cannot
    route on higher-order bag statistics. This is a deliberate low-capacity choice for a 214-ROI cohort.
  - z_max (coordinate-wise max over patches) is an order statistic; it is bag-size-invariant (max of more
    patches is still a single max) but it is the only non-smooth expert. With num_classes=1 every head is a
    tiny Linear(H->1), so total capacity stays modest.
  - Concept-free: no zero-shot text prompts, no bone/fibrosis labels. NOT norm-weighted: ||h|| is never
    used as a relevance signal; attention weights come from the learned gated-attention head only, and the
    gate comes from the learned gate_MLP only.

CONSTRAINTS satisfied. Self-contained nn.Module; torch / torch.nn / torch.nn.functional only; no external
files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, tanh, sigmoid, softmax, matmul (h.t() @ a),
mean, max, elementwise mul, sum, stack, view -- NO linalg.solve / eigh / svd, NO cdist, NO torch.median,
NO eig/solve. Permutation-invariant: softmax-attention pool, mean pool and max pool are all order-agnostic;
the gate reads the order-agnostic mean; expert and gate ordering is fixed. Bag-size-invariant: the
attention pool is a softmax-weighted mean over patches, z_mean is a mean over patches, z_max is a max over
patches -- all N-invariant; there are NO raw sums over patches feeding the output, NO std, NO /(N-1).
Deterministic in eval() (Dropout off; all other ops deterministic). n=1 SAFE: with one patch h is [1,H],
softmax([e_1]) = [1.0] so z_attn = h_1; z_mean = h_1; z_max = h_1; each head maps h_1 -> a scalar; the gate
reads h_1 -> a well-defined K-simplex weight; final logit is a convex combination of 3 finite scalars. No
div-by-zero, no /(N-1), no std, no kNN-with-k>N, no graph-with-0-edges issue anywhere.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, attn_dim=64,
                 gate_hidden=32, dropout=0.5):
        super().__init__()
        self.K = 3  # number of heterogeneous pooling experts: {attn, mean, max}
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

        # Expert (a): Ilse-2018 gated-attention pooling head (vectors/matrices, no lone scalar).
        self.attn_V = nn.Linear(hidden_dim, attn_dim)   # tanh branch
        self.attn_U = nn.Linear(hidden_dim, attn_dim)   # sigmoid gate branch
        self.attn_W = nn.Linear(attn_dim, 1)            # -> per-patch score

        # K independent classifier heads, one per expert pooled vector.
        self.head_attn = nn.Linear(hidden_dim, num_classes)
        self.head_mean = nn.Linear(hidden_dim, num_classes)
        self.head_max = nn.Linear(hidden_dim, num_classes)

        # Content gate over the K expert PREDICTIONS (reads the bag mean); 2-layer MLP -> K logits.
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden), nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, self.K))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                   # [N,H]

        # --- Expert (a): gated-attention pool ---------------------------------------------------
        gate_pre = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))   # [N, attn_dim]
        e = self.attn_W(gate_pre).squeeze(-1)                           # [N]
        a = F.softmax(e, dim=0)                                         # [N] attention weights over patches
        z_attn = h.t() @ a                                             # [H] weighted mean (matrix-vector)

        # --- Expert (b): mean pool; Expert (c): max pool ----------------------------------------
        z_mean = h.mean(dim=0)                                         # [H]
        z_max = h.max(dim=0).values                                    # [H]

        # --- K independent expert predictions ---------------------------------------------------
        logit_attn = self.head_attn(z_attn)                            # [num_classes]
        logit_mean = self.head_mean(z_mean)                            # [num_classes]
        logit_max = self.head_max(z_max)                               # [num_classes]
        logits = torch.stack([logit_attn, logit_mean, logit_max], dim=0)  # [K, num_classes]

        # --- Content gate over PREDICTIONS (reads order/size-invariant bag mean) -----------------
        descriptor = h.mean(dim=0)                                     # [H]
        gate = F.softmax(self.gate_mlp(descriptor), dim=0)             # [K] convex weights over experts

        # --- Convex combination of expert predictions -------------------------------------------
        final_logit = (gate.unsqueeze(-1) * logits).sum(dim=0)         # [num_classes]
        y = final_logit.view(-1)                                       # (num_classes,) -> (1,) for num_classes=1

        if return_attention:
            return y, a.detach(), None                                 # per-patch attention from expert (a)
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
