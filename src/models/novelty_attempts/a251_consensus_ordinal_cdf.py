"""a251 - Consensus ordinal-CDF read over a gated-attention weighted pool.

MECHANISM (backbone unchanged, novelty on top of the pool):
  Backbone is the standard gated-attention weighted pool:
      h     = Bottleneck(features)                       # Linear(D,128)+ReLU+Dropout(0.5)
      alpha = softmax( W( tanh(V h) * sigmoid(U h) ) )    # per-patch gated-attention weights, sum to 1
      z     = sum_i alpha_i h_i                           # WEIGHTED POOL (kept as the backbone)
  NOVELTY is a VARIANCE-REDUCING ordinal read that assembles the grade prediction as the
  CONSENSUS (uniform 1/K mean) of K linear cumulative-ordinal reads of the SAME pooled vector z,
  instead of one fragile scalar -> threshold read:
      u   = z @ P.t() + d                                 # [K] K ordinal locations, P:[K,128]
      q   = u.unsqueeze(-1) * G + Q                        # [K,3] per-head/per-boundary matmul, G,Q:[K,3]
      e1  = q[...,0]
      e2  = e1 + softplus(q[...,1])                        # monotone, can never invert
      e3  = e2 + softplus(q[...,2])                        # monotone
      g_k = sigmoid(u_k-e1)+sigmoid(u_k-e2)+sigmoid(u_k-e3)  # per-head expected grade in ~[0,3]
      y   = mean_k g_k                                     # consensus, uniform 1/K (no routing scalar)

WHY NEW (vs the documented exhausted set / WALL):
  - Directly targets the WALL (val<->test QWK ~ -0.95; read-variance, not capacity, is the ceiling):
    averaging K correlated ordinal reads of one pooled vector lowers read-variance, and because each
    head has its own learned direction P[k] and gain G[k], heads de-correlate on the hard G1<->G2 band
    where a single scalar->threshold read is noisiest.
  - Distinct from a47/a89/a150 (single scalar -> 3 FIXED-Parameter thresholds + a learnable temperature
    scalar, DE29-killed inert): here the thresholds are MATMUL outputs (u*G+Q) PER HEAD and the K reads
    are AVERAGED into a consensus.
  - Distinct from a40 (fused K cross-attention QUERY bag-reps in pooling space): a251 fuses K ordinal-CDF
    READS of the SINGLE pooled vector in ordinal-head space. Distinct from multi-head-concat (averaged,
    not concatenated; ordinal cumulative reads, not attention heads).
  - The uniform 1/K mean (no routing/gating scalar) echoes the a40>a39 finding that uniform fusion is the
    safe default. NO global temperature and NO lone learnable shape-scalar: directions, gains and biases
    are all matrices; the fusion weight is a fixed constant 1/K.

CONSTRAINTS: self-contained nn.Module (torch / nn / F only); permutation- & size-invariant; deterministic
in eval(); n=1 safe (pool over the single patch, no .std/unbiased, no div-by-zero); MPS-safe ops only
(matmul/@, elementwise, softmax, sigmoid, softplus, sum/mean).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, heads=4):
        super().__init__()
        assert num_classes == 1
        self.hidden_dim = hidden_dim
        self.K = heads
        # Bottleneck (baseline-style): Linear(D,128) + ReLU + Dropout(0.5)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # Gated-attention scorer (backbone weighted pool): tanh(V h) * sigmoid(U h) -> W
        self.attV = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attU = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attW = nn.Linear(hidden_dim, 1)
        # K ordinal locations: u = z @ P.t() + d   (P:[K,128], d:[K])
        self.P = nn.Parameter(torch.empty(self.K, hidden_dim).normal_(0.0, 0.02))
        self.d = nn.Parameter(torch.zeros(self.K))
        # Per-head, per-boundary gains/biases (matmul outputs, NOT lone scalars): G,Q:[K,3]
        # init: G ~ ones (location passes through), Q[:,0]=0.5, Q[:,1:]=0.5413 -> softplus(0.5413)~=1.0
        # so each head starts at boundaries tau = (0.5, 1.5, 2.5).
        self.G = nn.Parameter(torch.ones(self.K, 3))
        Q0 = torch.empty(self.K, 3)
        Q0[:, 0] = 0.5
        Q0[:, 1] = 0.5413
        Q0[:, 2] = 0.5413
        self.Q = nn.Parameter(Q0)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        sq = features.dim() == 2
        x = features.unsqueeze(0) if sq else features          # [B,N,D]
        B, N, _ = x.shape
        h = self.bottleneck(x.reshape(B * N, -1)).reshape(B, N, self.hidden_dim)  # [B,N,128]
        a = self.attW(self.attV(h) * self.attU(h)).squeeze(-1)  # [B,N] gated-attn logits
        alpha = F.softmax(a, dim=1)                             # [B,N] sums to 1 (n=1 safe)
        z = (alpha.unsqueeze(-1) * h).sum(dim=1)                # [B,128] WEIGHTED POOL (backbone)

        u = z @ self.P.t() + self.d                            # [B,K] K ordinal locations
        q = u.unsqueeze(-1) * self.G.unsqueeze(0) + self.Q.unsqueeze(0)  # [B,K,3] per-head matmul read
        e1 = q[..., 0]                                         # [B,K]
        e2 = e1 + F.softplus(q[..., 1])                        # monotone (>= e1)
        e3 = e2 + F.softplus(q[..., 2])                        # monotone (>= e2)
        g = (torch.sigmoid(u - e1) + torch.sigmoid(u - e2) + torch.sigmoid(u - e3))  # [B,K] per-head grade
        y = g.mean(dim=1)                                      # [B] consensus, uniform 1/K

        if sq:
            y = y.view(-1)[:1]
            alpha = alpha.squeeze(0)
        return y.view(-1), (alpha if return_attention else None), None


KWARGS = dict(input_dim=1280, num_classes=1)
