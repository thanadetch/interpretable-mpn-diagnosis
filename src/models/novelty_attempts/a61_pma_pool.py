"""a61 - Pooling-by-Multihead-Attention (PMA) with ONE learned seed query.

Hypothesis (H_pma_seed_query). The baseline ABMIL weights patches by
a *per-patch gating MLP* (tanh-gate * sigmoid-gate -> scalar -> softmax). That
scorer judges each patch in ISOLATION: its weight depends only on h_i, never on
"what the model is collectively looking for". Reticulin grading is a diffuse,
bag-wide judgement of fibre density; the patches that should drive the grade are
defined RELATIVE to a learned target pattern, not by an isolated per-patch gate.

PMA (Set-Transformer / Lee et al. 2019) replaces the gating scorer with a single
learnable SEED QUERY that *content-attends* to the patches via a scaled
dot-product:

    h_i   = Dropout(ReLU(Linear(1280->128) f_i))      # baseline encoder (capacity here)
    k_i   = Linear(128->128) h_i                       # per-patch key (per head)
    s_i   = <q_seed, k_i> / sqrt(d_head)               # content-based affinity
    a_i   = softmax_i(s_i)                              # pooling weights (over patches)
    bag   = sum_i a_i * h_i                             # one [128] bag rep (per head -> concat)
    y     = clamp( Linear(128->1)(bag), 0, 3 )          # plain linear, UNBOUNDED head

This is the canonical PMA with ONE seed (S=1), giving a SINGLE bag vector, then a
PLAIN linear readout. The pooling weight a_i is a query-KEY dot product, NOT a
per-patch tanh/sigmoid gate -> a genuinely different attention mechanism that the
baseline does not implement.

Why this is NOT a dead-end (each refuted family addressed):
  - NOT a29/a30 (multi-query xattn + FLATTEN(K*hidden) head, refuted). a29 used
    K=4 queries and concatenated K bag reps into a 512-d vector before a Linear,
    which is high-variance and overfit. Here S=1 -> a single 128-d bag rep ->
    Linear(128,1). No flatten, no K*hidden head. (KWARGS ships S=1, n_heads=2;
    the a62 ablation flips n_heads=1 to isolate "does multi-head content pooling
    beat single-head".)
  - NOT the gated-attention baseline: attention is a learned query vs per-patch
    KEY dot-product (content-based, relative to a target), not the baseline's
    isolated per-patch gating MLP. Different inductive bias, same param budget.
  - NOT coverage/extent/fraction (refuted 3x): no threshold, no sigmoid count,
    no "fraction of fibre-positive patches". The readout is a weighted MEAN of
    features through a linear head.
  - NOT norm-based: ||h|| is never used; weights come from learned content keys.
  - NOT top-k/argmax: softmax over ALL patches (soft, full-support pooling).
  - NOT attention+mean concat / mean+std / quantile: a single attention-pooled
    vector, no concatenation of pooling statistics.
  - HARD constraint respected: the softmax attention is the POOLING operation
    (per-patch weighting BEFORE aggregation); there is NO sigmoid/softmax GATE
    sitting between the bag representation and the scalar. The head is a plain
    Linear(128,1), unbounded, with clamp only at the very end.

Permutation invariance: softmax over patches + weighted sum is symmetric in the
patch index. Bag-size invariance: softmax normalises by sum over N, so the
attention weights and the pooled vector are scale-free in N -> duplicating the
bag gives the same weighted mean (same y up to dropout=eval determinism).

Multi-head detail (n_heads=2, head_dim=64): keys and the seed query are split
into n_heads groups of head_dim; softmax pooling is done PER HEAD over patches;
the per-head pooled values (slices of h) are concatenated back to 128-d, then a
single Linear(128,1). Heads stay small (head_dim=64) to keep capacity low.

Param count (input_dim=1280, hidden=128, n_heads=2, head_dim=64):
    bottleneck Linear(1280,128)+b = 163,968
    key_proj   Linear(128,128)+b  =  16,512
    seed query [n_heads, head_dim] = 1*128 =     128
    head       Linear(128,1)+b     =     129
    ----------------------------------------------
    total                          = 180,737   (< ~200K; < baseline 197,250)

Ablation companion: a62_pma_pool_1head.py - identical but n_heads=1 (head_dim
=128). Tests whether MULTI-head content pooling is the active ingredient over a
single content-pooling head. (Both are still PMA, distinct from gated baseline.)

Kill criterion: abandon if a61 val_qwk < 0.7888 at seed=2 (does not beat the
gated-attention baseline) AND a61 <= a62 on val (multi-head adds nothing). DoD =
multi-seed audit {0,1,2,3,42}, not seed=2 alone.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        n_heads: int = 2,
        dropout: float = 0.5,
        query_init_std: float = 0.02,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads."
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-patch keys (content). Values reuse the bottleneck output h
        # (no separate V projection) to keep params low.
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # ONE learnable seed query, split across heads: [n_heads, head_dim].
        self.seed_query = nn.Parameter(torch.randn(n_heads, self.head_dim) * query_init_std)

        # Plain linear readout from the single pooled bag vector. Init at the
        # prior mean (1.5 = midpoint of [0,3]) so the end-clamp does not
        # dead-zero gradients at random init.
        self.head = nn.Linear(hidden_dim, num_classes)
        with torch.no_grad():
            self.head.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        N = features.size(0)
        h = self.bottleneck(features)                       # [N, hidden]
        k = self.key_proj(h)                                # [N, hidden]

        # Split keys / values into heads: [N, n_heads, head_dim].
        k = k.view(N, self.n_heads, self.head_dim)
        v = h.view(N, self.n_heads, self.head_dim)          # values reuse h

        # Content affinity per head: seed_query [n_heads, head_dim] vs keys.
        #   scores[h, i] = <q_h, k_{i,h}> / sqrt(head_dim)
        scores = torch.einsum("hd,nhd->hn", self.seed_query, k) * self.scale  # [n_heads, N]
        attn = F.softmax(scores, dim=1)                     # [n_heads, N] pooling weights

        # Per-head attention-weighted sum of values, then concat across heads.
        pooled = torch.einsum("hn,nhd->hd", attn, v)        # [n_heads, head_dim]
        bag = pooled.reshape(-1)                            # [hidden]

        y = self.head(bag.unsqueeze(0)).squeeze(0)          # [1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # Mean attention across heads for interpretability: [N].
            return y, attn.mean(dim=0), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    n_heads=2,              # a61 main: 2-head content pooling
    dropout=0.5,
    query_init_std=0.02,
    clamp_output=True,
)
