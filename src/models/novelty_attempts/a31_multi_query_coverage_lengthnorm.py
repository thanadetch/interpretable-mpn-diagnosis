"""a31 - multi-query cross-attention + coverage prior + length-norm temperature
        (H17_multi_query_with_a17_coverage_stack, batch 16 main).

Hypothesis (H17): batch 15 confirmed multi-query cross-attention (K=4)
is the first family in 14 batches to beat a17 on val (a29 = 0.7997 vs
a17 = 0.7970). Independently, a17's H10 stack (coverage prior +
length-normalised softmax temperature) was the best mechanism on top
of single-head gated attention. The H10 mechanisms reweight ATTENTION
without putting σ between bag rep and y, so they are *compositional*
with anything that ALSO reweights attention. Multi-query cross-attn
also reweights attention. Therefore the two should stack.

If H10 mechanisms carry their +0.016 val gain independently of the
softmax-mean head structure (H10 was measured on top of SimpleGatedMIL,
not multi-query), then stacking them onto a29 could finally close the
~0.018 gap from a29 (0.7997) to the locked val gate (0.8182).

Mechanism (exact):
    h_i       = bottleneck(features_i)               # [N, hidden=128]
    k_i       = Linear(h_i)                          # [N, q_dim=64]
    n_i       = ||h_i||_2                            # [N], per-patch magnitude
    c_i       = sigmoid((n_i - tau) / beta)          # [N] in (0, 1), coverage prior (a09/a17)
    T         = softplus(c0_raw) / sqrt(N)           # scalar, length-norm temp (a07/a17)
    Q         = learnable parameters                 # [K=4, q_dim=64]
    scores0   = Q @ k.T * (q_dim ** -0.5)            # [K, N], cross-attn dot products
    scores    = (scores0 + alpha * log(c + eps)) * T # [K, N], H10 stack injected
    attn      = softmax(scores, dim=N)               # [K, N]
    bag_K     = attn @ h                             # [K, hidden]
    bag       = bag_K.flatten()                      # [K * hidden]
    y         = clamp(Linear(K*hidden, 1)(bag), 0, 3)

Coverage is broadcast across all K queries (same per-patch prior
reweights every query's attention). Length-norm scales all K
attention rows by the same T.

Initialisation: tau=1.0, beta=1.0, alpha=1e-3 (~ 0 → starts as pure
a29 + length-norm), c0 such that T ~ 1 at N=40 (c0_init = sqrt(40)
~ 6.32). Classifier bias initialised at 1.5 (target prior mean) to
avoid dead-clamp at init.

Ablation companion: `a32_multi_query_xattn_lengthnorm.py` -- identical
but `use_coverage = False` (alpha=0, tau/beta/eps unused). Isolates
"coverage prior" as the active ingredient on top of multi-query +
length-norm.

Kill criterion: abandon H17 if BOTH a31 and a32 val_qwk < a29 (0.7997)
at seed=2.

Param count (input_dim=1280, hidden_dim=128, q_dim=64, K=4):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=4, q_dim=64]                 =     256
    classifier Linear(K*hidden=512, 1) + bias  =     513
    tau, beta_raw, alpha_raw, c0_raw           =       4
    --------------------------------------------------
    total trainable                            = 172,997
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        query_dim: int = 64,
        n_queries: int = 4,
        dropout: float = 0.5,
        clamp_output: bool = True,
        query_init_std: float = 0.02,
        # H10 stack hyperparameters
        use_coverage: bool = True,
        tau_init: float = 1.0,
        beta_init: float = 1.0,
        alpha_init: float = 1e-3,
        c0_init: float = 6.324555,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert n_queries >= 1

        self.n_queries = n_queries
        self.scale = query_dim ** -0.5
        self.clamp_output = clamp_output
        self.use_coverage = use_coverage
        self.eps = eps

        # Same bottleneck shape as ABMIL / a17 / a29.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.key_proj = nn.Linear(hidden_dim, query_dim)
        self.queries = nn.Parameter(torch.randn(n_queries, query_dim) * query_init_std)
        self.classifier = nn.Linear(n_queries * hidden_dim, num_classes)
        nn.init.constant_(self.classifier.bias, 1.5)

        # Length-norm temperature (always on for H17).
        self.c0_raw = nn.Parameter(torch.tensor(_inv_softplus(c0_init)))

        # Coverage prior (a09/a17), optional via use_coverage flag.
        if use_coverage:
            self.tau = nn.Parameter(torch.tensor(float(tau_init)))
            self.beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
            self.alpha_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_init)))

    def _temperature(self, n: int) -> torch.Tensor:
        c0 = F.softplus(self.c0_raw)
        return c0 / math.sqrt(max(n, 1))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag).
        h = self.bottleneck(features)                       # [N, hidden]
        k = self.key_proj(h)                                # [N, q_dim]
        n = h.shape[0]

        # Cross-attention raw scores.
        scores = self.queries @ k.t() * self.scale          # [K, N]

        # Coverage prior (broadcast across K).
        if self.use_coverage:
            norms = torch.linalg.vector_norm(h, ord=2, dim=-1)        # [N]
            beta = F.softplus(self.beta_raw).clamp(min=1e-4)
            c = torch.sigmoid((norms - self.tau) / beta)              # [N] in (0, 1)
            alpha = F.softplus(self.alpha_raw)
            scores = scores + alpha * torch.log(c + self.eps).unsqueeze(0)  # [K, N]

        # Length-norm temperature (applied to all K rows).
        T = self._temperature(n)
        scores = scores * T

        attn = F.softmax(scores, dim=1)                     # [K, N]
        bag_per_query = attn @ h                            # [K, hidden]
        bag = bag_per_query.reshape(-1)                     # [K*hidden]
        y = self.classifier(bag.unsqueeze(0)).squeeze(0)    # [1]

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attn.mean(dim=0), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,            # a31 main: 4 queries (matches a29 best)
    dropout=0.5,
    query_init_std=0.02,
    use_coverage=True,      # a31 main: full H17 stack
    tau_init=1.0,
    beta_init=1.0,
    alpha_init=1e-3,
    c0_init=6.324555,
)

