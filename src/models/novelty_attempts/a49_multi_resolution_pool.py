"""a49 - multi-resolution gated pool (H21 main, batch 24).

Philosophy bucket: multi_scale_fusion

Hint targeted: H21_multi_resolution_pool — first batch in the previously
untried `multi_scale_fusion` philosophy bucket. Background: real
per-ROI µm scale labels (20 / 50 / 100 / 200 / unknown) are NOT
accessible inside the current `Model.forward(features)` API (the
trainer passes only the patch feature tensor; modifying the
dataset/trainer to expose scale violates §0 rule 11 of the playbook).
We therefore approach `multi_scale_fusion` via a *content-based*
multi-resolution proxy that lives entirely inside the model: two
parallel gated-attention pools over the same bag at two different
representational resolutions.

Hypothesis: reticulin fibrosis carries information at two distinct
granularities — (a) fine-grained per-patch texture (individual fibre
appearance, presence of single thick bundles) and (b) region-level
sub-pattern composition (clusters of similar tissue states that
together signal e.g. nodular replacement vs diffuse fine fibres). The
single-resolution ABMIL baseline only pools at (a); a40's
multi-query architecture pools at (b)-ish via K=4 queries but each
query still attends over all N raw patches. A *true* multi-resolution
pool first re-organises the N patches into M soft clusters (a
content-driven coarse view) and then attention-pools at both
resolutions before fusing — giving the head explicit access to two
representational scales.

Mechanism (exact):
    h_i  = bottleneck(features_i)                        # [N, hidden=128]

    # Fine branch: gated attention over N patches (Ilse-style).
    v_p  = tanh(V_p(h_i)); u_p = sigmoid(U_p(h_i))
    a_p  = softmax(W_p(v_p * u_p), dim=N)                # [N, 1]
    bag_fine   = (a_p * h).sum(0)                        # [hidden]

    # Coarse branch: M=8 learnable centroids c_m in hidden-dim.
    # Soft cluster assignment p[i, m] = softmax_m(<h_i, c_m> / sqrt(hidden)).
    # Cluster rep cl_m = sum_i p[i, m] * h_i / (sum_i p[i, m] + eps).
    p    = softmax(h @ C.T / sqrt(hidden), dim=M)        # [N, M]
    cl_m = (p.T @ h) / (p.sum(0).unsqueeze(1) + eps)     # [M, hidden]
    v_c  = tanh(V_c(cl_m)); u_c = sigmoid(U_c(cl_m))
    a_c  = softmax(W_c(v_c * u_c), dim=M)                # [M, 1]
    bag_coarse = (a_c * cl_m).sum(0)                     # [hidden]

    # Fuse and predict.
    y    = clamp(Linear(2*hidden, 1)([bag_fine; bag_coarse]), 0, 3)

Why this is genuinely multi-resolution (not a relabel of a29):
    * a29's K=4 queries each attend over the *raw* N patches → all K
      bag-reps live at the same resolution (per-patch).
    * a49's coarse branch first *collapses* N patches into M=8 soft
      clusters via content-similarity, then attention-pools over those
      M clusters. The coarse branch never sees individual patches at
      the attention stage; it sees their cluster averages. The fine
      branch is the only one that sees per-patch granularity.
    * The fusion is at the feature level (concat 2*hidden → Linear),
      not at the prediction level (which DE23 closed).

Pathology rationale: G2/G3 ROIs typically show *regions* of similar
fibrosis density (focal-to-diffuse coarse bundles) rather than
uniformly distributed individual coarse fibres. A pool that can
attend at the region level should be able to up-weight a single
fibrotic region in a bag where most patches are uninformative
background — exactly the case where per-patch attention struggles.

Ablation companion: a50_multi_resolution_pool_collapsed (M=1).
Forces the coarse branch to be a global mean (single cluster), so the
two branches reduce to "patch-gated-attn + global-mean concat". The
a49/a50 contrast isolates **whether multi-cluster coarsening adds
signal beyond a plain global-mean side-channel**.

Kill criterion (pre-registered): abandon the H21 family if both
a49.val_qwk < 0.79 AND a50.val_qwk < 0.79. If a49 - a50 < +0.005 val
the coarsening is not the active ingredient (fold into DE table).

Param count (input_dim=1280, hidden_dim=128, n_clusters=8):
    bottleneck Linear(1280, 128) + bias                = 163,968
    fine: V_p Linear(128, 128) + bias                  =  16,512
          U_p Linear(128, 128) + bias                  =  16,512
          W_p Linear(128, 1)   + bias                  =     129
    coarse centroids [M=8, 128]                        =   1,024
          V_c Linear(128, 128) + bias                  =  16,512
          U_c Linear(128, 128) + bias                  =  16,512
          W_c Linear(128, 1)   + bias                  =     129
    fuse  Linear(256, 1) + bias                        =     257
    --------------------------------------------------------
    total trainable                                    = 231,555
    (vs baseline 197,250 -> +34,305 / +17%; bigger than a40 172,609,
     within the "no hard cap" range but flagged for overfit risk;
     see also a50 with M=1 -> 230,531 params (loses only the M-1=7
     extra centroid rows).)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GatedAttention(nn.Module):
    """Ilse-style gated attention over a set [S, hidden] -> weights [S, 1]."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, hidden]
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        scores = self.W(v * u)              # [S, 1]
        return F.softmax(scores, dim=0)     # [S, 1]


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        n_clusters: int = 8,
        dropout: float = 0.5,
        clamp_output: bool = True,
        centroid_init_std: float = 0.02,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert n_clusters >= 1

        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.clamp_output = clamp_output
        self.eps = eps
        self.scale = 1.0 / math.sqrt(hidden_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Fine branch (per-patch gated attention).
        self.attn_fine = _GatedAttention(hidden_dim)

        # Coarse branch (cluster centroids + cluster-level gated attn).
        # When n_clusters == 1 the coarse branch degenerates to a plain
        # global mean of h, so we skip allocating dead parameters
        # (centroids never used; softmax over 1 element makes attn_coarse
        # gradient-free). This keeps a50's param count meaningful and
        # avoids unused-parameter optimizer warnings.
        if n_clusters > 1:
            self.centroids = nn.Parameter(
                torch.randn(n_clusters, hidden_dim) * centroid_init_std
            )
            self.attn_coarse = _GatedAttention(hidden_dim)
        else:
            self.register_parameter("centroids", None)
            self.attn_coarse = None

        # Fuse and predict.
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        nn.init.constant_(self.classifier.bias, 1.5)  # prior mean

    def _coarse_pool(self, h: torch.Tensor) -> torch.Tensor:
        """Soft-cluster h into M cluster reps via content similarity.

        Returns cluster reps `cl: [M, hidden]`. When n_clusters == 1
        this is just the global mean of h reshaped to [1, hidden].
        """
        if self.n_clusters == 1:
            return h.mean(dim=0, keepdim=True)               # [1, hidden]
        scores = h @ self.centroids.t() * self.scale         # [N, M]
        p = F.softmax(scores, dim=1)                          # [N, M]
        num = p.t() @ h                                       # [M, hidden]
        den = p.sum(dim=0, keepdim=True).t() + self.eps       # [M, 1]
        return num / den                                      # [M, hidden]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        h = self.bottleneck(features)                         # [N, hidden]

        # Fine branch.
        a_fine = self.attn_fine(h)                            # [N, 1]
        bag_fine = (a_fine * h).sum(dim=0)                    # [hidden]

        # Coarse branch.
        cl = self._coarse_pool(h)                             # [M, hidden]
        if self.attn_coarse is not None:
            a_coarse = self.attn_coarse(cl)                   # [M, 1]
            bag_coarse = (a_coarse * cl).sum(dim=0)           # [hidden]
        else:
            # n_clusters == 1: cl is [1, hidden] = global mean; no attn.
            bag_coarse = cl.squeeze(0)                        # [hidden]

        # Fuse and predict.
        bag = torch.cat([bag_fine, bag_coarse], dim=0)        # [2*hidden]
        y = self.classifier(bag.unsqueeze(0)).squeeze(0)      # [1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # Surface the fine-branch attention for interpretability
            # (matches the [N] shape the trainer's heatmap code expects).
            return y, a_fine.squeeze(-1), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    n_clusters=8,
    dropout=0.5,
    centroid_init_std=0.02,
)




