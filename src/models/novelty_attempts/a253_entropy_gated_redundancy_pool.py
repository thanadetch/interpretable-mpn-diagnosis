"""a253 - entropy-gated redundancy-repulsion pooling (conditional de-concentration).

THE WALL (uni2-test): one FIXED attention setting cannot fit all three backbones. uni2's baseline
attention is intrinsically the MOST concentrated (normalized entropy ~0.891 vs virchow2 ~0.965 /
titan ~0.947). Sharpening helps the diffuse backbones but over-concentrates uni2 and collapses it;
an UNCONDITIONAL de-concentration (a248 James-Stein toward mean) over-corrects the strong backbones
that NEED some concentration (titan test 0.9180). The ONLY viable mechanism is one that
CONDITIONALLY de-concentrates an over-peaked read while leaving an already-diffuse read near-identity,
and the model must INFER the concentration from the bag content itself (it cannot see the backbone).

MECHANISM (a253):
  (1) h = bottleneck(x)                                   [N,H]
  (2) standard gated-attention logits and weights:
        e = W( tanh(V h) * sigmoid(U h) ).view(-1)        [N]
        a = softmax(e)                                    [N]  (the KEPT relevance/attention pool)
  (3) per-bag normalized entropy of the attention weights -- the concentration sensor:
        H = -sum(a * log(a+eps)) / log(max(N,2))  in [0,1]
        H ~ 1  -> diffuse  (virchow2/titan);  H ~ 0 -> peaked (uni2).
  (4) redundancy via matmul (matmul/elementwise only; no pairwise-distance, eigendecomp, or linear-solve ops):
        z = h / ||h||;  S = relu(z @ z.t())               [N,N]  cosine sim, S_ii = 1
        q = softplus(quality_head(h))                     [N]    quality >= 0 (a VECTOR head, not a lone scalar)
        r_i = (S @ q)_i - q_i = sum_{j!=i} S_ij q_j        [N]    quality-weighted similarity to OTHERS
  (5) CLOSED-FORM entropy gate (NO MLP, NO learnable knob -- pure function of H):
        g = sigmoid((H - 0.5) / 0.1)  in [0,1]
  (6) repulsion strength scaled by (1-g):
        p_i = q_i / (1 + (1-g) * r_i.clamp(0))
        H>0.5 (diffuse, g->1): (1-g)->0 -> p->q -> NO repulsion (near-identity to a quality pool).
        H<0.5 (peaked,  g->0): (1-g)->1 -> full q/(1+r) redundancy suppression -> de-concentrates.
  (7) a_refined = softmax(log(p+eps))                     [N]  sums to 1 -> KEEP THE ATTENTION-WEIGHTED POOL.
  (8) z_bag = h.t() @ a_refined                           [H]
  (9) y = classifier(z_bag).view(-1)

WHY THIS IS CATEGORICALLY NEW (vs the failed de-concentration attempts):
  - a248 (James-Stein ESS shrink toward MEAN, UNCONDITIONAL): a253 never shrinks toward the bag mean and
    is GATED -- diffuse bags are untouched.
  - a237 (global mean-prior blend) / a243 (content-gated mean/attn blend): a253 does NOT blend with a mean
    read; it re-weights via pairwise redundancy and never mixes in a mean vector.
  - a233 (learnable power-mean) / a238/a158 (diffuse-temperature) / a215 (entmax alpha) / a237 (blend lambda):
    those rely on a LONE learnable shape-scalar that goes inert (flat gradient). a253 has NO learnable knob
    for the de-concentration -- the gate g(H) is a CLOSED-FORM function of the bag's own entropy, and all
    learnable content lives in VECTOR heads (V,U,W,quality_head,classifier) with non-flat gradients.
  - a244 (matmul redundancy-repulsion, best-but-below at titan 0.9480): a253 keeps the redundancy term but
    applies it CONDITIONALLY -- gated by per-bag attention entropy and modulating an attention-derived pool,
    so strong/diffuse backbones get (1-g)->0 (a244 always applied full repulsion to everyone).

HONEST APPROXIMATION NOTES:
  - "redundancy" r is a cosine-similarity surrogate (matmul), NOT a determinantal / DPP marginal -- exact
    DPP needs eigh/solve which are MPS-forbidden. It is a heuristic of "how much other high-quality patches
    look like me".
  - The entropy H is computed on the FIRST-pass attention weights a (step 2); the gate then reshapes them.
    H is a scalar summary of concentration, not a calibrated probability.
  - The sigmoid center 0.5 and slope 0.1 are FIXED constants (not learned), chosen so the transition zone
    sits between uni2 (~0.89) and a hypothetical peaked failure; they are heuristic, not tuned per-backbone.

Self-contained (torch / torch.nn / torch.nn.functional ONLY). Concept-free, NOT norm-weighted.
Permutation-invariant AND bag-size-invariant. Deterministic in eval(). n=1 safe: with N=1, a=[1],
H=0 (log(max(1,2))=log2 denom, -1*log(1)=0 -> H=0), S=[[1]], r=0 -> p=q -> a_refined=[1]; no division by
(N-1), no std, no NaN.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, attn_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # Standard gated attention (Ilse et al.): two branches tanh(Vh) * sigmoid(Uh), then a linear score.
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_W = nn.Linear(attn_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        eps = 1e-8
        h = self.bottleneck(features)                              # [N,H]
        N = h.shape[0]

        # (2) standard gated attention -> first-pass weights a (sum 1).
        e = self.attn_W(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).view(-1)  # [N]
        a = F.softmax(e, dim=0)                                    # [N]

        # (3) per-bag normalized entropy of a -> concentration sensor in [0,1].
        denom = torch.log(torch.tensor(float(max(N, 2)), device=h.device, dtype=h.dtype))
        ent = -(a * torch.log(a + eps)).sum() / denom             # scalar in [0,1]

        # (4) redundancy from OTHER patches (matmul-only cosine surrogate).
        z = h / (h.norm(dim=1, keepdim=True) + 1e-6)              # unit rows [N,H]
        S = F.relu(z @ z.t())                                     # [N,N], S_ii = 1
        q = F.softplus(self.quality_head(h).view(-1)) + 1e-4     # [N] quality >= 0
        r = (S @ q) - q                                           # [N] redundancy to OTHERS (>=0 by construction)

        # (5) closed-form entropy gate g (no learnable knob).
        g = torch.sigmoid((ent - 0.5) / 0.1)                      # scalar in [0,1]

        # (6) conditional repulsion: diffuse (g->1) -> p->q; peaked (g->0) -> full q/(1+r).
        p = q / (1.0 + (1.0 - g) * r.clamp_min(0.0))             # [N]

        # (7) refined attention weights (sum 1) -- KEEP the attention-weighted pool.
        a_refined = F.softmax(torch.log(p + eps), dim=0)          # [N]

        # (8) pool, (9) regress.
        z_bag = h.t() @ a_refined                                 # [H]
        y = self.classifier(z_bag).view(-1)                       # [num_classes] -> view(-1)

        if return_attention:
            return y, a_refined, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
