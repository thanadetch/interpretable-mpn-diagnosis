"""a249 - centered-contrast attention. NEW: score attention over each patch's DEVIATION from a
LEARNED, content-gated bag context, then pool the ORIGINAL h with those weights.

MECHANISM
  (1) h = bottleneck(features)                       [N,H]
  (2) m = h.mean(0)                                  size/perm-invariant bag context [H], n=1 safe
  (3) g = sigmoid(W_g m) * (W_c m)                   learned content-gated bulk vector [H]
                                                     (capacity in two Linear(H,H), NOT a lone scalar)
  (4) r = h - g.unsqueeze(0)                         per-patch contrast/residual [N,H]
  (5) e = W( tanh(V r) * sigmoid(U r) ); a = softmax(e)   gated-attention scorer reads the RESIDUAL
  (6) z = a @ h                                      attention-weighted pool of ORIGINAL h (backbone kept)

WHY NEW (vs the close priors)
  - a241 FiLM-modulates attention SHARPNESS from the mean but never forms a residual nor pools over it.
  - a166 subtracts a FROZEN data-derived PCA nuisance axis (fixed buffer, requires eigh at init, no
    per-bag learned context, no contrast-scored attention).
  - a199 adds a residual MLP block (h + FF(h)), not a context contrast.
  The learned content-GATED context g = sigmoid(W_g m) * (W_c m) keying attention on deviation-from-bulk
  is a new construction. Patches that merely echo the dominant bulk (low residual) get low attention;
  patches that DEVIATE (salient fibrosis-pattern outliers vs diffuse background) get attended.

WHY IT ADDRESSES THE WALL (disjoint-helper, NO global sharpness knob)
  Because attention keys on contrast-to-context rather than absolute content, it de-concentrates uni2
  (whose raw-feature attention over-concentrates on absolute-magnitude patches) while still letting
  virchow2/titan find salient regions -- a content-structural fix, not a temperature/sharpness scalar.
  The attention-weighted pool backbone (z = sum_i a_i h_i) is preserved per the round-2 directive.

CONSTRAINTS HONORED
  Self-contained nn.Module (torch / nn / F only); permutation- & size-invariant; deterministic in eval();
  n=1 safe (no unbiased std, no div-by-zero); MPS-safe ops only (matmul, elementwise, sigmoid/tanh/softmax,
  mean). No lone learnable shape-scalar as the core novelty -- capacity lives in linear maps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.ctx_gate = nn.Linear(hidden_dim, hidden_dim)    # -> sigmoid gate over bag context
        self.ctx_val = nn.Linear(hidden_dim, hidden_dim)     # -> context value
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                        # [N,H]
        m = h.mean(dim=0)                                    # [H] context, n=1 safe
        g = torch.sigmoid(self.ctx_gate(m)) * self.ctx_val(m)  # [H] learned content-gated bulk
        r = h - g.unsqueeze(0)                               # [N,H] residual/contrast
        e = self.attention_W(self.attention_V(r) * self.attention_U(r)).squeeze(-1)  # [N]
        a = F.softmax(e, dim=0)                              # [N] sums to 1
        z = a @ h                                            # [H] attention-weighted pool of ORIGINAL h
        y = self.classifier(z).view(-1)                      # shape (1,)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
