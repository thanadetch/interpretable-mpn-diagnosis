"""a153 — Grading-Wrapped Gated Attention (keep the smart baseline, WRAP its attention with a grading bias).

User direction: ABMIL baseline is already strong; ADOPT/WRAP it with a grading-aligned
component (for the thesis grading story) while still beating baseline. a153 keeps the FULL baseline
(bottleneck → Ilse gated attention → classifier) UNCHANGED and only ADDS a grading bias to the
attention LOGITS before softmax:

  h_i        = bottleneck(feat_i)                     # baseline bottleneck (Linear→ReLU→Dropout)
  a_i        = W(V(h_i)·U(h_i))                        # baseline gated-attention logit
  g_i        = w_fib·z_fib_i − softplus(λ_b)·relu(z_bone_i−z_fib_i) − softplus(λ_a)·relu(z_adip_i−z_fib_i)
               # grading bias: up-weight fibrosis (a152 ingredient) + avoid bone/fat (a148 ingredient)
  w_i        = softmax(a_i + g_i)                      # WRAP: baseline attention nudged toward grading
  y          = classifier(Σ w_i·h_i)                   # baseline classifier head

w_fib/λ_b/λ_a init 0 ⇒ a153 STARTS EXACTLY as the baseline (=pure SimpleGatedMIL) and LEARNS how
much grading bias to add. Tests the user's hypothesis: can wrapping the smart baseline with a
grading bias BEAT it (vs a120 which wrapped with bone-only and failed)? Ablation a154 = bias OFF
(reproduces baseline). Reads data_distract [feat|fib|bone|adipose]. Hybrid concept-norm.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_FIB_STD, _BONE_STD, _ADI_STD = 0.0281, 0.0193, 0.0179


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, use_grading=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.use_grading = use_grading
        # ---- baseline ABMIL (unchanged) ----
        self.bottleneck = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # ---- grading bias params (init 0 ⇒ starts = pure baseline) ----
        self.w_fib = nn.Parameter(torch.tensor(0.0))    # fibrosis up-weight
        self.lam_b = nn.Parameter(torch.tensor(0.0))    # bone avoidance
        self.lam_a = nn.Parameter(torch.tensor(0.0))    # adipose avoidance

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        a = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)   # [N] baseline attn logit
        if self.use_grading and features.shape[1] >= D + 3:
            fib = features[:, D]; bone = features[:, D + 1]; adi = features[:, D + 2]
            zf = (fib - fib.mean()) / _FIB_STD
            zb = (bone - bone.mean()) / _BONE_STD
            za = (adi - adi.mean()) / _ADI_STD
            g = self.w_fib * zf - F.softplus(self.lam_b) * F.relu(zb - zf) - F.softplus(self.lam_a) * F.relu(za - zf)
            a = a + g
        w = F.softmax(a, dim=0)
        agg = torch.mv(h.t(), w)
        y = self.classifier(agg).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, use_grading=True)
