"""a159 — Axis-shrunk gated attention (grading tilt on the strong gated head, init = baseline).

LIGHT, self-contained. Keeps ABMIL exactly and tilts its attention logits toward the
fibrosis axis by ONE learnable scalar γ:  logit_i = e_i + γ·ẑ_i, where ẑ_i is the within-bag
standardized fibrosis-axis score ⟨feat_i, v⟩ and v is the train G3−G0 mean-difference direction
(computed IN-MEMORY, leakage-safe; reused from a155._train_axis, no external file). γ init 0 →
attention is EXACTLY the baseline gated attention; the optimiser only tilts toward high-fibrosis
patches if it helps. v is a FIXED buffer (not learned) → only 1 extra DOF over the baseline
(much lighter than a153's bias path that val-overfit). Grading-aligned: up-weights the diffuse
fibrotic signal while preserving the head that already performs near-ceiling on titan.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a155_axis_attention_density import _train_axis, _current_seed


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))  # init 0 = exact baseline attention
        v = None
        if warm_start:
            got = _train_axis(input_dim, _current_seed())
            if got is None:
                raise ValueError(f"[a159] unknown input_dim {input_dim} (no fibrosis-axis mapping).")
            v = got[0].float().view(-1).clone()
        if v is None:
            v = F.normalize(torch.randn(input_dim), dim=0)
        self.register_buffer("v", v)  # FIXED axis (0 DOF)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        s = feats @ self.v
        if s.shape[0] >= 2:
            s = (s - s.mean()) / s.std().clamp(min=1e-6)
        else:
            s = torch.zeros_like(s)
        attn = F.softmax(e + self.gamma * s, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
