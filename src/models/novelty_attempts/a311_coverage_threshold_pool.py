"""a311 — Coverage-Threshold Pooling (per-bag-centered sigmoid coverage normalisation).

DISCLOSURE: this is a single fixed-mechanism drop-in, so it is subject in principle to the
disjoint-helper wall (titan/virchow2 want sparse pooling, uni2 wants dense; GATE2-at-seed-2 has been
shown provably unreachable by any backbone-agnostic drop-in). It is a PRINCIPLED wall-test, not a
p-hacking pile-on: it targets the exact gap the wall describes.

Mechanism (genuinely distinct from softmax/entmax pooling and from every prior candidate):
replace the softmax (a215 baseline) / entmax (ASGAP) normalisation of the gated-attention scores with
a *coverage* read. Compute the Ilse gated-attention score e_i as usual, center it by the bag's own
mean (so the operating point is relative to each bag's score distribution, permutation/size-invariant
and shift-invariant — the attention bias cancels, so unlike the inert learnable-alpha this cannot be
trivially absorbed by W's bias), then weight each patch by a SIGMOID:
    w_i = sigmoid( (e_i - mean(e)) / temp - beta ),   a = w / sum(w),   z = sum_i a_i h_i .

Why this could (in principle) spare uni2 while still helping the diffuse backbones — the wall gap:
  - DENSE like softmax: sigmoid never assigns exactly zero weight, so no patch is dropped. entmax/
    sparsemax zero out patches and that is precisely what CRASHED uni2 (a310, a306). Coverage is
    uni2-safe by construction.
  - DIFFUSE / bag-wide like the grading principle (overall reticulin-meshwork density, NOT a few
    standout patches): sigmoid SATURATES, so the many patches scoring above the bag mean share weight
    roughly equally instead of softmax's exponential concentration on the single top patch. This is a
    coverage/density read along the learned score — the one direction shown grade-informative
    (held-out coverage Spearman +0.84) and the only grounded direction the project endorses.
  - It does NOT weight by feature-norm ||h|| (ruled out: grade-uninformative on Virchow2).

temp (sharpness) and beta (coverage margin above the bag mean) are two learnable scalars; the rest of
the model — bottleneck, gated scorer, linear head — is identical to the baseline. Self-contained,
concept-free, permutation/size-invariant, deterministic at inference, MPS-safe, no new deps.
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
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.log_temp = nn.Parameter(torch.tensor(0.0))   # softplus -> temp ~0.69 init; learnable sharpness
        self.beta = nn.Parameter(torch.tensor(0.0))       # coverage margin above the bag mean
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)   # [N]
        temp = F.softplus(self.log_temp) + 1e-3                                         # > 0
        centered = e - e.mean()                                                        # per-bag centering
        w = torch.sigmoid(centered / temp - self.beta)                                 # dense coverage weights
        a = w / w.sum().clamp(min=1e-8)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
