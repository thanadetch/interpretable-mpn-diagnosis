"""a316 — Tangent/Fréchet gated MIL: direction-only spherical correction of the pooled vector.

GENUINELY NEW DOF (verified zero disk precedent: no frechet/karcher/geodesic/slerp/tangent/arccos in
337 modules). Every dead candidate acted on the attention WEIGHTS (softmax shape → crashes uni2) or the
MAGNITUDE (‖h‖ → inert/grade-uninformative). a316 leaves BOTH bit-for-bit baseline and acts ONLY on the
DIRECTION of the pooled vector via one closed-form Karcher (weighted Fréchet-mean) step on the unit
sphere. It PROVABLY collapses to the baseline softmax-mean as the bag's angular spread → 0 (uni2's
concentrated regime), and only deviates for angularly-spread diffuse bags (titan/virchow2).

DISCLOSURE: exploratory fish under the user's explicit "keep searching, it's just research" request.
The data ceiling may still cap it; we run it, disclose, and multi-seed-audit any seed-2 pass. uni2-safety
is by construction (correction is O(theta^2), vanishing on concentrated bags). ZERO extra params (197,250).
Self-contained, concept-free, permutation/size-invariant, deterministic eval, MPS-safe, no new deps.
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
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                                  # [N,H]
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)    # [N]
        a = F.softmax(e, dim=0)                                                         # [N] (baseline weights)
        c = torch.mv(h.t(), a)                                                          # [H] baseline pool
        R = c.norm()                                                                    # baseline magnitude
        u = h / (h.norm(dim=1, keepdim=True) + 1e-6)                                    # [N,H] unit dirs
        mu = torch.mv(u.t(), a)
        mu = mu / (mu.norm() + 1e-6)                                                    # [H] pole (mean dir)
        cos_i = (u @ mu).clamp(-1 + 1e-6, 1 - 1e-6)                                     # [N]
        theta_i = torch.arccos(cos_i)                                                   # [N] angular dist
        v = u - cos_i.unsqueeze(1) * mu.unsqueeze(0)                                    # [N,H] tangent comp
        v = theta_i.unsqueeze(1) * v / (v.norm(dim=1, keepdim=True) + 1e-6)             # log-map
        t = torch.mv(v.t(), a)                                                          # [H] weighted tangent centroid
        nt = t.norm()
        if nt > 1e-6:
            mu_new = torch.cos(nt) * mu + torch.sin(nt) * (t / (nt + 1e-6))             # exp-map (one Karcher step)
        else:
            mu_new = mu
        z = R * mu_new                                                                  # reattach baseline magnitude
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
