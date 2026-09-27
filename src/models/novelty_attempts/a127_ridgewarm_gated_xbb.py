"""a127 — warm-gated attention, head init along the train-only RIDGE direction
(better than the crude G3-G0 axis), backbone-agnostic. Tests if a stronger warm
direction lets the warm-gated win on ALL backbones at seed-2 (incl. titan).

Same as a113/a103 but the warm direction = ridge(grade ~ mean-pooled features)
solved train-only per backbone (data/ridge_dir_{bb}_reti_train_seed2.pt), chosen
by input_dim. forward reads ONLY 'features'. RAW logit. Deterministic at eval.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_DATA = Path(__file__).resolve().parents[3] / "data"
_RIDGE_BY_DIM = {
    1280: _DATA / "ridge_dir_virchow2_reti_train_seed2.pt",
    1536: _DATA / "ridge_dir_uni2_reti_train_seed2.pt",
    768: _DATA / "ridge_dir_titan_reti_train_seed2.pt",
}


def _dir(input_dim: int) -> Optional[torch.Tensor]:
    p = _RIDGE_BY_DIM.get(input_dim)
    if p is None or not p.exists():
        return None
    try:
        v = torch.load(p, map_location="cpu", weights_only=False)["axis"].float().view(-1)
        if v.numel() == input_dim:
            return v / v.norm().clamp(min=1e-8)
    except Exception:
        pass
    return None


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=True):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        if warm_start:
            d = _dir(input_dim)
            if d is not None:
                with torch.no_grad():
                    resp = self.bottleneck[0].weight.detach() @ d
                    resp = resp / resp.norm().clamp(min=1e-8)
                    self.classifier.weight.copy_(resp.view(1, -1) * 3.0)
                    self.classifier.bias.fill_(1.5)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        attn = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        agg = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        out = self.classifier(agg)
        if return_attention:
            return out, attn, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=True)
