"""a168 — Hybrid-normalized gated attention (FEATURE-norm lever; the round-31 hybrid-norm on titan).

Tests the user's "use norm instead" idea on the strong gated head. Normalizes the features before
the (unchanged) ABMIL head with the round-31 hybrid scheme:

    f' = (f − bag_mean) / global_std

  bag_mean  = per-dim mean over the patches of THIS bag      (per-bag CENTER — removes bag-level offset)
  global_std= per-dim std over ALL TRAIN patches             (global SCALE — computed IN-MEMORY, leakage-safe)

Per-bag centering removes a bag-level staining/exposure offset; global std scaling puts every dim on
a comparable scale without leaking per-bag scale. NOT a feature-norm-‖h‖ reweighting (that was ruled
out as grade-uninformative); this normalizes the feature SPACE, then the usual gated attention reads
it. Self-contained: global_std derived from the train split. 0 extra head DOF.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a155_axis_attention_density import _current_seed, _BB_BY_DIM, _DATA

_STD_CACHE: dict = {}


def _global_std(input_dim: int, seed: int):
    bb = _BB_BY_DIM.get(input_dim)
    key = (bb, seed)
    if key in _STD_CACHE:
        return _STD_CACHE[key]
    _src = str(Path(__file__).resolve().parents[2])
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from data.bag_dataset import GradingBagDatasetFull
    from train_grading_reti import patient_split
    ds = GradingBagDatasetFull(_DATA / f"features_{bb}_reti")
    train_idx, _, _ = patient_split(ds, seed=seed)
    n = 0
    s = torch.zeros(input_dim, dtype=torch.float64)
    ss = torch.zeros(input_dim, dtype=torch.float64)
    for i in train_idx:
        f = torch.load(ds.samples[i][0], map_location="cpu", weights_only=False)["feats"].double()
        s += f.sum(0); ss += (f * f).sum(0); n += f.shape[0]
    mean = s / n
    var = (ss / n) - mean * mean
    std = var.clamp(min=1e-8).sqrt().float()
    _STD_CACHE[key] = std
    return std


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=True):
        super().__init__()
        self.feat_dim = input_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        std = _global_std(input_dim, _current_seed()) if warm_start else torch.ones(input_dim)
        self.register_buffer("global_std", std)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        feats = (feats - feats.mean(0, keepdim=True)) / self.global_std.unsqueeze(0)
        h = self.bottleneck(feats)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, a, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
