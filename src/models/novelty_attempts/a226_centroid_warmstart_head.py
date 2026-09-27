"""a226 - In-memory grade-centroid WARM-START of the regression head (self-contained, real labels).

Self-contained version of the old prototype warm-start (no external .pt): at init, mean-pool each TRAIN
bag's frozen features and average per grade -> 4 grade centroids -> initialise the classifier weight along
the (G3-centroid - G0-centroid) direction (the empirical fibrosis axis from REAL labels). The head then
trains normally. Tests whether a label-informed head init helps. Concept-free (real grades), self-
contained (computed in memory from train), deterministic.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_BB = {768: "features_titan_reti", 1280: "features_virchow2_reti", 1536: "features_uni2_reti"}
_CACHE = {}


def _axis(input_dim):
    if input_dim in _CACHE:
        return _CACHE[input_dim]
    src = str(Path(__file__).resolve().parents[2])
    if src not in sys.path: sys.path.insert(0, src)
    from data.bag_dataset import GradingBagDatasetFull
    from train_grading_reti import patient_split
    data = Path(__file__).resolve().parents[2] / "data"
    ds = GradingBagDatasetFull(data / _BB[input_dim])
    tr = patient_split(ds, seed=2)[0]
    sums = {g: None for g in range(4)}; cnts = {g: 0 for g in range(4)}
    for i in tr:
        feat, lab, _ = ds[i]
        m = feat.float().mean(0)
        sums[lab] = m if sums[lab] is None else sums[lab] + m
        cnts[lab] += 1
    mu = {g: (sums[g] / cnts[g]) for g in range(4) if cnts[g] > 0}
    d = mu[3] - mu[0]
    d = d / (d.norm() + 1e-8)
    _CACHE[input_dim] = d
    return d


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # warm-start: set proj's first row toward the grade axis so bottleneck sees the fibrosis direction
        try:
            d = _axis(input_dim)
            with torch.no_grad():
                self.proj.weight[0].copy_(d)
        except Exception:
            pass

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features) + 0.0 * self.proj(features)  # proj kept in graph (warm-started row)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
