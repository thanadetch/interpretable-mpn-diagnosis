"""a166 — Nuisance-PC-removed gated attention (FEATURE-level denoise, new lever).

All prior titan-GATE1 candidates tweaked the WEIGHTING/pooling and hit the headroom wall. a166
attacks a DIFFERENT lever: it cleans the FEATURES before the (unchanged) ABMIL head.

At init it derives IN-MEMORY from the TRAIN split (leakage-safe, self-contained, no file):
  v_fib  = (μ_G3 − μ_G0)/‖·‖                                  # the fibrosis direction (keep this)
  u_nuis = leading PCA direction of the train features AFTER removing their v_fib component
           (= the dominant NON-fibrosis variation: staining / scale / batch nuisance)
Forward removes the nuisance component from every patch feature, then runs gated attention:
  f' = f − (f·u_nuis)·u_nuis ;  gated-attention(f') → grade

Rationale: removing the dominant grade-irrelevant variation should denoise the signal for ALL
backbones (incl. titan) without imposing a grading prior on the readout (so it does not fight the
near-ceiling titan head the way a155/a161/a162 did). u_nuis/v_fib are FIXED buffers (0 extra DOF
in the head). Self-contained; reuses a155._train_axis for v_fib.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a155_axis_attention_density import _train_axis, _current_seed, _BB_BY_DIM, _DATA

_NUIS_CACHE: dict = {}


def _nuisance_dir(input_dim: int, seed: int, v_fib: torch.Tensor):
    """Leading PCA direction of TRAIN features orthogonal to v_fib (in-memory, cached)."""
    bb = _BB_BY_DIM.get(input_dim)
    key = (bb, seed)
    if key in _NUIS_CACHE:
        return _NUIS_CACHE[key]
    _src = str(Path(__file__).resolve().parents[2])
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from data.bag_dataset import GradingBagDatasetFull
    from train_grading_reti import patient_split
    ds = GradingBagDatasetFull(_DATA / f"features_{bb}_reti")
    train_idx, _, _ = patient_split(ds, seed=seed)
    chunks = []
    for i in train_idx[::2]:  # subsample bags for speed
        bag = torch.load(ds.samples[i][0], map_location="cpu", weights_only=False)
        f = bag["feats"].float()
        if f.shape[0] > 60:
            f = f[torch.randperm(f.shape[0])[:60]]
        chunks.append(f)
    X = torch.cat(chunks, 0)
    X = X - X.mean(0, keepdim=True)
    # remove the fibrosis component, then take the leading PC of the residual
    X = X - (X @ v_fib).unsqueeze(1) * v_fib.unsqueeze(0)
    cov = (X.t() @ X) / max(X.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    u = evecs[:, -1]
    u = u / u.norm().clamp(min=1e-8)
    _NUIS_CACHE[key] = u
    return u


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
        u = None
        if warm_start:
            got = _train_axis(input_dim, _current_seed())
            if got is None:
                raise ValueError(f"[a166] unknown input_dim {input_dim}.")
            v_fib = got[0].float().view(-1)
            u = _nuisance_dir(input_dim, _current_seed(), v_fib).float().view(-1)
        if u is None:
            u = F.normalize(torch.randn(input_dim), dim=0)
        self.register_buffer("u_nuis", u)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        feats = feats - (feats @ self.u_nuis).unsqueeze(1) * self.u_nuis.unsqueeze(0)
        h = self.bottleneck(feats)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, a, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
