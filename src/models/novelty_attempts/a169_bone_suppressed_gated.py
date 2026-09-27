"""a169 — Bone-suppressed gated attention (SIDE-STUDY: uses concept scores to define the bone dir).

Tests the advisor's "avoid the dark bone trabeculae" principle directly: down-weight patches that
align with the BONE direction in feature space, then run the usual gated attention.

    u_bone = mean(high-bone patches) − mean(high-fibrosis patches)   # concept-defined, at init, in-memory
    a_i    = softmax(e_i)                                            # learned gated attention
    s_i    = σ(−κ · (feat_i · u_bone))                               # suppression: low weight for bone-like patches
    w_i    = a_i·s_i / Σ a_j·s_j ;  z = Σ w_i h_i ;  y = classifier(z)

κ learnable, init 2.0 (ACTIVE suppression from the start). u_bone is a FIXED buffer.

⚠️ NOT self-contained: u_bone is derived from the precomputed concept scores
(data/patch_concept_scores_all_reti.pt) → this is an INTERPRETABILITY / method side-study (like
a148), not a deployable novelty under the self-contained rule. Purpose: quantify whether explicitly
suppressing bone helps grading, and on which backbone.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a155_axis_attention_density import _BB_BY_DIM, _DATA, _current_seed

_BONE_CACHE: dict = {}


def _bone_dir(input_dim: int, seed: int):
    bb = _BB_BY_DIM.get(input_dim)
    key = (bb, seed)
    if key in _BONE_CACHE:
        return _BONE_CACHE[key]
    _src = str(Path(__file__).resolve().parents[2])
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from data.bag_dataset import GradingBagDatasetFull
    ds = GradingBagDatasetFull(_DATA / f"features_{bb}_reti")
    concepts = torch.load(_DATA / "patch_concept_scores_all_reti.pt", map_location="cpu", weights_only=False)
    feat_root = str(_DATA / f"features_{bb}_reti") + "/"
    F_, B_, Fi_ = [], [], []
    for i in range(0, len(ds.samples), 2):
        pt, _ = ds.samples[i]
        rel = str(pt).split(feat_root)[-1]
        c = concepts.get(rel)
        feats = torch.load(pt, map_location="cpu", weights_only=False)["feats"].float().numpy()
        if c is None or int(c.get("n", -1)) != feats.shape[0]:
            continue
        F_.append(feats); B_.append(np.asarray(c["bone_trabecular"], float)); Fi_.append(np.asarray(c["fibrosis_stroma"], float))
    X = np.concatenate(F_); B = np.concatenate(B_); Fi = np.concatenate(Fi_)
    bz = (B - B.mean()) / B.std(); fz = (Fi - Fi.mean()) / Fi.std()
    bone_ex = (bz > np.quantile(bz, 0.75)) & (fz < np.quantile(fz, 0.5))
    fib_ex = (fz > np.quantile(fz, 0.75)) & (bz < np.quantile(fz, 0.5))
    d = X[bone_ex].mean(0) - X[fib_ex].mean(0)
    d = d / (np.linalg.norm(d) + 1e-8)
    u = torch.tensor(d, dtype=torch.float32)
    _BONE_CACHE[key] = u
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
        self.kappa = nn.Parameter(torch.tensor(2.0))  # ACTIVE bone suppression
        u = _bone_dir(input_dim, _current_seed()) if warm_start else F.normalize(torch.randn(input_dim), dim=0)
        self.register_buffer("u_bone", u)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        s_bone = feats @ self.u_bone
        if s_bone.shape[0] >= 2:
            s_bone = (s_bone - s_bone.mean()) / s_bone.std().clamp(min=1e-6)
        supp = torch.sigmoid(-F.softplus(self.kappa) * s_bone)  # down-weight bone-aligned patches
        w = a * supp
        w = w / w.sum().clamp(min=1e-8)
        z = torch.mv(h.t(), w)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, w, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
