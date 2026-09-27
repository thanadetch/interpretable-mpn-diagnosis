"""a140 — Adaptive Grading-Prior Aggregator (AGPA).

Direct response to the per-backbone HEADROOM finding (a135/a136): a fixed grading-prior
helps backbones whose free head underperforms (virchow2/uni2) but HURTS the one whose
free head is already near-ceiling (titan 0.954). AGPA lets a SINGLE model learn, per
backbone, HOW MUCH to lean on the grading prior vs a free attention head:

    y = α · y_prior + (1-α) · y_free ,   α = sigmoid(alpha_raw)  (learned)

  y_prior = a135 bone-aware diffuse fibrosis-density readout (grading-faithful, low-DOF)
  y_free  = baseline ABMIL readout (flexible)

Hypothesis: training drives α→high where the prior helps (virchow2/uni2) and α→low where
the free head is already optimal (titan) → the model never gets dragged below the free
head on any backbone. The LEARNED α is itself an interpretable diagnostic of how readily
each FM's features expose the grade linearly (thesis value independent of any QWK win).

Input: data_bonefib bag = [feat | bone | fib] (1282/1538/770-d). forward reads ONLY
'features'. Prior warm-started per-fold (axis+anchors+bone consts like a135). RAW logit.
Ablation: α frozen=1 ⇒ pure prior (=a135); α frozen=0 ⇒ pure free (=baseline).
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_DATA = Path(__file__).resolve().parents[3] / "data"
_BB_BY_DIM = {1280: "virchow2", 1536: "uni2", 768: "titan"}
_BONE_MEAN, _BONE_STD = -0.0486, 0.0193
_FIB_MEAN, _FIB_STD = 0.0045, 0.0281


def _current_seed(default: int = 2) -> int:
    for i, a in enumerate(sys.argv):
        if a == "--seed" and i + 1 < len(sys.argv):
            try: return int(sys.argv[i + 1])
            except ValueError: pass
        if a.startswith("--seed="):
            try: return int(a.split("=", 1)[1])
            except ValueError: pass
    return default


def _proto(input_dim: int):
    bb = _BB_BY_DIM.get(input_dim)
    if bb is None: return None
    p = _DATA / f"prototypes_{bb}_reti_train_seed{_current_seed()}.pt"
    return p if p.exists() else None


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
                 alpha_mode="learn", warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.alpha_mode = alpha_mode      # "learn" | "prior" (α=1) | "free" (α=0)
        # ---- prior path (a135 bone-aware diffuse) ----
        v_init = None; w_s_init, c_s_init = 1.0, 1.5
        if warm_start:
            p = _proto(input_dim)
            if p is not None:
                try:
                    blob = torch.load(p, map_location="cpu", weights_only=False)
                    v0 = blob["axis"].float().view(-1)
                    if v0.numel() == input_dim:
                        v0 = v0 / v0.norm().clamp(min=1e-8); v_init = v0.clone()
                        pr = blob["prototypes"]
                        anch = torch.tensor([float(pr[g].float().view(-1) @ v0) for g in range(4)])
                        amin, amax = float(anch.min()), float(anch.max()); span = max(amax - amin, 1e-6)
                        w_s_init = 3.0 / span; c_s_init = -(3.0 / span) * amin
                except Exception: pass
        if v_init is None:
            v_init = F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        self.lam_raw = nn.Parameter(torch.tensor(0.0))
        self.w_s = nn.Parameter(torch.tensor(float(w_s_init)))
        self.c_s = nn.Parameter(torch.tensor(float(c_s_init)))
        # ---- free path (baseline SimpleGatedMIL) ----
        self.bottleneck = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # ---- blend ----
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5 equal blend at init

    def _alpha(self, device):
        if self.alpha_mode == "prior": return torch.ones((), device=device)
        if self.alpha_mode == "free":  return torch.zeros((), device=device)
        return torch.sigmoid(self.alpha_raw)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        # prior path (bone-aware diffuse)
        if features.shape[1] >= D + 2:
            bone = (features[:, D] - _BONE_MEAN) / _BONE_STD
            fib = (features[:, D + 1] - _FIB_MEAN) / _FIB_STD
            w = F.softmax(-F.softplus(self.lam_raw) * F.relu(bone - fib), dim=0)
        else:
            w = torch.full((feats.shape[0],), 1.0 / feats.shape[0], device=feats.device)
        z = torch.mv(feats.t(), w)
        y_prior = self.w_s * torch.dot(z, self.v) + self.c_s
        # free path (gated MIL)
        h = self.bottleneck(feats)
        attn = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        agg = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y_free = self.classifier(agg).squeeze(-1)
        a = self._alpha(feats.device)
        y = (a * y_prior + (1.0 - a) * y_free).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, alpha_mode="learn", warm_start=True)
