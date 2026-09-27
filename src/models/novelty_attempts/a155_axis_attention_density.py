"""a155 — Fibrosis-Axis-Attention Diffuse Density (FULLY SELF-CONTAINED, no external files).

Grading-aligned aggregator that runs on the ORIGINAL features only (`--data_root data`) and is
ENTIRELY self-contained in this module: it needs NO prototype file, NO concept dir, NO separate
pre-run script. At construction it derives the fibrosis direction v IN-MEMORY from the TRAIN
split of the features (leakage-safe), then up-weights high-fibrosis patches and pools:

  v       = (μ_G3 − μ_G0)/‖·‖   computed at init from TRAIN ROIs only        # data-driven init, leakage-safe
            (μ_g = mean Virchow2 patch feature over grade-g TRAIN ROIs; split = locked patient_split(seed))
  s_i     = ⟨feat_i, v⟩                                                       # per-patch fibrosis-axis score
  s_z     = (s_i − bag_mean)/bag_std                                          # within-bag standardize
  w_i     = softmax(λ · s_z)                                                  # up-weight high-fibrosis patches (λ learnable)
  z       = Σ w_i·feat_i ;  y = w_s·⟨z, v⟩ + c_s                              # diffuse pool → axis readout → grade

The fibrosis axis is a DATA-DRIVEN INITIALISATION (computed once from the train split, like a
class-mean init) — v, w_s, c_s, λ are then all learnable. Self-contained ⇒ portable: clone the
repo + have `data/features_<bb>_reti/`, and the model just runs (nothing to build/upload first).
Thesis-clean: "the fibrosis direction is initialised as the train grade-G3−G0 mean-feature
difference; the aggregator up-weights patches along it and reads out the bag-wide density."
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
_AXIS_CACHE: dict = {}   # (bb, seed) -> (axis[D], anchors[4]); in-memory only, no file written


def _current_seed(default: int = 2) -> int:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--seed" and i + 1 < len(argv):
            try: return int(argv[i + 1])
            except ValueError: pass
        if a.startswith("--seed="):
            try: return int(a.split("=", 1)[1])
            except ValueError: pass
    return default


def _train_axis(input_dim: int, seed: int):
    """Compute the fibrosis axis + per-grade anchors IN-MEMORY from the TRAIN split.
    No file is read or written. Cached per (backbone, seed) for the process lifetime."""
    bb = _BB_BY_DIM.get(input_dim)
    if bb is None:
        return None
    key = (bb, seed)
    if key in _AXIS_CACHE:
        return _AXIS_CACHE[key]
    _src = str(Path(__file__).resolve().parents[2])  # repo/src
    if _src not in sys.path: sys.path.insert(0, _src)
    from data.bag_dataset import GradingBagDatasetFull
    from train_grading_reti import patient_split
    feats_dir = _DATA / f"features_{bb}_reti"
    if not feats_dir.is_dir():
        raise FileNotFoundError(
            f"[a155] self-contained init needs the original features dir {feats_dir} "
            f"(no other files required). Ensure data/features_{bb}_reti/ is present.")
    ds = GradingBagDatasetFull(feats_dir)
    train_idx, _, _ = patient_split(ds, seed=seed)
    sums = {g: None for g in range(4)}; cnt = {g: 0 for g in range(4)}
    for i in train_idx:
        bag = torch.load(ds.samples[i][0], map_location="cpu", weights_only=False)
        f = bag["feats"].float(); g = int(ds.samples[i][1])
        sums[g] = f.sum(0) if sums[g] is None else sums[g] + f.sum(0); cnt[g] += f.shape[0]
    means = {g: (sums[g] / cnt[g]) for g in range(4) if cnt[g] > 0}
    axis = means[3] - means[0]; axis = axis / axis.norm().clamp(min=1e-8)
    anchors = torch.tensor([float(means[g] @ axis) for g in range(4)])
    _AXIS_CACHE[key] = (axis, anchors)
    return _AXIS_CACHE[key]


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        v_init = None; w_s_init, c_s_init = 1.0, 1.5
        if warm_start:
            got = _train_axis(input_dim, _current_seed())
            if got is None:
                raise ValueError(f"[a155] unknown input_dim {input_dim} (no backbone mapping for the fibrosis axis).")
            axis, anchors = got
            v_init = axis.float().view(-1).clone()
            amin, amax = float(anchors.min()), float(anchors.max()); span = max(amax - amin, 1e-6)
            w_s_init = 3.0 / span; c_s_init = -(3.0 / span) * amin
        if v_init is None:
            v_init = F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        self.w_s = nn.Parameter(torch.tensor(float(w_s_init)))
        self.c_s = nn.Parameter(torch.tensor(float(c_s_init)))
        self.lam_raw = nn.Parameter(torch.tensor(-1.0))   # λ=softplus(lam_raw), init ~0.31 (gentle)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        s = feats @ self.v
        if s.shape[0] >= 2:
            sz = (s - s.mean()) / s.std().clamp(min=1e-6)
            w = F.softmax(F.softplus(self.lam_raw) * sz, dim=0)
        else:
            w = torch.full((feats.shape[0],), 1.0 / feats.shape[0], device=feats.device)
        z = torch.mv(feats.t(), w)
        y = (self.w_s * torch.dot(z, self.v) + self.c_s).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
