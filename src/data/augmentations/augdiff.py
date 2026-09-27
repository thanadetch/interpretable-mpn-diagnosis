"""augdiff - diffusion-based feature augmentation for MIL (train-only, drop-in).

Compact re-implementation of the CORE idea of AugDiff (Shao et al., IEEE T-AI 2024,
arXiv:2303.06371): instead of augmenting gigapixel pixels, learn the *feature* manifold
with a small conditional diffusion model and generate on-manifold augmented views of a
bag's patch embeddings. Unlike Mixup / feature_noise (off-manifold), the reverse-denoise
step keeps samples on the learned feature distribution while injecting diversity.

Mechanism (SDEdit-style: partial forward-noise -> learned reverse-denoise):
  1. `_ensure(pool)` (lazy, once): pool all train patch embeddings, per-grade; z-normalise;
     train a tiny grade-conditional DDPM epsilon-predictor on them.
  2. `__call__`: forward-diffuse the bag to timestep t0 = round(strength * T), then
     DDIM-denoise back to 0 conditioned on the bag's grade -> an augmented, label-preserving,
     on-manifold view of the SAME bag. Higher `strength` = stronger (more diverse) augmentation.

`strength` in (0,1] = noise level / augmentation intensity. Requires the train pool.
Everything runs on the model's device (MPS-safe) and is deterministic given the global seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

# --- fixed diffusion / training budget (kept small so the lazy train is ~1-2 min on MPS) ---
T_STEPS = 50            # diffusion timesteps
DDIM_STEPS = 10         # reverse steps at augmentation time
TRAIN_ITERS = 1200      # denoiser training iterations
BATCH = 256
HIDDEN = 512
LR = 1e-3
MAX_BANK = 20000        # total patches sampled across grades to train on
N_GRADES = 4


class _Denoiser(nn.Module):
    """Tiny grade-conditional epsilon-predictor: (x_t, t, grade) -> predicted noise."""

    def __init__(self, dim: int):
        super().__init__()
        self.t_emb = nn.Embedding(T_STEPS, 64)
        self.g_emb = nn.Embedding(N_GRADES, 32)
        self.net = nn.Sequential(
            nn.Linear(dim + 64 + 32, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_t, self.t_emb(t), self.g_emb(g)], dim=1)
        return self.net(h)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = MAX_BANK):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._model: Optional[_Denoiser] = None
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._acp: Optional[torch.Tensor] = None   # alphas_cumprod [T]
        self._tried = False

    # ---- diffusion schedule helpers -------------------------------------------------
    def _schedule(self, device):
        betas = torch.linspace(1e-4, 0.02, T_STEPS, device=device)
        self._acp = torch.cumprod(1.0 - betas, dim=0)  # [T]

    def _ensure(self, pool, device, dtype) -> None:
        if self._tried:
            return
        self._tried = True
        # 1) gather per-grade patches
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            if 0 <= g < N_GRADES:
                chunks[g].append(item[0].float())
        if not chunks:
            return
        per_grade_cap = max(1, self.max_bank // N_GRADES)
        feats, grades = [], []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > per_grade_cap:  # cap per grade before concat -> bounded memory
                allp = allp[torch.randperm(allp.shape[0])[:per_grade_cap]]
            feats.append(allp)
            grades.append(torch.full((allp.shape[0],), g, dtype=torch.long))
        X = torch.cat(feats, dim=0)
        G = torch.cat(grades, dim=0)
        dim = X.shape[1]

        # 2) z-normalise (critical for stable diffusion on raw embeddings)
        self._mean = X.mean(dim=0, keepdim=True)
        self._std = X.std(dim=0, keepdim=True).clamp(min=1e-6)
        Xn = ((X - self._mean) / self._std).to(device, dtype)
        G = G.to(device)

        # 3) train the tiny conditional DDPM
        self._schedule(device)
        acp = self._acp
        model = _Denoiser(dim).to(device, dtype)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        n = Xn.shape[0]
        model.train()
        with torch.enable_grad():  # __call__ is @torch.no_grad(); training needs grad
            for _ in range(TRAIN_ITERS):
                bi = torch.randint(0, n, (min(BATCH, n),), device=device)
                x0 = Xn[bi]
                g = G[bi]
                t = torch.randint(0, T_STEPS, (x0.shape[0],), device=device)
                a = acp[t].unsqueeze(1)
                eps = torch.randn_like(x0)
                x_t = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * eps
                pred = model(x_t, t, g)
                loss = F.mse_loss(pred, eps)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        self._model = model

    # ---- augmentation ---------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool, features.device, features.dtype)
        if self._model is None:
            return features, float(label)
        g = int(round(float(label)))
        if not (0 <= g < N_GRADES):
            return features, float(label)

        mean = self._mean.to(features.device, features.dtype)
        std = self._std.to(features.device, features.dtype)
        acp = self._acp
        x0 = (features - mean) / std                      # [N, D] normalised
        N = x0.shape[0]
        gt = torch.full((N,), g, dtype=torch.long, device=features.device)

        # forward-diffuse to t0 = round(strength * (T-1))
        t0 = int(round(self.s * (T_STEPS - 1)))
        t0 = max(1, min(T_STEPS - 1, t0))
        a0 = acp[t0]
        eps = torch.randn_like(x0)
        x_t = math.sqrt(float(a0)) * x0 + math.sqrt(float(1.0 - a0)) * eps

        # DDIM reverse from t0 -> 0 (deterministic), grade-conditioned
        ts = torch.linspace(t0, 0, DDIM_STEPS + 1, device=features.device).round().long()
        ts = torch.unique_consecutive(ts)
        for j in range(len(ts) - 1):
            t_cur, t_nxt = int(ts[j]), int(ts[j + 1])
            a_cur = acp[t_cur]
            a_nxt = acp[t_nxt] if t_nxt > 0 else torch.tensor(1.0, device=features.device)
            tvec = torch.full((N,), t_cur, dtype=torch.long, device=features.device)
            e = self._model(x_t, tvec, gt)
            x0_pred = (x_t - torch.sqrt(1.0 - a_cur) * e) / torch.sqrt(a_cur)
            x_t = torch.sqrt(a_nxt) * x0_pred + torch.sqrt(1.0 - a_nxt) * e

        out = x_t * std + mean                            # de-normalise
        return out.to(features.dtype), float(label)
