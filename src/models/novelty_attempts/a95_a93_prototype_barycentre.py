"""a95 (== a93) — Calibrated prototype-distance BARYCENTRE for diffuse fibrosis grading.

One line:
    Map the DIFFUSE bag-mean fibrosis projection to a grade via a *barycentre*
    (expected grade) over a softmax soft-assignment to the FOUR frozen, train-only
    per-grade prototype ANCHOR coordinates, with James-Stein shrinkage of the
    anchors toward their centroid — a calibrated prototype-distance readout with
    NO free slope/intercept (only a temperature and a shrinkage scalar learn).

Why this fits the GRADING semantics:
    Reticulin fibrosis grade is the OVERALL / DIFFUSE density of the fibre
    meshwork across the WHOLE marrow — not a property of a few patches, and the
    feature-norm ||h|| is a nuisance (Spearman ~0 with grade). So we DIFFUSE-pool
    (plain mean, every patch equal — no attention, no top-k, no norm-weighting)
    and read out a *scalar* fibrosis score by projecting the bag mean onto the
    supervised train axis. The novelty is the READOUT: instead of a free affine
    head whose slope+intercept SmoothL1 leaves under-determined (and which then
    drifts fold-to-fold => wide val<->test gap), we PIN the four grade levels to
    data-derived anchor coordinates and read the *expected grade* under a softmax
    soft-assignment to those anchors. The only readout DOF the fold can fit is one
    temperature (assignment softness) + one shrinkage scalar — far fewer ways to
    overfit the score->grade map => tighter cross-fold agreement.

Mechanism (single bag features f in R^{N x 1280}):
    1) z   = mean_i f_i                       # [1280] DIFFUSE pool (every patch equal)
    2) s   = <z, v>                           # scalar diffuse fibrosis score;
                                              #   v warm-started at train axis (seed=2)
    3) a_g = <P_g, v0>  (g=0..3)  FROZEN      # per-grade anchor coords along the axis;
                                              #   v0 = axis at init, P_g = train prototypes.
                                              #   Numerically (-11.02,-4.04,+1.40,+8.51) —
                                              #   cleanly monotone. Stored as a buffer
                                              #   (0 trainable params). Anchoring is done in
                                              #   1-D along the supervised axis ON PURPOSE:
                                              #   full-1280-d prototype distances are near-
                                              #   equidistant for any bag-mean (curse of
                                              #   dimensionality) so a full-D barycentre
                                              #   collapses to ~uniform E[grade]=1.5; the
                                              #   discriminative structure lives along the axis.
    4) rho = rho_max * sigmoid(rho_raw) in [0, rho_max)  (James-Stein shrinkage; 1 scalar)
       a_g'= (1-rho)*a_g + rho * mean_g(a_g)             # contract anchors toward centroid
    5) tau = exp(log_tau) > 0  (1 scalar);  d_g = (s - a_g')^2
       p   = softmax(-d_g / tau^2);  y = sum_g g * p_g   # expected grade in [0,3], RAW out
    Returns the 3-tuple (y, p.detach(), None). Trainer rounds+clips at eval.

At init this maps a bag ~ prototype_g to E[grade] ~ g (verified ~0.04/1.07/1.93/2.97).

Trainable params (main path): v (1280) + log_tau (1) + rho_raw (1) = 1282 active.
A free affine head (w_s, c_s = 2 params) is *present* so the a96 ablation can flip it
on, but in the anchor path it is bypassed (receives zero gradient). Total = 1284 << 197250.

Why orthogonal to refuted attempts:
  - NOT a74 dist-match: a74 matches the bag's per-patch soft-QUANTILE vector to per-grade
    reference quantile DISTRIBUTIONS of per-patch scores; a93 has NO per-patch distribution
    and NO quantiles — it compares ONE diffuse bag-mean scalar to four scalar anchor points.
  - NOT a89/a47 cumulative-link: those fit THREE free monotone boundaries (a fold-slidable
    S-curve); a93 fits NO boundaries — grade levels are PINNED at the frozen anchors, the
    map has zero free slope/intercept.
  - NOT a25 prototype-as-attention: a25 used the prototype direction as a per-patch softmax
    SCORER then weighted-mean; a93 uses prototypes as bag-level GRADE ANCHORS, no per-patch
    weighting.
  - NOT mean-pool+linear (a53/a92): those use a FREE affine head on the bag mean; a93 uses
    the frozen-anchor barycentre.
  - NOT coverage/moments/pairwise/PMA/graph-smooth/L2-norm/consensus/ensemble/heavy-reg:
    no fraction/threshold, no concatenated stats, no pairwise term, no learned query, no
    affinity matrix, no ||h|| weighting, no model averaging, no input noise.
  The genuinely new object: a CALIBRATED multi-anchor prototype-distance barycentre with
  shrinkage — the cache's 4 per-grade prototypes were only ever used as an init direction
  or for coverage/quantile-dist-match, never as barycentric distance anchors.

Permutation- & bag-size-invariance: the ONLY use of the bag is z = mean_i f_i (mean is
permutation-invariant and normalises by N => duplicating the bag leaves z, hence s and y,
unchanged up to numerical noise). Everything after z is a fixed scalar function. No dropout,
no randomness => DETERMINISTIC at inference (eval()). forward reads ONLY 'features'.

Ablation companion: a96 imports THIS Model and flips use_anchors=False. The diffuse pool
z=mean f_i and warm-started projection s=<z,v> are BYTE-IDENTICAL; only the readout changes
to a FREE affine head y = w_s*s + c_s (warm-started to map the anchor span to [0,3]). a93 vs
a96 on PAIRED cross-fold Δ answers: does pinning the score->grade map to the train anchors
reduce the val<->test gap vs a free affine readout on the same diffuse score? Kill if a93 <= a96 paired.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_axis_and_anchors(
    path: Path, input_dim: int, num_grades: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (unit axis [input_dim], anchors [num_grades]) from train-only prototypes.

    anchors[g] = <P_g, v0> with v0 the unit train axis (seed=2). Returns (None, None)
    if the cache is absent / malformed so the model still constructs with safe defaults.
    Uses ONLY train-derived quantities — no test info, no runtime features.
    """
    if not path.is_file():
        return None, None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "axis" not in blob or "prototypes" not in blob:
        return None, None
    v = blob["axis"].float().view(-1)
    if v.numel() != input_dim:
        return None, None
    v0 = v / v.norm().clamp(min=1e-8)
    protos = blob["prototypes"]
    anchors = []
    for g in range(num_grades):
        key = g if g in protos else (float(g) if float(g) in protos else None)
        if key is None:
            return v0, None
        P = protos[key].float().view(-1)
        if P.numel() != input_dim:
            return v0, None
        anchors.append(float(P @ v0))
    return v0, torch.tensor(anchors, dtype=torch.float32)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        num_grades: int = 4,
        rho_max: float = 0.5,            # James-Stein shrinkage ceiling (rho in [0, rho_max))
        tau_init: float = 4.0,           # initial softmax temperature (>0)
        use_anchors: bool = True,        # a95 main = True; a96 ablation flips to False
        warm_start: bool = True,         # init v at train axis; init free head to anchor span
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,      # RAW logits; trainer rounds+clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert num_grades >= 2, "need >= 2 grade anchors."
        self.input_dim = int(input_dim)
        self.num_grades = int(num_grades)
        self.rho_max = float(rho_max)
        self.use_anchors = bool(use_anchors)
        self.clamp_output = bool(clamp_output)

        # Load train axis + per-grade anchor coords (init-only use of seed=2 cache).
        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        axis, anchors = _load_axis_and_anchors(path, self.input_dim, self.num_grades)

        # Learnable projection direction v (the only high-dim trainable object).
        # Warm-started at the train axis; falls back to a small random unit-ish vector.
        if warm_start and axis is not None:
            v_init = axis.clone()
        else:
            v_init = torch.randn(self.input_dim)
            v_init = v_init / v_init.norm().clamp(min=1e-8)
        self.v = nn.Parameter(v_init)

        # Frozen per-grade anchor coordinates a_g (buffer, 0 trainable params).
        # Defensive fallback to an evenly-spaced monotone span if the cache is missing,
        # so the model is still well-defined and monotone.
        if anchors is None:
            anchors = torch.linspace(-num_grades, num_grades, steps=self.num_grades)
        self.register_buffer("anchors", anchors.view(-1).float())
        # Grade level values g = 0..num_grades-1 (buffer).
        self.register_buffer(
            "grade_levels", torch.arange(self.num_grades, dtype=torch.float32)
        )

        # Learnable temperature via log_tau (tau = exp(log_tau) > 0).
        self.log_tau = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(max(tau_init, 1e-4))))).clone()
        )
        # Learnable James-Stein shrinkage scalar (rho = rho_max * sigmoid(rho_raw)).
        # rho_raw large-negative => rho ~ 0 (no shrinkage) at init.
        self.rho_raw = nn.Parameter(torch.tensor(-4.0))

        # --- Free affine readout (active ONLY when use_anchors=False; see a96). ---
        # Warm-started so y = w_s*s + c_s maps the anchor span [a_min, a_max] -> [0, 3]:
        #   w_s = (G-1) / (a_max - a_min),  c_s = -w_s * a_min.
        a = self.anchors
        a_min = float(a.min())
        a_max = float(a.max())
        span = max(a_max - a_min, 1e-6)
        w_init = (self.num_grades - 1) / span
        c_init = -w_init * a_min
        self.w_s = nn.Parameter(torch.tensor(float(w_init)))
        self.c_s = nn.Parameter(torch.tensor(float(c_init)))

    def _diffuse_score(self, features: torch.Tensor) -> torch.Tensor:
        """s = <mean_i f_i, v> — scalar diffuse fibrosis score (perm/bag-size invariant)."""
        z = features.mean(dim=0)          # [input_dim] DIFFUSE pool (every patch equal)
        return torch.dot(z, self.v)       # scalar

    def _barycentre(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Anchor-pinned barycentre readout. Returns (y_scalar, soft_assignment p)."""
        rho = self.rho_max * torch.sigmoid(self.rho_raw)          # in [0, rho_max)
        centroid = self.anchors.mean()
        a_shrunk = (1.0 - rho) * self.anchors + rho * centroid    # James-Stein contraction
        tau = torch.exp(self.log_tau)                             # > 0
        d = (s - a_shrunk) ** 2                                   # [G] squared distances
        p = torch.softmax(-d / (tau * tau), dim=0)                # [G] soft assignment
        y = torch.dot(self.grade_levels, p)                       # expected grade in [0, G-1]
        return y, p

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] single bag (batch_size=1 trainer contract). Uses ONLY 'features'.
        s = self._diffuse_score(features)             # scalar diffuse score

        if self.use_anchors:
            y, p = self._barycentre(s)                # frozen-anchor barycentre readout
            aux = p.detach()
        else:
            y = self.w_s * s + self.c_s               # FREE affine head (a96 ablation)
            aux = None

        y = y.view(-1)                                # [1] RAW grade logit
        if self.clamp_output:
            y = y.clamp(0.0, float(self.num_grades - 1))

        return y, aux, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    num_grades=4,
    rho_max=0.5,
    tau_init=4.0,
    use_anchors=True,        # a95 main = anchor-pinned barycentre readout
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,      # RAW logits out; trainer rounds+clips at eval
)
