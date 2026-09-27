"""a93 — Calibrated PROTOTYPE-ANCHOR barycentre readout (shrinkage-pinned map).

VARIANCE / PAIRED-FOLD angle, NOT a seed=2-val angle. The documented session
post-mortem: 90+ aggregators + 3 backbones + fusion ALL tie the baseline on
PAIRED cross-fold Δ≈0; the bottleneck is VARIANCE / generalisation on a tiny
cohort, not signal (a84 cleared both seed=2 gates yet LOST 4/5 folds paired).
So a93 does not invent a new pooling operator or a new fibrosis signal. It keeps
the SAME diffuse-density signal (plain mean over patches, projected on the
fibrosis direction) and changes ONLY how the scalar is mapped to a grade: it
PINS the score->grade map to the four FROZEN, train-only per-grade PROTOTYPE
anchor coordinates and reads out the BARYCENTRE (expected grade) under a
shrinkage-regularised soft assignment. There is NO free linear head slope/
intercept, and NO free fitted boundaries — the entire map is determined by data
(train prototypes), with a single learnable temperature as the only calibration
DOF. That removed slope/intercept freedom is precisely the fold-overfit lever
this design attacks.

GRADING PRINCIPLE honoured: grade = OVERALL / DIFFUSE reticulin density across
the whole marrow, NOT a few patches, and NOT weighted by ||h|| (norm-salience is
Spearman~0 => nuisance). So a93 pools DIFFUSELY (plain unweighted mean over
patches) and projects that ONE diffuse vector onto ONE fibrosis direction.
There is NO per-patch scorer, NO selection, NO top-k, NO attention, NO norm
weighting, NO affinity / graph step.

Why anchor in 1-D and not full 1280-d (numerically verified)
------------------------------------------------------------
The four raw 1280-d prototypes are NEARLY EQUIDISTANT from any bag-mean (squared
Euclidean distance in 1280-d is swamped by isotropic nuisance variance => the
softmax over full-D prototype distances collapses to ~uniform, E[grade]~1.5 for
every bag — verified numerically). The discriminative structure lives almost
entirely along the train fibrosis axis: the prototypes' axis coordinates are
(-11.0, -4.0, +1.4, +8.5) — cleanly monotone and well separated. So a93 measures
distance to each prototype ONLY along the calibrated fibrosis axis (a supervised-
direction Mahalanobis collapse), where the anchors are informative and the
readout is monotone and well-resolved across the whole grade range.

Mechanism (exact; single bag features f in R^{N x 1280})
--------------------------------------------------------
    z      = mean_i f_i                      # [1280] DIFFUSE pool (whole-bag density)
    s      = <z, v>                          # scalar diffuse fibrosis score
                                             # v = learnable unit-ish fibrosis direction,
                                             #     warm-started at train axis (seed=2).
    a_g    = <P_g, v0>   (g = 0..3)          # [4] FROZEN prototype anchor coordinates,
                                             #     v0 = the train axis at INIT (a buffer),
                                             #     a_g precomputed once (train-only).
    # --- shrinkage toward the global prototype centre (variance regulariser) ---
    abar   = mean_g a_g                      # global anchor centroid (scalar)
    a_g'   = (1 - rho) * a_g + rho * abar     # shrink anchors toward centroid, rho in [0,1)
    # --- calibrated barycentric soft assignment ---
    d_g    = (s - a_g')^2                     # [4] squared 1-D distance to each anchor
    p_g    = softmax_g( -d_g / tau^2 )        # [4] soft grade assignment, tau > 0 learnable
    y      = sum_g g * p_g                    # BARYCENTRE = expected grade in [0, 3]

RAW output (no clamp); the trainer rounds+clips at eval. `y` is a smooth,
strictly monotone increasing function of s over [a_0', a_3'] (a soft-nearest-
anchor interpolation), bounded in [0, 3] by construction — it CANNOT run off to
an arbitrary slope the way a free Linear(z->1) head can.

Why a93's readout should reduce val<->test VARIANCE vs a free linear head
-------------------------------------------------------------------------
  - A free Linear(z->1) head y = <z,w'> + c' has an unconstrained global slope
    AND intercept: SmoothL1 is satisfied by any affine map, so the fold-specific
    slope/intercept are under-constrained and small-fold noise perturbs them,
    widening the val<->test gap. a93's score->grade map has ZERO free slope and
    ZERO free intercept: the four grade levels are PINNED to the train prototype
    anchor coordinates a_g (data, not fold-fit), so the only fitted DOF in the
    readout is ONE shared temperature tau (how soft the assignment is). Far fewer
    ways to overfit the map => (hypothesis) tighter val<->test agreement.
  - Shrinkage `rho` pulls the anchors toward their common centroid, a James-
    Stein-style contraction of the calibration targets that trades a little
    train fit for lower variance; rho is a single learnable scalar (squashed to
    [0, rho_max)) so the model picks how much to trust the raw anchor spread vs
    the shrunk one. This is the explicit variance knob, distinct from a free head.
  - The direction v stays NEAR the train axis: it is warm-started there and the
    encoder capacity that previously absorbed slope freedom (the bottleneck MLP)
    is REMOVED — there is no hidden ReLU layer for the fold to overfit, only the
    1280-d projection v. Lower capacity, fewer fold-specific DOF.

Warm-start (train-only, seed=2; pure INITIALISATION / frozen anchors, no test info)
-----------------------------------------------------------------------------------
v is initialised at data/prototypes_virchow2_reti_train_seed2.pt['axis'] (unit,
seed=2 train-only) and then learned. The anchor coordinates a_g are computed ONCE
at construction as <P_g, v0> with v0 = the axis at init, and stored as a FROZEN
buffer — they are train-derived calibration constants, never updated, never
touched by val/test. Injects NO test information. Falls back to a deterministic
synthetic monotone anchor set + a random unit direction if the cache is absent
(model still constructs; warm-start benefit lost).

Distinction from the refuted list (honest)
-------------------------------------------
  - NOT a74 dist-match: a74 builds the bag's per-patch soft-QUANTILE vector and
    matches it to per-grade reference QUANTILE distributions of per-patch scores.
    a93 has NO per-patch score distribution and NO quantiles: it uses the single
    DIFFUSE bag-mean projection s and compares it to four scalar prototype
    anchors. Different object (one bag-mean scalar vs a per-patch distribution),
    different reference (4 prototype anchor points vs per-grade quantile curves).
  - NOT a89/a47 cumulative-link: a89 fits THREE free monotone boundaries + a
    temperature (a free, fold-slidable S-curve). a93 fits NO boundaries — the
    grade levels are PINNED at the frozen prototype anchors a_g; the only DOF is
    one temperature + one shrinkage scalar. The readout family is anchored to
    data, not fitted.
  - NOT a25 prototype-as-attention: a25 used a prototype DIRECTION as a per-patch
    softmax attention SCORER then weighted-mean. a93 uses prototypes as CALIBRATED
    GRADE ANCHORS for the bag-level readout; there is no attention, no per-patch
    weighting.
  - NOT mean-pool + linear (a53/a92): those map the bag mean through a FREE
    affine head. a93 maps it through the FROZEN-anchor barycentre (no free slope/
    intercept). The a94 ablation isolates exactly this difference.
  - NOT coverage/moments/pairwise/PMA/graph-smooth/L2-norm/ensemble/heavy-reg:
    no fraction/threshold, no concatenated stats, no pairwise term, no learned
    query, no affinity, no norm weighting, no model averaging, no input noise.

Permutation- & bag-size-invariance: z = mean_i f_i is a plain mean over patches
=> permutation-INVARIANT and unchanged by bag duplication (bag-size-invariant).
Everything after z (s, anchors, shrinkage, softmax, barycentre) is a fixed scalar
map independent of N and patch order. Deterministic at inference: there is NO
Dropout and NO stochastic op anywhere; two eval passes are bit-identical. N=1 is
safe (z = f_0; the readout is well-defined for any scalar s).

Ablation companion: a94_prototype_barycentre_freehead.py — imports this Model and
flips `use_anchors=False`, replacing the FROZEN-anchor barycentre readout with a
FREE Linear(1) head on the SAME diffuse score s (y = w_s * s + c_s, w_s/c_s
learnable). The diffuse pool and the projection v are byte-identical between
a93/a94, so a93 (frozen-anchor barycentre) vs a94 (free affine head) isolates
EXACTLY the active ingredient = "does pinning the score->grade map to the train
prototype anchors (no free slope/intercept) reduce the val<->test gap vs a free
affine readout on the same diffuse score?".

Kill criterion: abandon if a93 <= a94 on PAIRED cross-fold Δ (the frozen-anchor
readout adds no robustness over a free head). DoD = PAIRED cross-fold audit over
most of {0,1,2,3,42}, both val & test — NOT seed=2 val alone.

Param count (input_dim=1280)
----------------------------
    v          [1280]  learnable fibrosis direction        = 1280
    log_tau    scalar  (tau = exp(log_tau) > 0)            =    1
    rho_raw    scalar  (rho = rho_max * sigmoid(rho_raw))  =    1
    -----------------------------------------------------------------
    total (a93)                                            = 1282   (<< 197,250)
    a94 adds a free Linear(1->1): w_s, c_s = +2 params      = 1284
Anchors a_g and v0 are FROZEN buffers (0 trainable params). This is an extremely
low-capacity, deterministic readout aimed squarely at variance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_axis_and_anchors(
    path: Path, input_dim: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Train-only (seed=2) unit fibrosis axis v0 [D] and prototype anchor
    coordinates a_g = <P_g, v0> [4], precomputed once.

    Returns None (caller falls back to a deterministic synthetic set) if the
    cache is missing or malformed, so the module never hard-crashes a screen.
    Uses ONLY the train-derived axis + prototypes — no test info, no labels.
    """
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        v0 = blob["axis"].float().view(-1)
        if v0.numel() != input_dim:
            return None
        v0 = v0 / v0.norm().clamp(min=1e-8)
        protos = blob["prototypes"]
        P = torch.stack([protos[g].float().view(-1) for g in range(4)], dim=0)  # [4, D]
        if P.shape != (4, input_dim):
            return None
        anchors = P @ v0  # [4] prototype coordinates along the fibrosis axis
        return v0, anchors
    except Exception:
        return None


def _synthetic_axis_and_anchors(
    input_dim: int, random_seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic fallback: a random unit direction + monotone anchors that
    map roughly to grades 0..3 (warm-start benefit lost, interface stays sane).
    """
    g = torch.Generator().manual_seed(int(random_seed))
    v0 = torch.randn(input_dim, generator=g)
    v0 = v0 / v0.norm().clamp(min=1e-8)
    # Spread anchors so the barycentre readout is monotone and well-resolved.
    anchors = torch.tensor([-9.0, -3.0, 3.0, 9.0], dtype=torch.float32)
    return v0, anchors


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        use_anchors: bool = True,        # a94 ablation flips to False (free affine head)
        warm_start: bool = True,         # init v at the train fibrosis axis (seed=2)
        prototype_path: Optional[str] = None,
        tau_init: float = 4.0,           # initial assignment temperature (>0, learnable)
        tau_eps: float = 1e-3,           # floor so tau stays strictly > 0
        rho_max: float = 0.5,            # max anchor shrinkage toward centroid
        rho_init: float = 0.1,           # initial shrinkage fraction in [0, rho_max)
        random_seed: int = 2,            # used only if warm_start=False or cache missing
        clamp_output: bool = False,      # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        self.use_anchors = bool(use_anchors)
        self.clamp_output = bool(clamp_output)
        self.tau_eps = float(tau_eps)
        self.rho_max = float(rho_max)
        self.input_dim = int(input_dim)

        loaded = (
            _load_axis_and_anchors(
                Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH,
                self.input_dim,
            )
            if warm_start
            else None
        )
        if loaded is None:
            v0, anchors = _synthetic_axis_and_anchors(self.input_dim, random_seed)
        else:
            v0, anchors = loaded

        # Learnable fibrosis direction (warm-started at v0; then free to adapt).
        self.v = nn.Parameter(v0.clone())

        # FROZEN train-only anchors + the axis at init (buffers => 0 trainable).
        self.register_buffer("anchors", anchors)                    # [4]
        self.register_buffer("grades", torch.arange(4, dtype=torch.float32))  # [4]

        # Learnable assignment temperature tau = exp(log_tau) > 0.
        self.log_tau = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(max(tau_init, 1e-3)))))
        )

        # Learnable shrinkage rho = rho_max * sigmoid(rho_raw) in [0, rho_max).
        r0 = float(min(max(rho_init / max(self.rho_max, 1e-8), 1e-4), 1.0 - 1e-4))
        self.rho_raw = nn.Parameter(torch.logit(torch.tensor(r0)))

        # a94 ablation: a FREE affine head on the same diffuse score s.
        # Defined always (so the two files share an identical Model class) but
        # used ONLY when use_anchors=False. Warm-started near a sensible
        # score->grade slope so the ablation is a fair baseline.
        self.free_head = nn.Linear(1, 1)
        with torch.no_grad():
            # anchors span ~ [a0, a3]; a unit-ish slope mapping that span to [0,3].
            span = (anchors[-1] - anchors[0]).abs().clamp(min=1e-3)
            self.free_head.weight.fill_(float(3.0 / span))
            self.free_head.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] single bag (batch_size=1 trainer contract).
        # ONLY `features` is consumed (+ frozen train-only anchors).
        z = features.mean(dim=0)                       # [D] DIFFUSE pool (whole-bag density)
        s = torch.dot(z, self.v)                       # scalar diffuse fibrosis score

        if self.use_anchors:
            # Shrink anchors toward their centroid (James-Stein-style contraction).
            rho = self.rho_max * torch.sigmoid(self.rho_raw)        # [0, rho_max)
            abar = self.anchors.mean()
            anchors = (1.0 - rho) * self.anchors + rho * abar       # [4] shrunk anchors
            # Calibrated barycentric soft assignment over the 4 frozen grade levels.
            tau = self.log_tau.exp().clamp(min=self.tau_eps)        # > 0
            d = (s - anchors) ** 2                                  # [4] squared 1-D dist
            p = torch.softmax(-d / (tau * tau), dim=0)              # [4] soft assignment
            y = (p * self.grades).sum().view(-1)                    # [1] BARYCENTRE / E[grade]
        else:
            # a94 ablation: free affine readout on the same diffuse score.
            p = None
            y = self.free_head(s.view(1, 1)).view(-1)               # [1] w_s * s + c_s

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, (p.detach() if p is not None else None), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    use_anchors=True,        # a93 main = frozen prototype-anchor barycentre readout
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    tau_init=4.0,
    tau_eps=1e-3,
    rho_max=0.5,
    rho_init=0.1,
    random_seed=2,
    clamp_output=False,      # RAW logits out; trainer rounds+clips at eval
)
