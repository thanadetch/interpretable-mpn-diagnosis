"""a89 — Calibrated ordinal readout on a diffuse-pooled fibrosis scalar.

ROBUSTNESS / VARIANCE angle (not a seed=2-val angle). The session post-mortem
is unambiguous: 86 aggregators + 3 backbones + fusion ALL tie the baseline on
PAIRED cross-fold Δ≈0. The bottleneck is NOT signal — it is VARIANCE /
generalisation (the val<->test gap). Even a84 cleared both seed=2 gates yet was
a seed=2 LOTTERY (lost on 4/5 folds paired). So a89 does not try to extract a
new fibrosis signal; it tries to make the SAME diffuse-density signal read out
more STABLY across folds by replacing the free Linear grade head with a
CALIBRATED MONOTONE ORDINAL readout (a proportional-odds-style cumulative
sigmoid). A free Linear head has an unconstrained slope/intercept that the tiny
training fold can overfit (a slightly mis-set slope inflates the val<->test
gap); the calibrated readout constrains the score->grade map to a smooth,
monotone, threshold-partitioned family whose only fold-fit freedom is WHERE the
grade boundaries sit and HOW SHARP they are — far fewer ways to overfit the map,
hence (hypothesis) lower variance and a smaller val<->test gap.

GRADING PRINCIPLE honoured: grade = OVERALL / DIFFUSE reticulin density across
the whole marrow, NOT a few patches, and NOT weighted by ||h|| (norm-salience
is Spearman~0 with grade => nuisance). So a89 pools DIFFUSELY (plain unweighted
mean over patches) and projects that single diffuse vector onto ONE fibrosis
direction to get a scalar density score. There is NO per-patch scorer, NO
selection, NO top-k, NO attention, NO norm weighting.

Mechanism (exact)
-----------------
    h_i = Dropout(ReLU(Linear(1280->128) f_i))     # baseline-identical encoder
    z   = mean_i h_i                                 # DIFFUSE pool (whole-bag density)
    s   = <z, w> + c                                 # scalar fibrosis density score
    # --- CALIBRATED ORDINAL READOUT (the active ingredient) ---
    b_1 = b1_raw                                      # first grade boundary
    b_2 = b_1 + softplus(d_2)                         # monotone: b_1 < b_2
    b_3 = b_2 + softplus(d_3)                         # monotone: b_2 < b_3
    t   = softplus(t_raw) + eps                       # learnable temperature > 0
    y   = sum_{k=1..3} sigmoid( (s - b_k) / t )       # expected ordinal count in [0,3]

`y` is the expected number of crossed grade boundaries = E[# {k : grade >= k}],
the proportional-odds expected ordinal class, a smooth monotone increasing
function of s in (0, 3). RAW output (no clamp); the trainer rounds+clips at eval.

Why a CALIBRATED ordinal readout should reduce variance vs a free Linear head
-----------------------------------------------------------------------------
  - A free Linear(z->1) head, y = <z,w'> + c', has an arbitrary global slope:
    the SmoothL1 target is satisfied by any affine map, so the fold-specific
    slope is under-constrained and small-fold noise perturbs it, widening the
    val<->test gap. The cumulative-sigmoid head is bounded in [0,3] and its
    map s->y is a fixed monotone S-curve family; the only fitted DOF are the
    three boundary LOCATIONS and one shared SHARPNESS t. Fewer ways to overfit
    the readout => (hypothesis) tighter val<->test agreement.
  - Shared temperature t couples all three boundaries to ONE calibration knob,
    so the model cannot independently over-sharpen one boundary on a lucky fold
    — a built-in regulariser on readout sharpness. This is the explicit
    variance/calibration target, distinct from a per-threshold free head.
  - Monotone thresholds (softplus-cumsum) guarantee b_1<b_2<b_3 at every step,
    so the ordinal geometry can never invert during training on a noisy fold.

Warm-start (train-only, seed=2; pure INITIALISATION, no test info)
------------------------------------------------------------------
The projection w is initialised along the train fibrosis axis (the same
data/prototypes_virchow2_reti_train_seed2.pt['axis'] idiom as a83): W1 @ axis
gives the bottleneck-space response to the fibrosis direction, and w is pointed
along it so s starts as a readout of diffuse fibrosis density. Boundaries start
at (0.5, 1.5, 2.5) (the natural half-integer grade midpoints) and t starts at 1,
so at init y(s) rounds like a sensible regression baseline. Falls back to default
init if the cache is absent (model still constructs).

Distinction from a47/a48 (honest)
----------------------------------
a47 IS a cumulative-link head and it TIED. a47 differs from a89 in mechanism:
  (1) a47's scalar s comes from a learned GATED-ATTENTION aggregator (attn_V,
      attn_U, attn_W -> softmax -> weighted bag), i.e. a few-patch-weighted
      score; a89's scalar is an explicit DIFFUSE-POOL projection z=mean h then
      s=<z,w> — no attention, no weighting, every patch enters equally (the
      grading principle made structural).
  (2) a47 has NO temperature (fixed unit sharpness); a89 adds a learnable
      shared t>0 as the explicit CALIBRATION knob — the variance lever.
  (3) a47 was pitched as an ordinal-bias / val-improvement head; a89 is pitched
      and tested as a VARIANCE / val<->test-gap reducer (the actual bottleneck).
If, despite these, a89 merely reproduces a47's tie (Δ≈0 paired cross-fold), say
so honestly: it confirms the cumulative-link family adds nothing here and the
diffuse-pool scalar route is just another tie. The DoD is PAIRED cross-fold
robustness (val<->test gap), NOT seed=2 val.

NOT in the refuted list: no rank/norm-salience (no ||h|| weighting), no top-k,
no plain coverage/moments/quantiles/pairwise/dist-match (single diffuse pool +
ordinal map, no concatenated stats), no per-patch severity/consensus (no
per-patch scorer), no PMA/learned-query (plain mean), no graph-smooth (no
affinity), no L2-norm head, no ensemble. The fresh element vs a47 is the
diffuse-pool scalar + the temperature-calibrated readout aimed at variance.

Permutation- & bag-size-invariance: z = mean_i h_i is a plain mean over patches
=> permutation-INVARIANT and unchanged by bag duplication (bag-size-invariant).
Everything after z (s, boundaries, t, cumulative sigmoid) is a fixed scalar map,
independent of N and of patch order. Deterministic at inference: the only
stochastic op is Dropout, which is identity under model.eval(); no test-time
randomness, no MC-dropout, no sampling. Two eval passes are bit-identical.

Ablation companion: a90 = plain Linear(z->1) regression head (use_ordinal=False),
importing this Model and flipping ONE flag. a89 (calibrated ordinal) vs a90
(linear head) isolates EXACTLY: "does a calibrated ordinal readout reduce the
val<->test variance vs a free linear head on the same diffuse-pooled scalar?"
The bottleneck encoder and the diffuse pool are byte-identical between them.

Param count (input_dim=1280, hidden=128)
----------------------------------------
    bottleneck Linear(1280,128)+b = 163,968
    proj_w     [128] + bias c     =     129   (s = <z,w> + c)
    b1_raw, d2_raw, d3_raw        =       3   (monotone boundaries)
    t_raw                         =       1   (learnable temperature)
    ----------------------------------------------
    total (a89, ordinal)          = 164,101   (< 197,250 baseline)
    total (a90, linear ablation)  = 164,097   (proj_w head reused, ordinal off)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_axis(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    """Unit fibrosis axis (1280-d) from train-only prototypes (seed=2).

    Returns None if the cache is absent so the model still constructs (the
    projection then uses its default init). Uses ONLY the train-derived axis —
    no test information, no runtime features.
    """
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    v = blob["axis"].float().view(-1)
    if v.numel() != input_dim:
        return None
    return v / v.norm().clamp(min=1e-8)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        use_ordinal: bool = True,        # a90 ablation flips this to False (plain Linear)
        # Boundaries start at (0.5, 1.5, 2.5) = half-integer grade midpoints.
        b1_init: float = 0.5,
        delta_init: float = 0.5413248,   # softplus(0.5413) ≈ 1.0 => spacing 1.0
        temp_init: float = 1.0,          # initial calibration temperature t (>0)
        temp_eps: float = 1e-3,          # floor so t stays strictly > 0
        warm_start: bool = True,         # init projection along train fibrosis axis
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,      # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        self.use_ordinal = bool(use_ordinal)
        self.clamp_output = bool(clamp_output)
        self.temp_eps = float(temp_eps)

        # Baseline-identical encoder (this is where the capacity lives).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Scalar fibrosis density score s = <z, w> + c on the diffuse-pooled z.
        # Shared by BOTH the ordinal readout (a89) and the linear ablation (a90):
        # when use_ordinal=False, y = s directly (a plain Linear(z->1) head).
        self.score = nn.Linear(hidden_dim, 1)

        # Calibrated ordinal readout parameters (active only when use_ordinal).
        # Monotone boundaries b_1 < b_2 < b_3 via softplus-cumsum; one shared
        # learnable temperature t > 0 = the explicit calibration / variance knob.
        self.b1_raw = nn.Parameter(torch.tensor(float(b1_init)))
        self.delta_2_raw = nn.Parameter(torch.tensor(float(delta_init)))
        self.delta_3_raw = nn.Parameter(torch.tensor(float(delta_init)))
        # invert softplus so softplus(t_raw) + eps == temp_init at start.
        t0 = max(float(temp_init) - self.temp_eps, 1e-4)
        self.t_raw = nn.Parameter(torch.tensor(torch.expm1(torch.tensor(t0)).clamp(min=1e-6).log().item()))

        # Warm-start the projection toward the fibrosis axis (train-only, seed=2).
        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            axis = _load_fibrosis_axis(path, input_dim)
            if axis is not None:
                with torch.no_grad():
                    w1 = self.bottleneck[0].weight.detach()        # [hidden, input]
                    resp = w1 @ axis                               # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    # Unit-ish slope: s spans ~[0,3] across the fibrosis axis,
                    # matching the half-integer boundary init (0.5,1.5,2.5).
                    self.score.weight.copy_(resp.view(1, -1) * 3.0)
                    self.score.bias.fill_(1.5)                     # mid-grade start

    def _ordinal_readout(self, s: torch.Tensor) -> torch.Tensor:
        """s: scalar tensor -> y = sum_k sigmoid((s - b_k)/t), the expected
        ordinal count in (0, 3). Monotone boundaries + shared temperature t>0.
        """
        b1 = self.b1_raw
        b2 = b1 + F.softplus(self.delta_2_raw)
        b3 = b2 + F.softplus(self.delta_3_raw)
        t = F.softplus(self.t_raw) + self.temp_eps                 # strictly > 0
        y = (
            torch.sigmoid((s - b1) / t)
            + torch.sigmoid((s - b2) / t)
            + torch.sigmoid((s - b3) / t)
        )
        return y

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] single bag (batch_size=1 trainer contract).
        h = self.bottleneck(features)              # [N, hidden]
        z = h.mean(dim=0)                          # [hidden] DIFFUSE pool
        s = self.score(z).view(-1)                 # [1] scalar fibrosis score

        if self.use_ordinal:
            y = self._ordinal_readout(s).view(-1)  # [1] calibrated ordinal count
        else:
            y = s                                  # [1] plain Linear(z->1) head (a90)

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    use_ordinal=True,        # a89 main = calibrated monotone cumulative-sigmoid readout
    b1_init=0.5,
    delta_init=0.5413248,    # -> boundary spacing 1.0 => (0.5, 1.5, 2.5)
    temp_init=1.0,
    temp_eps=1e-3,
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,      # RAW logits out; trainer rounds+clips at eval
)
