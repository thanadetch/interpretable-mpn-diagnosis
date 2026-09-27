"""a99 — Closed-form SPECTRAL low-frequency readout (leading non-trivial graph
mode) BLENDED with the diffuse mean. Round-4 main.

LENS: spectral-graph. Reticulin grade is the OVERALL / DIFFUSE fibre density of a
*coherent* meshwork spanning the whole marrow, NOT a property of a few patches and
NOT a feature-magnitude (||h|| Spearman ~0 with grade => nuisance, never weight by
it). a99 reads out, in CLOSED FORM, the AMPLITUDE OF THE SINGLE DOMINANT COHERENT
TISSUE MODE — the leading NON-TRIVIAL eigenvector of the CENTERED patch-similarity
(Gram) operator — and blends it onto the diffuse mean. The centering removes the
trivial DC / all-ones eigenvector (which IS the mean), so the spectral term is a
genuinely new lever orthogonal to the diffuse pool: it is large when many patches
co-vary together (a diffuse, contiguous meshwork) and small when the bag is just a
centroid plus incoherent per-patch noise (lone bone fragments, edge/stain
artefacts, which sit in the high-frequency tail and project weakly onto the top
mode => suppressed by construction).

================================================================================
WHY THIS IS ORTHOGONAL TO EVERYTHING TRIED (the load-bearing argument)
================================================================================
A plain MEAN-POOL is EXACTLY the projection of the bag onto the TRIVIAL all-ones /
DC eigenvector (the stationary mode of a row-stochastic affinity). a83's iterative
smoothing h <- (1-a)h + a(A@h) is a polynomial low-pass whose fixed point IS that
DC mode — which is why it COLLAPSED to the mean (a83<=a84 tied). a59/a62 weight by
cosine-to-CENTROID, i.e. agreement with the DC mode again. a99 does the OPPOSITE:
it DEFLATES the DC mode (centers the unit patch directions so the all-ones
direction is removed) and reads the LEADING REMAINING eigenvector u of the centered
Gram G = Ec Ec^T. u is the soft membership of each patch in the most coherent fibre
subspace AFTER the bag mean is taken out — the principal axis of CO-VARIATION among
patches, not the location of the bag.

  - NOT mean-pool / diffuse pool (a14/a88/a92): the DC/mean mode is explicitly
    REMOVED (centering) before the spectral readout. We still ADD the mean back as
    z_dc so diffuse density is never lost, but the NEW lever is purely the
    non-trivial mode. The a100 ablation (gamma=0) recovers EXACTLY this null.
  - NOT iterative graph-smoothing (a83): no residual mixing, no learnable mixing
    coeff. Power iteration here estimates an EIGENVECTOR of a FIXED operator
    (closed-form spectral filter); the result is a POOLING-WEIGHT PATTERN u, not a
    rewritten feature field that is then mean-pooled.
  - NOT consensus / cosine-to-centroid (a59/a62): those measure agreement with the
    DC mode (the centroid). a99 removes the centroid and reads the leading
    NON-TRIVIAL mode — a different eigenvector (Fiedler-style direction of
    agreement, not the location).
  - NOT rank / norm-salience: ||h|| never enters; the Gram uses L2-NORMALISED patch
    directions (cosine geometry) => magnitude-invariant.
  - NOT top-k / argmax / selection: u_i is a dense, signed, soft membership; no
    patch is dropped. NOT robust-trimmed / geometric-median.
  - NOT per-patch severity / coverage / threshold / fraction: no per-patch scalar
    scorer, no sigmoid fraction, no count above a cutoff.
  - NOT moments / quantiles / soft-rank / pairwise-Gini / dist-match: z_spec is a
    single bilinear (eigen)projection, not a moment field, order statistic,
    pairwise spread, or histogram match.
  - NOT PMA / set-transformer / learned-query attention: the pooling weight u is a
    PARAMETER-FREE data-driven eigenvector of the patch Gram, not a learned
    seed/query/softmax scorer.
  - NOT cumulative-link / ordinal head: the readout is a single Linear -> raw
    scalar. NOT heavy-dropout/input-noise (baseline dropout only).
  - NOT 2/3-way backbone fusion / multi-mechanism ensemble: one backbone, one
    pooled vector, one head.

================================================================================
FORWARD-PASS MATH (exact)
================================================================================
features f in R^{N x 1280} (one bag).
  h_i    = Dropout(ReLU(W1 f_i + b1))            in R^{hidden}   # baseline encoder
  z_dc   = mean_i h_i                            in R^{hidden}   # DC / diffuse mode
  -- cosine geometry (magnitude-invariant), built on DETACHED encoder features so
     u is a data-driven readout PATTERN we do not backprop the eigensolver through:
  e_i    = h_i.detach() / ||h_i.detach()||       in R^{hidden}   # unit directions
  ec_i   = e_i - mean_i(e)                        # CENTER => remove trivial DC mode
  -- leading NON-TRIVIAL spectral mode of the centered patch Gram G = Ec Ec^T by
     T=8 FIXED power-iteration steps WITHOUT materialising G (O(N*hidden)):
       v <- deterministic seed (max centered-norm patch direction; fallback ec_0)
       repeat T times:
           proj = Ec @ v          in R^N          #  u-side iterate
           u    = proj / ||proj||                 # [N] current mode estimate
           v    = Ec^T @ u        in R^{hidden}    # feature-side iterate
           v    = v / ||v||
     u is the unit leading eigenvector (||u||=1) of G; u DETACHED.
  m_i    = sqrt(N) * u_i                          # [N] bag-size-normalised membership
  -- DETERMINISTIC SIGN FIX (eigenvector sign is arbitrary => must be pinned):
     s = sign( (1/N) sum_i m_i * (f_i . fib_axis) )   # seed=2 train axis, fixed
     if s == 0: s = sign(mean_i m_i)                  # fallback
     m <- s * m
  -- spectral amplitude in ENCODER space, pooled as a MEAN (bag-size-invariant);
     hc uses the NON-detached h so gradients reach the encoder via the amplitude,
     while m stays a fixed data-driven weight pattern:
       z_spec = (1/N) sum_i m_i * (h_i - z_dc)    in R^{hidden}
  -- blend (learnable scalar gamma, NOT a gate on the scalar; it mixes two FEATURE
     vectors before the single linear head):
       gamma  = sigmoid(gamma_logit)  in (0,1), INIT 0.1   # start near diffuse pool
       z      = z_dc + gamma * z_spec
  y      = w_head . z + b_head                    in R              # RAW scalar logit

The head is WARM-STARTED along the train-only (seed=2) fibrosis axis: w_head is set
to the (normalised) bottleneck response W1 @ axis, so y starts as a linear readout
of the diffuse fibrosis density. Pure INITIALISATION (then fully learned); injects
no test information, uses only the train-derived axis.

Permutation-invariance: z_dc and e_bar are means (symmetric); G and its top
eigenvector are permutation-EQUIVARIANT (permuting patches permutes u consistently),
and z_spec = (1/N) sum_i m_i hc_i is a permutation-INVARIANT contraction. The sign
fix uses only symmetric sums. => y is permutation-invariant.
Bag-size-invariance: e_i are unit (scale-free in N); u is normalised to ||u||=1;
membership m_i = sqrt(N) u_i is pooled as a MEAN (1/N sum). Duplicating the bag
shrinks each u_i by 1/sqrt(2) but doubles the count; the sqrt(N) rescale and the
1/N mean exactly cancel, so z_spec is invariant; z_dc is a mean. Deterministic at
inference: Dropout off, FIXED T power-iteration steps with a deterministic seed and
a deterministic sign fix — no sampling, no float-tie branching beyond the documented
fallbacks. (N<=1: the centered Gram is degenerate => spectral term is skipped and
z = z_dc, so single-patch bags stay finite and well-defined.)

================================================================================
MAIN vs ABLATION
================================================================================
a99 (MAIN) = gamma learnable, INIT 0.1: starts ~mean-pool, can ADD the leading
centered-graph spectral mode if it helps.
a100 (ABLATION) = gamma_fixed=0.0 (constant buffer, NO gamma_logit param) => z =
z_dc = plain diffuse mean-pool + a BYTE-IDENTICAL head. a100 imports a99's Model and
only flips gamma_fixed. a99 vs a100 isolates EXACTLY: "does adding the leading
centered-graph spectral mode beat the diffuse mean?"

Kill criterion: abandon if a99 <= a100 on PAIRED Δ across most of folds
{0,1,2,3,42} (both val & test). Multi-fold paired audit, NOT seed=2 alone.

Param count (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    head       Linear(128,1)+b     =    129
    gamma_logit scalar             =      1
    --------------------------------------------
    total                          = 164,098   (< ~197K cap)
The spectral readout (centering, power iteration, sign fix, projection) adds ZERO
learnable parameters. (a100 has 164,097 — the gamma_logit becomes a buffer.)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AXIS_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_axis(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    """Unit fibrosis axis (1280-d) from the seed=2 TRAIN-ONLY prototype cache.

    Returns None if absent so the model still constructs. Train-only, init/
    inference-FIXED use only; injects no test information, no runtime features.
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
        power_iters: int = 8,            # T: fixed deterministic power-iteration steps
        gamma_init: float = 0.1,         # initial spectral blend (start near diffuse pool)
        gamma_fixed: Optional[float] = None,  # a100 ablation sets 0.0 (spectral OFF)
        warm_start: bool = True,         # init head along train fibrosis axis (seed=2)
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,      # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert power_iters >= 1, "power_iters (T) must be >= 1."
        assert 0.0 < float(gamma_init) < 1.0, "gamma_init must be in (0,1)."
        self.power_iters = int(power_iters)
        self.clamp_output = bool(clamp_output)
        self.gamma_fixed = None if gamma_fixed is None else float(gamma_fixed)
        if self.gamma_fixed is not None:
            assert 0.0 <= self.gamma_fixed <= 1.0, "gamma_fixed must be in [0,1]."

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Single linear readout from the blended (DC + spectral) representation.
        self.head = nn.Linear(hidden_dim, num_classes)

        # Learnable spectral blend gamma = sigmoid(gamma_logit), INIT 0.1 so we START
        # at the diffuse pool and only add the coherent mode if it helps. The ablation
        # (gamma_fixed=0) registers a constant buffer instead so NO gamma parameter is
        # trained and z == z_dc EXACTLY (byte-identical head, plain mean-pool).
        if self.gamma_fixed is None:
            gi = math.log(float(gamma_init) / (1.0 - float(gamma_init)))
            self.gamma_logit = nn.Parameter(torch.tensor(gi, dtype=torch.float32))
        else:
            self.register_buffer(
                "gamma_const", torch.tensor(self.gamma_fixed, dtype=torch.float32)
            )

        # Fibrosis axis for the deterministic SIGN FIX of the eigenvector (and for
        # warm-starting the head). Train-only (seed=2); fixed buffer, no trainable params.
        path = Path(prototype_path) if prototype_path else _DEFAULT_AXIS_PATH
        axis = _load_fibrosis_axis(path, input_dim)
        if axis is not None:
            self.register_buffer("fib_axis", axis)
        else:
            self.fib_axis = None

        # Warm-start the head toward the fibrosis axis (train-only, seed=2).
        if warm_start and axis is not None:
            with torch.no_grad():
                w1 = self.bottleneck[0].weight.detach()        # [hidden, input]
                resp = w1 @ axis                                # [hidden] bottleneck response
                resp = resp / resp.norm().clamp(min=1e-8)
                self.head.weight.copy_(resp.view(1, -1) * 3.0)
                self.head.bias.fill_(1.5)                       # mid-grade start

    def _gamma(self) -> torch.Tensor:
        if self.gamma_fixed is None:
            return torch.sigmoid(self.gamma_logit)
        return self.gamma_const

    def _leading_mode(self, ec: torch.Tensor) -> torch.Tensor:
        """Leading eigenvector u [N] (||u||=1) of the centered patch Gram
        G = Ec Ec^T, by T fixed deterministic power-iteration steps WITHOUT
        materialising G. ec: [N, hidden] CENTERED unit-direction patch features.

        Iteration (feature-space, O(N*hidden)):
            proj = Ec v ; u = proj/||proj|| ; v = Ec^T u ; v = v/||v||.
        Detached: u is a data-driven readout PATTERN, not a path we backprop the
        eigensolver through (the head still backprops through the encoder via the
        z_dc and z_spec amplitudes).
        """
        n = ec.size(0)
        ecd = ec.detach()
        # Deterministic seed: the patch with the largest centered norm (most
        # off-centroid coherent direction); argmax tie-breaks to the lowest index,
        # and an all-zero ec falls back to a uniform u below.
        norms = ecd.norm(dim=1)                                 # [N]
        seed_idx = int(torch.argmax(norms).item())
        v = ecd[seed_idx].clone()                               # [hidden]
        v = v / v.norm().clamp(min=1e-8)
        u = ecd.new_full((n,), 1.0 / math.sqrt(n))              # safe uniform fallback [N]
        for _ in range(self.power_iters):
            proj = ecd @ v                                      # [N]  = Ec v
            nrm = proj.norm().clamp(min=1e-8)
            u = proj / nrm                                      # [N] current mode estimate
            v = ecd.t() @ u                                     # [hidden] = Ec^T u
            v = v / v.norm().clamp(min=1e-8)
        return u                                                # [N], ||u||=1

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract). forward uses
        # ONLY 'features'; no labels, no test-time info.
        h = self.bottleneck(features)                           # [N, hidden]
        n = h.size(0)

        z_dc = h.mean(dim=0)                                    # [hidden] DIFFUSE / DC mode

        gamma = self._gamma().to(h.dtype)
        # Spectral term needs >=2 distinct patches; centering a single patch gives
        # ec==0 (degenerate Gram). Skip => z = z_dc for tiny bags (stays finite).
        use_spec = (self.gamma_fixed != 0.0) and (n > 1)
        if use_spec:
            # Cosine geometry (magnitude-invariant) on DETACHED encoder features.
            e = F.normalize(h.detach(), dim=1, eps=1e-8)        # [N, hidden] unit dirs
            ec = e - e.mean(dim=0, keepdim=True)                # CENTER => remove DC mode
            u = self._leading_mode(ec)                          # [N] unit eigenvector ||u||=1

            # Bag-size normalisation: unit eigenvector entries are O(1/sqrt(N)); the
            # sqrt(N) rescale makes m_i O(1) so the MEAN pool below is bag-size /
            # duplication invariant.
            mw = math.sqrt(float(n)) * u                        # [N] membership weights

            # Deterministic SIGN FIX: the eigenvector sign is arbitrary, so pin it so
            # "more coherent fibre" is the positive direction. Use the MEAN of the
            # (bag-size-invariant) membership * per-patch fibre projection.
            if self.fib_axis is not None:
                fib = features @ self.fib_axis                  # [N] per-patch fibre proj
                align = torch.dot(mw, fib) / float(n)           # bag-size-invariant scalar
            else:
                align = mw.mean()                               # fallback: sign(mean m)
            s = torch.sign(align)
            s = torch.where(s == 0, torch.ones_like(s), s)      # 0 -> +1 fallback
            mw = mw * s

            # Project the CENTERED encoder representation onto the coherent mode as a
            # MEAN over patches. hc uses the NON-detached h so gradients reach the
            # encoder via the amplitude, while mw stays a fixed data-driven pattern.
            hc = h - z_dc                                       # [N, hidden] centered enc
            z_spec = torch.mv(hc.t(), mw) / float(n)            # [hidden] mode amplitude (mean)
            attn = mw                                           # report membership weights
            z = z_dc + gamma * z_spec
        else:
            attn = None
            z = z_dc

        y = self.head(z).view(-1)                               # [1] RAW grade logit
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    power_iters=8,         # a99 main: fixed closed-form spectral filter (T=8 steps)
    gamma_init=0.1,        # learnable gamma init 0.1 (start near diffuse pool)
    gamma_fixed=None,      # learnable gamma (a100 ablation sets 0.0 => spectral OFF)
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
