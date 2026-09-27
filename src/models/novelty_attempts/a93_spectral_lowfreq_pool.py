"""a93 — Closed-form spectral LOW-FREQUENCY readout (leading non-trivial graph mode).

LENS: spectral-graph. Reticulin grade is the OVERALL / DIFFUSE fibre density of a
*coherent* meshwork spanning the marrow, not a few patches. Build the patch-
similarity graph and read out the DOMINANT COHERENT TISSUE MODE — the leading
NON-TRIVIAL eigenvector of the centered patch-similarity (Gram) operator — in
CLOSED FORM (a fixed number of deterministic power-iteration steps = a fixed
spectral filter, NOT a learnable iterative smoother). High-frequency outlier
patches (lone bone fragments, edge/stain artefacts) live in the high-frequency
tail of the spectrum and are SUPPRESSED by construction; the smooth low-frequency
component that the whole tissue shares is what we pool.

================================================================================
WHY THIS IS ORTHOGONAL TO EVERYTHING TRIED (this is the load-bearing argument)
================================================================================
The single most important fact: a plain MEAN-POOL is *exactly* the projection of
the bag onto the TRIVIAL all-ones / DC eigenvector (the zeroth graph mode, the
stationary distribution of a row-stochastic affinity). a83's iterative smoothing
h <- (1-a)h + a(A@h) is a Neumann/polynomial low-pass whose fixed point IS that
DC mode — which is precisely why it COLLAPSED to the mean (a83<=a84 tied). a59/a62
weight by cosine-to-CENTROID, i.e. agreement with the DC mode again.

a93 deliberately does the OPPOSITE: it DEFLATES the DC mode (centers the patch
features so the all-ones direction is removed) and reads the LEADING REMAINING
eigenvector u of the centered Gram G = Hc Hc^T. u is the soft membership of each
patch in the single most coherent fibre subspace AFTER the bag mean is taken out.
This is a genuinely different object from the mean and from consensus-to-centroid:
it is the principal axis of *co-variation* among patches, not the location of the
bag. The pooled bag rep is the projection of the (centered) bag onto that mode:

    z_spec = sum_i u_i * hc_i      (the low-frequency / smooth component amplitude)

so a93 reads the AMPLITUDE OF THE DOMINANT COHERENT MODE, which is large when many
patches co-vary together (a diffuse meshwork) and small when the signal is just
the centroid plus incoherent per-patch noise.

  - NOT mean-pool / diffuse pool (a14/a88/a92): the DC/mean mode is explicitly
    REMOVED (centering) before the readout; z_spec is the FIRST non-trivial mode,
    orthogonal to the mean. We still ADD the mean back as a separate term (so we
    never lose diffuse density), but the new lever is purely the spectral mode.
  - NOT iterative graph-smoothing (a83): no h <- (1-a)h + a(A@h) residual mixing,
    no learnable mixing coeff. Power iteration here estimates an EIGENVECTOR of a
    FIXED operator (closed-form spectral readout); the result is a *pooling
    weight pattern* u, not a rewritten feature field that is then mean-pooled.
  - NOT consensus / cosine-to-centroid (a59/a62): those measure agreement with
    the DC mode (the centroid). a93 removes the centroid and reads the leading
    *non-trivial* mode — a different eigenvector entirely (Fiedler-style, the
    principal direction of agreement, not the location).
  - NOT rank / norm-salience: ||h|| never enters; the Gram uses L2-NORMALISED
    patch directions (cosine geometry), so the mode is magnitude-invariant.
  - NOT top-k / argmax / selection: u_i is a dense, signed, soft membership; no
    patch is dropped. NOT robust-trimmed / geometric-median.
  - NOT per-patch severity / coverage / threshold / fraction: no per-patch scalar
    scorer, no sigmoid fraction, no count of patches above a cutoff.
  - NOT distribution moments / quantiles / soft-rank / pairwise-Gini / dist-match:
    z_spec is a single bilinear (eigen)projection, not a moment field, an order
    statistic, a pairwise spread, or a match to per-grade reference histograms.
  - NOT PMA / set-transformer / learned-query attention: the pooling weight u is
    a PARAMETER-FREE data-driven eigenvector of the patch Gram, not produced by a
    learned seed/query/softmax scorer.
  - NOT cumulative-link / ordinal head: the readout is a single Linear -> raw
    scalar. NOT heavy-dropout/input-noise (baseline dropout only).
  - NOT 2/3-way backbone fusion, NOT a multi-mechanism ensemble: one backbone,
    one pooled vector, one head.

This is, to my knowledge, the only attempt that pools the LEADING NON-TRIVIAL
spectral mode in closed form. The closest tried idea (a83) used the SAME graph
but the OPPOSITE end of the spectrum (it converged to the DC mode); a93 deflates
exactly that mode and keeps the next one. That is a concrete, checkable
distinction, not a relabel.

================================================================================
GENERALISATION / VARIANCE ARGUMENT (why PAIRED cross-fold, not a seed=2 lottery)
================================================================================
The val<->test variance on this 214-ROI cohort is driven by a FEW bags whose
summary swings (a sharpened attention bag, or a handful of bright artefact
patches dragging the mean). a93 attacks that variance structurally:
  - Removing the DC mode makes the readout INSENSITIVE to a constant offset
    shared by the whole bag (global stain/exposure shifts that move the centroid
    but not the relative co-variation).
  - Reading the LEADING coherent mode is a known DENOISER: incoherent
    per-patch outliers project weakly onto the top eigenvector (they sit in the
    high-frequency tail), so a few artefact patches cannot dominate z_spec the
    way they can dominate a mean or a sharp attention. This is a smoothing /
    shrinkage on the bag SUMMARY, lowering summary variance across folds rather
    than fitting one seed.
  - The spectral readout adds essentially ZERO learnable parameters (only a tiny
    fixed-size head + one blend scalar): low capacity => less overfit => the
    paired Δ should hold across folds {0,1,2,3,42}, not just clear a seed=2 gate.
  - We still carry the mean term, so in the worst case (no coherent mode beyond
    the centroid) a93 degrades gracefully toward the diffuse pool rather than to
    a high-variance attention collapse.

================================================================================
FORWARD-PASS MATH (exact)
================================================================================
features f in R^{N x 1280} (one bag).
  h_i      = Dropout(ReLU(W1 f_i + b1))           in R^{hidden}     # baseline encoder
  hd_i     = h_i  detached for the GRAPH ONLY (graph is a data-driven readout
             pattern, not a path we backprop the eigensolver through; the head
             still backprops through h via z_dc and z_spec amplitudes).
  -- build cosine geometry (magnitude-invariant) --
  e_i      = hd_i / ||hd_i||                       in R^{hidden}     # unit directions
  e_bar    = mean_i e_i
  ec_i     = e_i - e_bar                            # CENTER => REMOVE the DC mode
  -- leading NON-TRIVIAL spectral mode by deterministic power iteration on the
     patch Gram G = Ec Ec^T (G is [N,N] PSD; we never form it explicitly, we use
     the equivalent feature-space iteration to stay O(N*hidden)) --
  v        = ec_(argmax ||ec_i||) (deterministic seed; falls back to ec_0)
  repeat T times (T fixed, e.g. 8):                # closed-form spectral filter
      g    = Ec^T v          in R^{hidden}         # (Ec^T Ec) v done in two matmuls
      w    = Ec  g           in R^{N}              # = G v  (up to the split below)
      v    = w / ||w||                             # normalise -> top eigenvector of G
  u        = v                                     # [N] unit leading eigenvector ||u||=1
  m_i      = sqrt(N) * u_i                          # [N] bag-size-normalised membership
             (unit eigenvector entries are O(1/sqrt(N)); the sqrt(N) rescale makes
              m_i O(1) so that the MEAN pool below is invariant to bag size / dup)
  -- SIGN FIX (determinism + grade alignment): the eigenvector sign is arbitrary;
     fix it so the mode points along the train fibrosis axis a (seed=2, train-only,
     init/inference-fixed). s = sign( (1/N) sum_i m_i * (f_i . a) ); if s==0 fall
     back to sign(mean_i m_i). m <- s * m.  Makes the readout DETERMINISTIC and
     orients "more coherent fibre" as the positive direction. --
  -- pooled representations --
  z_dc     = mean_i h_i                            in R^{hidden}     # DC / diffuse density
  z_spec   = (1/N) sum_i m_i * hc_i  where hc_i = h_i - z_dc         # in R^{hidden}
             (amplitude of the leading coherent mode in ENCODER space, pooled as a
              MEAN => bag-size-invariant; uses the non-detached h so gradients flow
              to the encoder through the projection, while m itself is a detached
              data-driven weight pattern)
  -- blend (learnable scalar gamma, NOT a gate on the scalar; it mixes two
     FEATURE vectors before the single linear head) --
  gamma    = sigmoid(gamma_logit)  in (0,1)        # learnable, init small
  z        = z_dc + gamma * z_spec                 in R^{hidden}
  y        = w_head . z + b_head                   in R                # RAW scalar logit

Permutation-invariance: e_bar, z_dc are means (symmetric); the Gram G and its top
eigenvector are permutation-EQUIVARIANT (permuting patches permutes the entries of
u consistently), and z_spec = sum_i u_i hc_i is a permutation-INVARIANT contraction.
The sign fix uses only symmetric sums. => y is permutation-invariant.
Bag-size-invariance: e_i are unit (scale-free in N); the eigenvector u is
normalised to ||u||=1; we then rescale to membership m_i = sqrt(N) u_i and pool as
a MEAN z_spec = (1/N) sum_i m_i hc_i. Duplicating the bag shrinks each u_i by
1/sqrt(2) but doubles the patch count; the sqrt(N) rescale and the 1/N mean exactly
cancel that, so z_spec is unchanged up to numerical noise; z_dc is a mean. Both
terms are bag-size-invariant. Deterministic at inference: Dropout off, fixed T
power-iteration steps with a deterministic seed and a deterministic sign fix; no
sampling, no data-dependent branching on float ties beyond the documented fallbacks.

================================================================================
ABLATION — removes EXACTLY the active ingredient
================================================================================
a94 (companion) sets gamma_fixed=0.0: z = z_dc only => EXACT diffuse mean-pool +
the SAME linear head, with the spectral mode computed but NOT used. The encoder,
head, warm-start, and dropout are byte-identical (a94 imports a93's Model and only
flips the documented gamma flag). a93 (gamma learnable, init small) vs a94
(gamma=0) isolates EXACTLY: "does adding the leading NON-TRIVIAL coherent spectral
mode to the diffuse pool help / stabilise vs the diffuse pool alone?". Nothing else
differs. (A second, stricter ablation — replace u by the all-ones/DC vector — would
make z_spec == 0 after centering, recovering the same null, confirming the lever is
specifically the NON-TRIVIAL mode.)

Kill criterion: abandon if a93 <= a94 on PAIRED Δ across most of folds {0,1,2,3,42}
(both val & test). Multi-fold paired audit, NOT seed=2 alone.

Param count (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    head       Linear(128,1)+b     =    129
    gamma_logit scalar             =      1
    --------------------------------------------
    total                          = 164,098   (< ~197K cap)
The spectral readout (centering, power iteration, sign fix, projection) adds ZERO
learnable parameters.
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
    inference-FIXED use only; injects no test information.
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
        gamma_init: float = 0.1,         # initial spectral blend weight (start near diffuse)
        gamma_fixed: Optional[float] = None,  # a94 ablation sets 0.0 (spectral OFF)
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

        # Learnable spectral blend gamma = sigmoid(gamma_logit), init small so we
        # START at the diffuse pool and only add the coherent mode if it helps.
        # The ablation (gamma_fixed=0) registers a constant buffer instead so NO
        # gamma parameter is trained and z == z_dc exactly.
        if self.gamma_fixed is None:
            gi = math.log(float(gamma_init) / (1.0 - float(gamma_init)))
            self.gamma_logit = nn.Parameter(torch.tensor(gi, dtype=torch.float32))
        else:
            self.register_buffer(
                "gamma_const", torch.tensor(self.gamma_fixed, dtype=torch.float32)
            )

        # Fibrosis axis for the deterministic SIGN FIX of the eigenvector (and for
        # warm-starting the head). Train-only (seed=2); fixed, no trainable params.
        path = Path(prototype_path) if prototype_path else _DEFAULT_AXIS_PATH
        axis = _load_fibrosis_axis(path, input_dim)
        if axis is not None:
            self.register_buffer("fib_axis", axis)
        else:
            self.fib_axis = None

        # Warm-start the head toward the fibrosis axis (train-only, seed=2).
        if warm_start and axis is not None:
            with torch.no_grad():
                w1 = self.bottleneck[0].weight.detach()       # [hidden, input]
                resp = w1 @ axis                               # [hidden] bottleneck response
                resp = resp / resp.norm().clamp(min=1e-8)
                self.head.weight.copy_(resp.view(1, -1) * 3.0)
                self.head.bias.fill_(1.5)                      # mid-grade start

    def _gamma(self) -> torch.Tensor:
        if self.gamma_fixed is None:
            return torch.sigmoid(self.gamma_logit)
        return self.gamma_const

    def _leading_mode(self, ec: torch.Tensor) -> torch.Tensor:
        """Leading eigenvector u [N] of the centered patch Gram G = ec ec^T,
        by T deterministic power-iteration steps (closed-form spectral filter).

        ec: [N, hidden] CENTERED unit-direction patch features (DC mode removed).
        We never materialise G [N,N]; G v = ec (ec^T v) stays O(N*hidden).
        Returns a unit vector u (||u||=1), the soft membership of each patch in the
        dominant coherent (low-frequency) mode. Detached: u is a data-driven
        readout PATTERN, not a path we backprop the eigensolver through.
        """
        n = ec.size(0)
        ecd = ec.detach()
        # Deterministic seed: the patch with the largest centered norm (most
        # off-centroid coherent direction); tie-break to index 0 via argmax.
        norms = ecd.norm(dim=1)                                # [N]
        seed_idx = int(torch.argmax(norms).item())
        v = ecd[seed_idx].clone()                              # [hidden]
        v = v / v.norm().clamp(min=1e-8)
        u = ecd.new_full((n,), 1.0 / math.sqrt(n))             # safe fallback [N]
        for _ in range(self.power_iters):
            # G u_vec is computed in feature space: project patches onto v, then
            # recombine. Equivalent power iteration on the [N,N] Gram G = ecd ecd^T.
            proj = ecd @ v                                     # [N]  = ecd v
            nrm = proj.norm().clamp(min=1e-8)
            u = proj / nrm                                     # [N] current mode estimate
            # Pull the corresponding feature-space direction back for next step:
            v = ecd.t() @ u                                    # [hidden] = ecd^T u
            vn = v.norm().clamp(min=1e-8)
            v = v / vn
        return u                                               # [N], ||u||=1

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        h = self.bottleneck(features)                          # [N, hidden]
        n = h.size(0)

        z_dc = h.mean(dim=0)                                   # [hidden] DIFFUSE / DC mode

        gamma = self._gamma().to(h.dtype)
        use_spec = (self.gamma_fixed != 0.0) and (n > 2)
        if use_spec:
            # Cosine geometry (magnitude-invariant) on DETACHED encoder features
            # for building the readout pattern u.
            e = F.normalize(h.detach(), dim=1, eps=1e-8)       # [N, hidden] unit dirs
            ec = e - e.mean(dim=0, keepdim=True)               # CENTER => remove DC mode
            u = self._leading_mode(ec)                         # [N] unit eigenvector ||u||=1

            # Bag-size normalisation: the unit eigenvector has O(1/sqrt(N)) entries,
            # so a raw projection sum_i u_i hc_i would scale like sqrt(N). Rescale
            # to a per-patch MEMBERSHIP weight m_i = sqrt(N) * u_i (O(1) entries) and
            # pool as a MEAN (1/N sum). Duplicating the bag shrinks each u_i by
            # 1/sqrt(2) but doubles the count; the sqrt(N) rescale + 1/N mean exactly
            # cancel that, so z_spec is invariant to bag size / duplication.
            mw = math.sqrt(float(n)) * u                       # [N] membership weights

            # Deterministic SIGN FIX: orient the mode along the fibrosis axis so
            # "more coherent fibre" is the positive direction (eigenvector sign is
            # otherwise arbitrary -> would break determinism). align uses the MEAN
            # of the (bag-size-invariant) membership * per-patch fibre proj.
            if self.fib_axis is not None:
                fib = (features @ self.fib_axis)               # [N] per-patch fibre proj
                align = torch.dot(mw, fib) / float(n)          # bag-size-invariant
            else:
                align = mw.mean()
            s = torch.sign(align)
            s = torch.where(s == 0, torch.ones_like(s), s)     # 0 -> +1 fallback
            mw = mw * s

            # Project the CENTERED encoder representation onto the coherent mode as
            # a MEAN over patches. hc uses the NON-detached h so gradients reach the
            # encoder via the amplitude, while mw stays a fixed data-driven pattern.
            hc = h - z_dc                                      # [N, hidden] centered enc
            z_spec = torch.mv(hc.t(), mw) / float(n)           # [hidden] mode amplitude (mean)
            u = mw                                             # report membership weights
            z = z_dc + gamma * z_spec
        else:
            u = None
            z = z_dc

        y = self.head(z).view(-1)                              # [1] RAW grade logit
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, u, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    power_iters=8,         # a93 main: fixed closed-form spectral filter (8 steps)
    gamma_init=0.1,        # start near the diffuse pool, add the coherent mode if it helps
    gamma_fixed=None,      # learnable gamma (a94 ablation sets 0.0 => spectral OFF)
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
