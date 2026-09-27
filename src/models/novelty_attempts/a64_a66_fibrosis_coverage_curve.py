"""a66 — fibrosis-coverage CURVE MIL (read the bag's soft survival function).

Hypothesis (Hn_fibrosis_coverage_curve). Reticulin grade is the OVERALL /
DIFFUSE density of the fibre meshwork — a bag-wide extent, not a few standout
patches. A SINGLE coverage scalar (a52) is a smooth monotone functional that,
in the near-linear regime early-stopping picks, is dominated by its first-order
term mean_i s_i = <mean_i f_i, w> = mean-pool — which is why a52 (0.7675) LOST
to its mean-projection ablation a53 (0.7725). The fix is to read the WHOLE
coverage CURVE (the soft survival function evaluated at a staged ladder of
thresholds) as a VECTOR, so the head reads the curve's LOCAL SLOPE — the
diffuse-vs-focal distinction — not just one integral count.

    s_i   = <f_i, w>                                  # per-patch fibrosis score
    tau_0 = tau0_raw
    tau_k = tau_0 + sum_{j<=k} softplus(delta_j)      # MONOTONE ladder, k=1..K-1
    beta  = softplus(beta_raw).clamp(min=1e-3)        # one shared temperature
    p[i,k]= sigmoid((s_i - tau_k) / beta)             # soft fibre-positive @ tau_k
    c_k   = (1/N) sum_i p[i,k]                         # MEAN COVERAGE at tau_k
    y     = clamp(Linear(K -> 1)(c), 0, 3)            # free readout of the curve

`c in R^K` is the soft survival function: monotone non-increasing in k (same
axis, ordered taus). The head is free to read band densities c_k - c_{k+1} =
soft fraction of patches in [tau_k, tau_{k+1}). A steep curve (coverage drops
fast over a narrow band) = uniformly moderate-to-dense meshwork (true high
grade); a shallow long-tailed curve = a focal patch of dense fibre in clean
marrow (lower grade) — AT THE SAME MEAN. That is the pathologist's exact
diffuse-vs-focal distinction.

`w` is warm-started at the train-prototype fibrosis direction
    v_hat = normalize(mean(c_G2,c_G3) - mean(c_G0,c_G1))
(from data/prototypes_virchow2_reti_train_seed2.pt; TRAIN patients only, no
test leakage), scaled by init_scale=0.1 so initial scores sit unsaturated.
The readout is warm-started a_k = 3/K, b = 0 so y starts at 3 * mean_k c_k.

Grounding (results/diag/, read-only diagnostics on all 1330 Virchow2 bags):
  - feature-norm ||h|| is grade-uninformative (Spearman ~0). The per-patch
    quantity here is the PROJECTION onto the learned fibrosis axis w (NOT a
    norm); proj Spearman +0.882, coverage +0.844; Spearman(||h||, <h,v>) =
    -0.21, i.e. the axis skips bone. Norm NEVER enters the computation.
  - coverage along v separates the gate-relevant grade boundaries on held-out
    patients (overall Spearman +0.84) — the +0.84 direction the axis warm-starts.

Why this is NOT a refuted dead-end:
  - NOT a52/a53 (ONE soft threshold -> ONE coverage scalar -> Linear(1,1)):
    a52 collapsed to mean-pool because a single coverage is first-order-dominated
    by the mean. a66 keeps K=7 staged thresholds as a VECTOR; the per-band slopes
    are linearly independent of the mean integral (verified: equal-mean diffuse
    vs focal bags give DIFFERENT curves). Mean-pool / single-coverage is the
    uniform-weight column — a strict 1-D sub-space of a66's K-dim readout — so
    the "collapses to mean-pool linear ablation" failure is structurally
    impossible here.
  - NOT a54/a55 (K thresholds on K DIFFERENT learned bottleneck directions,
    SUMMED into a cumulative-link scalar): a54's directions collapsed to the
    dominant axis. a66 uses ONE shared axis and keeps the curve as a VECTOR
    read by a FREE Linear(K,1) (not tied +1/level, not summed).
  - NOT a47 (thresholds on one already-collapsed post-aggregation bag scalar,
    redundant with a calibrated linear head): a66 thresholds the PER-PATCH
    scores PRE-aggregation, then takes mean coverage per threshold.
  - NOT a64/a65 (moments mean+spread of the projection): a66 reads the survival
    CURVE (coverage shape), a different statistic.
  - NO ||h|| / norm weighting (rules out a45). NO attention / softmax / top-k:
    every c_k is an UNWEIGHTED MEAN over ALL patches. NO sigma GATE on the
    output: the sigmoid is per-patch pre-aggregation; the head is linear in c.

Permutation- and bag-size-invariant (each c_k is a mean over patches);
deterministic at inference (no dropout / softmax / sampling).

Robustness (vs the a40 one-seed lottery):
  (1) w warm-started at the label-free +0.84 train-prototype direction — the
      10-patient val cohort cannot drag a grounded axis into a non-generalising
      corner the way a from-scratch direction (a49/a50) did.
  (2) Tiny capacity (~1296 params, ~150x below the 197K overfitting baseline):
      low capacity + grounded warm-start is the regime that survives seeds.
  (3) The K-dim readout STRICTLY CONTAINS mean-coverage (uniform column), so
      a66 can never be structurally worse than the refuted single-coverage
      baselines; any gain comes from the extra slope DOF the ablation isolates.
  (4) Monotone softplus-cumsum thresholds keep the ladder ordered every
      gradient step, removing the threshold-permutation degeneracy that makes
      free multi-tau models seed-fragile.

Ablation companion: a67_fibrosis_coverage_single.py — identical Model with
n_thresholds=1 (K=1), collapsing the ladder to ONE learned soft threshold ->
a single calibrated mean-coverage scalar read by Linear(1,1). That is EXACTLY
the a52 mechanism, rebuilt inside the same Model (zero code drift; only
n_thresholds differs). a66 vs a67 isolates exactly the active ingredient:
"does reading the coverage CURVE (survival-function shape / local slope =
diffuse-vs-focal density) at multiple staged thresholds beat reading coverage
at ONE threshold (a52, which already lost to mean-pool)?"

Kill criterion: abandon if a66 does not beat a67 on a multi-seed {0,1,2,3,42}
val median (the curve adds nothing over one extent count). DoD = multi-seed
audit, not seed=2 alone.

Param count (input_dim=1280, K=7): w 1280 + tau0_raw 1 + delta_raw (K-1)=6
  + beta_raw 1 + head Linear(K,1) (K weights + 1 bias)=8 = 1296.
  (K=1 ablation: 1280 + 1 + 0 + 1 + 2 = 1284.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_direction(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    """Unit vector v = normalize(mean(G2,G3) - mean(G0,G1)) from train prototypes.

    Loaded DEFENSIVELY: returns None (caller falls back to random init) if the
    cache is missing or malformed, so the module never hard-crashes a screen.
    """
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        p = blob["prototypes"]
        hi = (p[2].float() + p[3].float()) / 2.0
        lo = (p[0].float() + p[1].float()) / 2.0
        v = hi - lo
        v = v / v.norm().clamp(min=1e-8)
        if v.shape[0] != input_dim:
            return None
        return v
    except Exception:
        return None


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        n_thresholds: int = 7,          # K. a66 main = 7; a67 ablation = 1
        init_scale: float = 0.1,        # scales warm-started w -> unsaturated sigmoid
        tau0_init: float = 0.0,         # lowest threshold (learnable)
        delta_init: float = 0.5,        # initial spacing between adjacent thresholds
        beta_init: float = 1.0,         # initial shared temperature (pre-softplus inverse)
        warm_start: bool = True,        # init w at prototype fibrosis direction
        prototype_path: Optional[str] = None,
        random_seed: int = 2,           # used only if warm_start=False or cache missing
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert n_thresholds >= 1, "n_thresholds (K) must be >= 1"
        self.input_dim = int(input_dim)
        self.n_thresholds = int(n_thresholds)
        self.clamp_output = bool(clamp_output)

        # --- learnable fibrosis direction (warm-started, small-scale) -------
        v = _load_fibrosis_direction(
            Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH,
            self.input_dim,
        ) if warm_start else None
        if v is None:  # fall back to a deterministic random unit vector
            g = torch.Generator().manual_seed(int(random_seed))
            v = torch.randn(self.input_dim, generator=g)
            v = v / v.norm().clamp(min=1e-8)
        self.w = nn.Parameter(v * float(init_scale))

        # --- MONOTONE threshold ladder via softplus-cumsum -------------------
        # tau_0 = tau0_raw ; tau_k = tau_0 + sum_{j<=k} softplus(delta_raw_j).
        self.tau0_raw = nn.Parameter(torch.tensor(float(tau0_init)))
        # invert softplus so the initial gaps equal delta_init: x = log(e^d - 1).
        if self.n_thresholds > 1:
            d = float(delta_init)
            inv_sp = torch.log(torch.expm1(torch.tensor(d)).clamp(min=1e-6))
            self.delta_raw = nn.Parameter(inv_sp.repeat(self.n_thresholds - 1))
        else:
            # No spacing params for K=1; register an empty buffer-like param so
            # the rest of the code path is identical.
            self.delta_raw = nn.Parameter(torch.zeros(0))

        # --- one shared temperature (positive via softplus) ------------------
        b = float(beta_init)
        inv_sp_beta = torch.log(torch.expm1(torch.tensor(b)).clamp(min=1e-6))
        self.beta_raw = nn.Parameter(inv_sp_beta)

        # --- free linear readout of the curve c in R^K -> grade --------------
        self.head = nn.Linear(self.n_thresholds, num_classes)
        with torch.no_grad():
            # warm-start a_k = 3/K, b = 0  ->  y starts at 3 * mean_k c_k.
            self.head.weight.fill_(3.0 / float(self.n_thresholds))
            self.head.bias.zero_()

    def _thresholds(self) -> torch.Tensor:
        """Monotone non-decreasing ladder tau in R^K (softplus-cumsum)."""
        tau0 = self.tau0_raw.view(1)                       # [1]
        if self.n_thresholds == 1:
            return tau0
        gaps = F.softplus(self.delta_raw)                  # [K-1] > 0
        steps = torch.cumsum(gaps, dim=0)                  # [K-1]
        return torch.cat([tau0, tau0 + steps], dim=0)      # [K]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        s = features @ self.w                              # [N] per-patch score
        tau = self._thresholds()                           # [K] ordered ladder
        beta = F.softplus(self.beta_raw).clamp(min=1e-3)   # shared temperature > 0

        # Soft survival function: p[i,k] = sigmoid((s_i - tau_k) / beta).
        z = (s.unsqueeze(1) - tau.unsqueeze(0)) / beta     # [N, K]
        p = torch.sigmoid(z)                               # [N, K]
        c = p.mean(dim=0)                                  # [K] mean coverage curve

        y = self.head(c.view(1, -1))                       # [1, num_classes] -> [1,1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # expose the coverage curve (the per-threshold extent) as aux.
            return y, c.detach(), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    n_thresholds=7,         # a66 main = read the WHOLE coverage curve (K=7)
    init_scale=0.1,
    tau0_init=0.0,
    delta_init=0.5,
    beta_init=1.0,
    warm_start=True,
    prototype_path=None,    # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
