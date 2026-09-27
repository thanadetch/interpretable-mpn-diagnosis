"""a64 — fibrosis MEAN + SPREAD (second-order dispersion) readout.

LENS: second-order / dispersion. Read the MEAN and the SPREAD (standard
deviation) of the per-patch fibrosis projection along a warm-started,
fine-tunable direction. NO ||h|| term anywhere. NO coverage threshold. NO
per-patch nonlinear severity MLP (the recurring collapse-to-mean trap).

Forward-pass math (single bag, features f in R^{N x D}):

    w           = warm-started, learnable direction in raw 1280-d space,
                  initialised at v_hat = normalize(mean(c_G2,c_G3)
                                                   - mean(c_G0,c_G1))
                  from data/prototypes_virchow2_reti_train_seed2.pt
                  (train patients only, no test leakage).
    s_i         = <f_i, w>                              # [N] 1-D projection
    m1          = mean_i s_i                            # 1st moment (mean density)
    var         = mean_i (s_i - m1)^2                   # 2nd central moment
    sd          = sqrt(var + eps)                       # spread (std), eps for grad-safety
    summary     = [ m1 , sd ]                           # 2-vector
    y           = clamp( Linear(2 -> 1)([m1, sd]) , 0, 3 )

GRADING RATIONALE (diffuse density, advisor's prior).
    Grade = OVERALL / DIFFUSE density of the reticulin meshwork. Two
    statistics, both bag-wide and both along the *one* axis the pathology
    varies on (held-out coverage Spearman +0.84, proj_hilo Spearman +0.882):
      - m1  = average density along v_hat. This IS the mean-pool signal
              (mean_i <f_i,w> = <mean_i f_i, w>), the strongest single
              statistic in results/diag/direction_separability.md.
      - sd  = how UNIFORM that density is. DIFFUSE high density (true G2/G3:
              dense fibre everywhere) has high m1 AND low sd. FOCAL/patchy
              high density (a few hot tiles in clean marrow — boundary G1 or
              a G0 with a couple of positive tiles) has the SAME m1 but HIGH
              sd. The learned head can therefore subtract a spread penalty:
              y ~ a*m1 - b*sd + c, encoding "diffuse, not focal" directly as
              a statistic rather than as attention concentration. This is
              exactly the advisor's diffuseness prior.

WHY THIS CANNOT COLLAPSE TO MEAN-POOL (the a56/a57, a62 failure mode).
    The within-bag variance of a FIXED-axis 1-D projection,
    var = mean_i (s_i - m1)^2 = (mean_i s_i^2) - (mean_i s_i)^2,
    depends on the SECOND raw moment mean_i s_i^2, which is NOT a linear
    functional of the mean-pooled feature mean_i f_i and is NOT recoverable
    from m1. The nonlinearity (centering + squaring) lives at the BAG level,
    AFTER the per-patch map — the one place a nonlinearity cannot be absorbed
    into a per-patch linear projection (Jensen gap). Hence the recurring
    "nonlinear per-patch design collapses to its mean-pool linear ablation"
    (a56 0.784 < a57 0.799) is structurally impossible here: the active
    ingredient sd is by construction inexpressible by the ablation a65.

DISTINCT FROM EVERY TRIED MODULE.
    - a53 / mean-projection: that IS m1 alone (this module's ablation a65).
      a64 adds the 2nd central moment, an orthogonal statistic never tried.
    - a52 / coverage: a single soft-threshold COUNT mean_i sigma(s_i - tau).
      Coverage compresses the distribution to one fraction; spread reads its
      width. They are different functionals (a52 was <= mean-projection on a
      single threshold; this is not a threshold at all).
    - a56-a63 / per-patch nonlinear severity pooling: those put the
      nonlinearity in a per-patch MLP and collapse to mean(phi). Here the
      per-patch map is a FROZEN-warm-started LINEAR projection; the only
      nonlinearity is the bag-level variance.
    - a62 / consensus-weighted mean: contrasts each patch to the bag CENTROID
      then re-weights -> still a 1st-order weighted mean that collapses to
      the mean. a64 reads the spread itself as an output, never re-weights.
    - DE07 (legacy mean+std on UNI2, raw features, no learned axis): that was
      mean+std of the RAW 1280-d feature (a high-dim std vector, norm-flavoured,
      marginal). a64 takes the std of the 1-D projection along the GROUNDED
      +0.84 fibrosis axis — a single, grade-aligned scalar, not a raw-feature
      std vector.

SEED-ROBUSTNESS ARGUMENT.
    1. Warm start: w begins at the +0.84 held-out coverage direction, so the
       optimiser starts grade-aligned and only refines; it cannot be driven
       into a non-generalising corner by the 214-ROI val cohort (the a40
       single-seed-lottery root cause).
    2. Tiny capacity: w (1280) + head Linear(2->1) (3) = 1283 trainable
       params (<< the ~197K that overfit this cohort, and < the baseline).
       No bottleneck, no MLP, no per-seed-fragile high-capacity block.
    3. Population statistic: var/std is a smooth, low-variance bag summary
       (mean over N patches); it does not chase individual patches the way
       top-K / attention concentration does. Both m1 and sd are computed the
       same way for every bag, so the readout generalises by construction.
    4. Honest falsifiability: a64 vs a65 is decisive. If spread adds nothing,
       a64 ~ a65 ~ mean-pool and we record that the dispersion term is inert
       on Virchow2 — no one-seed lottery is claimed.

Permutation-invariant (mean / variance are symmetric in i). Bag-size-invariant
(both are means over patches; var uses the biased 1/N estimator so it is a pure
mean of squared deviations, well-defined for any N>=1; for N==1 var=0 -> sd=0,
i.e. a single-patch bag reads as zero-spread, which is correct). Deterministic
at inference. Frozen features only, no ||h|| weighting.

Param count (input_dim=1280): w 1280 + head Linear(2,1) weight 2 + bias 1 = 1283.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_direction(path: Path, input_dim: int) -> torch.Tensor:
    """Unit vector v = normalize(mean(G2,G3) - mean(G0,G1)) from train prototypes.

    Train patients only (seed=2 split) -> no test leakage. This is the +0.84
    held-out coverage direction (results/diag/direction_separability.md).
    """
    assert path.is_file(), (
        f"Prototype cache not found: {path}\n"
        f"Run `python scripts/compute_grade_prototypes.py` to generate it."
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    p = blob["prototypes"]
    hi = (p[2].float() + p[3].float()) / 2.0
    lo = (p[0].float() + p[1].float()) / 2.0
    v = hi - lo
    v = v / v.norm().clamp(min=1e-8)
    assert v.shape[0] == input_dim, f"direction dim {v.shape[0]} != input_dim {input_dim}"
    return v


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        use_spread: bool = True,        # a64 main = mean + spread; a65 ablation = mean only
        init_scale: float = 1.0,        # scales warm-started w (1.0 keeps prototype-scale s_i)
        warm_start: bool = True,        # init w at prototype fibrosis direction (vs random)
        prototype_path: Optional[str] = None,
        random_seed: int = 2,           # used only if warm_start=False
        var_eps: float = 1e-6,          # numerical floor inside sqrt for grad safety
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.use_spread = bool(use_spread)
        self.var_eps = float(var_eps)
        self.clamp_output = bool(clamp_output)

        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            v = _load_fibrosis_direction(path, input_dim)
        else:
            g = torch.Generator().manual_seed(int(random_seed))
            v = torch.randn(input_dim, generator=g)
            v = v / v.norm().clamp(min=1e-8)

        # Learnable, warm-started fibrosis direction in RAW 1280-d input space.
        self.w = nn.Parameter(v * float(init_scale))

        # Readout dim: [m1] (mean only) or [m1, sd] (mean + spread).
        in_feats = 2 if self.use_spread else 1
        self.head = nn.Linear(in_feats, num_classes)
        with torch.no_grad():
            # Warm-start the head so that at init y ~ m1-driven and in the grade
            # range. With s_i at prototype scale (G0 mean ~ -9, G3 mean ~ +9),
            # a small positive slope on m1 plus a +1.5 bias puts the initial
            # output near the middle of [0,3] and monotone in density.
            self.head.weight.zero_()
            self.head.weight[0, 0] = 0.15          # slope on m1 (mean density)
            if self.use_spread:
                self.head.weight[0, 1] = -0.05     # spread PENALTY: focal -> lower grade
            self.head.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] -- ONE bag (trainer contract, batch_size=1).
        s = features @ self.w                          # [N] per-patch fibrosis projection
        m1 = s.mean()                                  # scalar mean density

        if self.use_spread:
            # Biased (1/N) within-bag variance of the projection. This is the
            # second central moment; NOT a linear functional of mean_i f_i.
            var = (s - m1).pow(2).mean()               # = mean_i s_i^2 - m1^2
            sd = torch.sqrt(var + self.var_eps)        # spread (std), grad-safe
            summary = torch.stack([m1, sd]).view(1, -1)  # [1, 2]
        else:
            summary = m1.view(1, 1)                     # [1, 1] -- mean-only ablation

        y = self.head(summary)                          # [1, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # Expose the per-patch projection as a (read-only) saliency map.
            return y, s.detach(), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    use_spread=True,        # a64 main = second-order mean + spread readout
    init_scale=1.0,
    warm_start=True,
    prototype_path=None,    # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    var_eps=1e-6,
    clamp_output=True,
)
