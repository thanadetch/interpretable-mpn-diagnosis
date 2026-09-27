"""a52 — fibrosis-coverage MIL (grading-aligned density/extent readout).

Hypothesis (Hn_fibrosis_coverage). Reticulin grading is the *overall
density / extent* of the fibre meshwork — clinically "what fraction of the
tissue is fibre-positive" — NOT the average feature nor the single most
intense patch. So aggregate by **coverage along a fibrosis direction**:

    s_i = <f_i, w>                       # per-patch fibrosis score; w in raw 1280-d
    c   = mean_i sigmoid(s_i - tau)      # soft FRACTION of fibre-positive patches
    y   = clamp(Linear(c), 0, 3)         # linear map coverage -> grade

`w` is warm-started at the train-prototype fibrosis direction
    v = normalize(mean(c_G2,c_G3) - mean(c_G0,c_G1))
(from data/prototypes_virchow2_reti_train_seed2.pt; train patients only,
no test leakage) and then learned. `init_scale` keeps the initial per-patch
scores in a non-saturated range for the sigmoid.

Grounding (results/diag/, read-only diagnostics on all 1330 Virchow2 bags):
  - feature-norm ||h|| is grade-uninformative (Spearman -0.005) -> do NOT
    weight by norm (rules out a45). It tracks tissue-vs-background, not bone.
  - coverage along v separates the gate-relevant boundaries on held-out
    patients: AUC G0|G1 0.81, G1|G2 0.84, G2|G3 0.79 (overall Spearman +0.84).
  - the fibrosis direction skips bone: per-patch Spearman(||h||, <h,v>) = -0.21.

Why this is NOT a dead-end:
  - DE22 (a25) used a prototype direction as an *attention scorer*
    (softmax -> weighted MEAN) and underfit. This uses the direction for a
    *coverage / density* readout (count the fibre-positive fraction), the
    opposite of attention concentration.
  - DE11/DE13/DE15 built coverage/presence on sigma(||h||) (norm) — which the
    diagnostic shows is grade-uninformative. This uses a *fibrosis direction*.
  - No sigma GATES the prediction (DE11-13): the sigmoid is per-patch
    pre-aggregation; the head is linear in the coverage scalar.

Ablation companion: a53_fibrosis_meanproj.py — identical except readout =
mean projection `mean_i s_i` (= mean-pool + linear, NO sigmoid). a52 vs a53
isolates the active ingredient = "does fibrosis EXTENT (coverage) beat
fibrosis AVERAGE (mean-pool)?". NOTE the diagnostic's crude single-threshold
illustration had mean-projection slightly ahead — so this is a genuine test,
not a foregone win.

Permutation- and bag-size-invariant (coverage = mean over patches);
deterministic at inference.

Kill criterion: abandon if a52 val_qwk < 0.81 at seed=2 (does not clear the
official gate region) AND a52 does not beat a53 on val (coverage adds nothing
over mean). Definition of done = multi-seed audit {0,1,2,3,42}, not seed=2 alone.

Param count (input_dim=1280): w 1280 + tau 1 + head Linear(1,1) 2 = 1283.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_direction(path: Path, input_dim: int) -> torch.Tensor:
    """Unit vector v = normalize(mean(G2,G3) - mean(G0,G1)) from train prototypes."""
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
        readout: str = "coverage",      # "coverage" (a52 main) | "mean" (a53 ablation)
        init_scale: float = 0.1,        # scales warm-started w so sigmoid starts unsaturated
        tau_init: float = 0.0,          # coverage threshold (learnable)
        warm_start: bool = True,        # init w at prototype fibrosis direction
        prototype_path: Optional[str] = None,
        random_seed: int = 2,           # used only if warm_start=False
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert readout in ("coverage", "mean"), readout
        self.readout = readout
        self.clamp_output = clamp_output

        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            v = _load_fibrosis_direction(path, input_dim)
        else:
            g = torch.Generator().manual_seed(int(random_seed))
            v = torch.randn(input_dim, generator=g)
            v = v / v.norm().clamp(min=1e-8)

        # Learnable fibrosis direction (warm-started, small-scale so per-patch
        # scores start in a non-saturated sigmoid range).
        self.w = nn.Parameter(v * float(init_scale))
        # Coverage threshold (only used by the "coverage" readout).
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        # Linear map from the scalar bag summary -> grade. Init toward y ~ 3*summary
        # so coverage in [0,1] starts spanning the grade range.
        self.head = nn.Linear(1, num_classes)
        with torch.no_grad():
            self.head.weight.fill_(3.0)
            self.head.bias.zero_()

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        s = features @ self.w  # [N] per-patch fibrosis score

        if self.readout == "coverage":
            pos = torch.sigmoid(s - self.tau)   # [N] soft fibre-positive indicator
            summary = pos.mean().view(1)        # scalar coverage (fraction)
            weight = pos
        else:  # "mean": mean projection == mean-pool(features) . w  (linear)
            summary = s.mean().view(1)
            weight = F.softmax(s, dim=0)

        y = self.head(summary)                  # [1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, weight, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    readout="coverage",     # a52 main = nonlinear extent/density
    init_scale=0.1,
    tau_init=0.0,
    warm_start=True,
    prototype_path=None,    # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
