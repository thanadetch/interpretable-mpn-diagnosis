"""a91 — Pure VARIANCE-REDUCTION: heavily regularised diffuse grading model.

Angle (a ROBUSTNESS / generalisation target, NOT a seed=2 val target). This
session's documented finding is that 86 aggregators + 3 backbones + fusion ALL
tie the baseline on PAIRED cross-fold Δ≈0: the bottleneck is VARIANCE /
generalisation on a tiny cohort, NOT signal. Even a84 cleared both seed=2 gates
but was a seed=2 LOTTERY (lost 4/5 folds paired). So the win bar here is PAIRED
cross-fold robustness: shrink the val<->test gap, not maximise seed=2 val.

Mechanism (deliberately minimal — the active ingredient is REGULARISATION, not
a new pooling operator). Reticulin grade = the OVERALL / DIFFUSE reticulin
density of the marrow, so we DIFFUSE-POOL (plain mean over patches) — every
patch enters equally, NO attention, NO selection, NO per-patch scorer, NO
norm-salience (||h|| is grade-uninformative). The ONE thing a91 changes versus
the plain diffuse mean-pool baseline (= a92) is the amount of regularisation:

    if training:  f <- f + eps,  eps ~ N(0, (noise_frac * scale)^2)   # TRAIN ONLY
    h_i = Dropout(p=0.7)(ReLU(Linear(1280->128) f_i))                  # heavier dropout
    z   = mean_i h_i                                                   # DIFFUSE pool
    y   = Linear(128 -> 1)(z)                                          # raw grade logit

Hypothesis: the val<->test collapse is OVERFITTING the tiny cohort. Two cheap,
well-understood regularisers that DIRECTLY shrink generalisation variance:
  - Train-time Gaussian INPUT noise (a Tikhonov-style / jitter regulariser).
    It perturbs the frozen Virchow2 features each step so the head cannot lock
    onto fold-specific feature idiosyncrasies. std is data-driven per bag:
    noise_frac * (RMS feature scale of THIS bag), so it tracks the real
    Virchow2 feature scale (~6) without a hand-tuned absolute number and is
    bag-size invariant (RMS is a mean over all entries). Computed under
    torch.no_grad on the input only — pure data augmentation, not a parameter.
  - Heavier Dropout p=0.7 (vs baseline 0.5) on the bottleneck activations.

The hypothesis is explicitly that STRONGER regularisation REDUCES the val<->test
gap and yields a SMALL but ROBUST paired gain (lower variance, not higher mean).

Determinism at inference is CRITICAL (this is a regularisation idea, not a
test-time-augmentation idea): both the input noise and the dropout are gated on
self.training, so under model.eval() the forward is EXACTLY

    h_i = ReLU(Linear(1280->128) f_i);  z = mean_i h_i;  y = Linear(128->1)(z)

with NO randomness — two eval passes are bit-identical. No MC-dropout, no
test-time jitter, no sampling.

Permutation-invariance: the only cross-patch op is the mean over patches (and
the per-patch encoder is applied row-wise), so any permutation of rows leaves z
(hence y) unchanged. Bag-size-invariance: z = mean normalises by N; the noise
std uses an RMS (also a mean over entries) so it does not grow with N.

Warm-start (a83 idiom, pure INITIALISATION, train-only, seed=2): the head's
weight is initialised along the train-only fibrosis axis projected through the
bottleneck input weights, so the readout points at the diffuse fibrosis
direction from step 0. Injects NO test information; identical between a91/a92 so
it does not confound the ablation.

Why this is FRESH vs the refuted list:
  - NOT rank/norm-salience, NOT top-k, NOT per-patch severity, NOT consensus,
    NOT PMA/learned-query, NOT graph-smooth, NOT plain coverage/moments/
    quantiles/pairwise/dist-match, NOT L2-norm, NOT ensemble: there is NO new
    pooling operator at all. Pooling is the plain diffuse mean (the documented
    honest null). The ONLY active ingredient is the REGULARISATION budget
    (train-time input noise + heavier dropout), which targets generalisation
    VARIANCE directly — the actual documented bottleneck — rather than adding
    yet another signal-extraction mechanism that ties on signal.

Ablation companion: a92 imports this Model and flips noise_frac=0.0 + dropout
0.5 => the STANDARD diffuse mean-pool baseline. a91 vs a92 isolates EXACTLY
"does heavier regularisation reduce the val<->test gap / improve paired
generalisation?". Encoder/head/warm-start are byte-identical; only the
regularisation budget differs.

Param count (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    head       Linear(128,1)+b     =    129
    -------------------------------------------
    total                          = 164,097   (< 197,250 baseline)
The Gaussian input noise and dropout add ZERO learnable parameters.
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

    Returns None if the cache is absent so the model still constructs (head
    then uses its default init). Uses ONLY the train-derived axis — no test
    information, no runtime features.
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
        dropout: float = 0.7,            # MAIN a91 = heavier dropout (baseline 0.5 = a92)
        noise_frac: float = 0.1,         # TRAIN-ONLY Gaussian input noise std = frac * RMS scale
        warm_start: bool = True,         # init head along train fibrosis axis (a83 idiom)
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,      # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert noise_frac >= 0.0, "noise_frac must be >= 0."
        self.noise_frac = float(noise_frac)
        self.clamp_output = clamp_output

        # Baseline-identical diffuse encoder (this is where the capacity lives).
        # Dropout p is the regularisation knob the ablation flips.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Linear readout from the diffuse-pooled representation.
        self.head = nn.Linear(hidden_dim, num_classes)

        # Warm-start the head toward the fibrosis axis (train-only, seed=2).
        # W1 @ axis gives the bottleneck-space response to the fibrosis
        # direction; pointing the head at that response means y starts as a
        # linear readout of the diffuse fibrosis density. Pure initialisation.
        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            axis = _load_fibrosis_axis(path, input_dim)
            if axis is not None:
                with torch.no_grad():
                    w1 = self.bottleneck[0].weight.detach()        # [hidden, input]
                    resp = w1 @ axis                               # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    self.head.weight.copy_(resp.view(1, -1) * 3.0)
                    self.head.bias.fill_(1.5)                      # mid-grade start

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).

        # --- TRAIN-ONLY Gaussian input noise (regulariser; OFF at eval) -------
        # Gated on self.training so model.eval() is fully deterministic. The std
        # is data-driven (noise_frac * RMS feature scale of THIS bag) so it
        # matches the real Virchow2 feature scale and is bag-size invariant.
        # Computed under no_grad on the input only => pure data augmentation.
        x = features
        if self.training and self.noise_frac > 0.0:
            with torch.no_grad():
                scale = features.detach().pow(2).mean().clamp(min=1e-12).sqrt()  # RMS scalar
                noise = torch.randn_like(features) * (self.noise_frac * scale)
            x = features + noise
        # ----------------------------------------------------------------------

        h = self.bottleneck(x)                       # [N, hidden] (Dropout off at eval)
        z = h.mean(dim=0)                            # [hidden] DIFFUSE pool
        y = self.head(z).view(-1)                    # [1] RAW grade logit
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.7,           # a91 main = heavier dropout
    noise_frac=0.1,        # a91 main = train-time input noise ON
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
