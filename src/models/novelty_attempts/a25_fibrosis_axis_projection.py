"""a25 — fibrosis-axis projection MIL (H3_projection_onto_fibrosis_axis, batch 13 main).

Hypothesis (H3_projection_onto_fibrosis_axis): batches 1-12 confirmed
that every per-patch *augmentation* of the gated-attention softmax-mean
overfits and lands val_qwk in [0.77, 0.80] (DE11-DE21). The diagnostic
across DE11-15 was explicit: "Future batches should consider REPLACING
the softmax-mean rather than augmenting it." H3 does exactly that.

Replace the learned 2-layer gated-attention (V * U -> W) with a
*frozen* per-patch fibrosis score `s_i = <f_i, axis>`, where
    axis = (c_G3 - c_G0) / ||c_G3 - c_G0||
and `c_g` is the mean Virchow2 patch feature over all TRAIN-PATIENT
patches with grade `g` (computed offline by
`scripts/compute_grade_prototypes.py`; cached to
`data/prototypes_virchow2_reti_train_seed{seed}.pt`). The axis is a
fixed `register_buffer` -> 0 trainable params for the attention scorer.

Pathology rationale: reticulin fibrosis is an *ordinal density* signal.
G3 patches differ from G0 patches along a population-level direction
in Virchow2 feature space (the prototype-difference vector). Using
that direction as the attention scorer makes the aggregator
*explicitly* upweight fibre-positive patches, regardless of bag size.

Mechanism (exact):
    h_i    = bottleneck(features)              # [N, 128]    (same as baseline)
    s_i    = <features_i, axis>                # [N], frozen direction in raw 1280-d space
    attn_i = softmax(s_i * tau, dim=0)         # [N]; tau learnable
    bag    = sum_i attn_i * h_i                # [128]
    y      = clamp(Linear(bag), 0, 3)

This is structurally simpler than ABMIL (no V, U, W gated
attention -> -33k params) and uses zero training-time information
beyond what `patient_split` already exposes (train labels were used
offline to compute the axis; at training time the module never reads
labels). Architecturally orthogonal to every aXX so far.

Ablation companion: a26_random_axis_projection.py - identical
architecture but `axis = unit-norm random vector` (fixed seed=2,
frozen). Tests cleanly whether the *prototype direction* carries the
signal or whether the soft-max-pool-over-scalar-score itself is the
active ingredient.

Kill criterion: abandon H3 if BOTH a25 and a26 val_qwk < 0.78 at
seed=2 (i.e. both projections fail to beat the a16 reproducible
baseline 0.7811 by any meaningful margin).

Param count (with hidden_dim=128, input_dim=1280):
    bottleneck Linear(1280, 128) + bias = 163,968
    classifier Linear(128, 1)   + bias  =     129
    tau (scalar)                        =       1
    axis buffer (frozen, not counted)   =       0
    --------------------------------------------
    total trainable                     = 164,098
    (vs ABMIL 197,250 - i.e. fewer trainable params)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default location of the cached prototype file (produced offline by
# scripts/compute_grade_prototypes.py). Resolved at module-load time
# relative to the repo root (this file lives at src/models/novelty_attempts/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = (
    _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"
)


def _load_prototype_axis(path: Path) -> torch.Tensor:
    """Load the frozen `(c_G3 - c_G0)` unit-vector from disk."""
    assert path.is_file(), (
        f"Prototype cache not found: {path}\n"
        f"Run `python scripts/compute_grade_prototypes.py` to generate it."
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    axis = blob["axis"].float()
    assert axis.dim() == 1, f"axis must be 1D, got shape {tuple(axis.shape)}"
    # Renormalise defensively (in case the cache was saved unnormalised).
    axis = axis / axis.norm().clamp(min=1e-8)
    return axis


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        clamp_output: bool = True,
        tau_init: float = 1.0,
        prototype_path: Optional[str] = None,
        random_axis: bool = False,
        random_seed: int = 2,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output

        # Same bottleneck shape as ABMIL.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

        # Build the frozen attention-scorer direction.
        if random_axis:
            # Reproducible random unit vector for the ablation companion.
            g = torch.Generator().manual_seed(int(random_seed))
            v = torch.randn(input_dim, generator=g)
            axis = v / v.norm().clamp(min=1e-8)
        else:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            axis = _load_prototype_axis(path)
            assert axis.shape[0] == input_dim, (
                f"axis dim {axis.shape[0]} != input_dim {input_dim}"
            )

        # Frozen direction in raw 1280-d feature space.
        self.register_buffer("axis", axis)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag, batch_size=1 trainer contract).
        h = self.bottleneck(features)                        # [N, hidden]
        s = features @ self.axis                             # [N], scalar projection
        attention = F.softmax(s * self.tau, dim=0)           # [N]
        bag = torch.mv(h.t(), attention)                     # [hidden]
        y = self.classifier(bag.unsqueeze(0)).squeeze(0)     # [1]

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attention, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau_init=1.0,
    prototype_path=None,    # None -> use _DEFAULT_PROTOTYPE_PATH (seed=2 train split)
    random_axis=False,      # False = prototype axis (H3 main); True = random ablation
    random_seed=2,
)

