"""a85 — Uniform-anchored DIFFUSE attention (start at mean-pool, sharpen only if needed).

Diagnostic motivation. The grading principle states that the reticulin grade is
the OVERALL / DIFFUSE fibre density across the whole ROI — "what fraction of the
marrow is fibrotic" — and explicitly NOT a handful of standout patches. Standard
gated attention (the baseline) is free to collapse onto a few high-score patches,
which is the WRONG inductive bias for a diffuse-density label and is one source of
the val<->test variance that plagues this 214-ROI cohort (a single sharpened bag
can swing the val score). So we put an explicit DIFFUSENESS PRIOR on the pooling
weights themselves.

Mechanism (the active ingredient). Parameterise the pooling weights as a convex
blend of a UNIFORM (mean-pool) anchor and the baseline gated-attention softmax:

    scores_i = attention_W( tanh(V h_i) * sigmoid(U h_i) )      # baseline gated scorer
    a_i      = softmax_i(scores_i)                              # [N] sharp attention
    u_i      = 1 / N                                            # [N] uniform anchor
    lambda   = sigmoid(lambda_logit)  in (0, 1)                 # learnable scalar
    w_i      = (1 - lambda) * u_i + lambda * a_i                # blended pooling weights
    z        = sum_i w_i * h_i                                  # pooled bag rep [hidden]
    logits   = classifier(z)                                    # RAW scalar logit

lambda is INITIALISED SMALL (lambda_init ~ 0.1, via lambda_logit = logit(0.1)
~ -2.197) so the model STARTS essentially at mean-pooling (fully diffuse) and may
only sharpen toward gated attention if the SmoothL1 objective demands it. Because
w is always a proper convex combination of two probability vectors it is itself a
valid probability vector (sum_i w_i = 1, w_i >= 0), so z is a well-formed weighted
mean — permutation- and bag-size-invariant — regardless of lambda.

Why this is a FRESH angle (not a re-run of the refuted families):
  - NOT rank / norm-salience: ||h|| is never used; scores come only from the
    learned gated scorer (which the diagnostics show carries grade signal).
  - NOT top-k / hard selection: every patch keeps a strictly positive weight
    >= (1-lambda)/N, so no patch is ever dropped — the opposite of top-k.
  - NOT plain coverage / moments / quantiles / pairwise / dist-match: there is
    no threshold, no fraction, no moment field, no pairwise term; the readout is
    a single attention-weighted mean.
  - NOT per-patch severity / consensus / PMA: aggregation is one weighted mean
    over the baseline encoder, no per-patch value head, no induced queries.
  The genuinely new lever is a LEARNABLE SCALAR diffuseness prior that anchors
  the pooling weights at uniform and lets the data decide how far (if at all) to
  sharpen — a regulariser on the *shape* of the attention distribution, not a new
  scorer. Capacity is the baseline (197,250) + 1 (lambda_logit) = 197,251.

Optional warm-start. The seed=2 train-only fibrosis axis
(data/prototypes_virchow2_reti_train_seed2.pt['axis'], 1280-d) can be added to the
gated scores as a fixed, grade-aligned tilt (axis_weight > 0) so that, IF the model
chooses to sharpen, it sharpens along the clinically meaningful direction rather
than an arbitrary one. Default axis_weight=0.0 keeps the scorer purely learned;
the axis is read train-only and adds NO test leakage and NO trainable parameters.
RAW logits are returned (the trainer rounds+clips at eval, matching the baseline);
this module never clamps.

Ablation companion: a86_gated_attention_lambda1.py — imports this Model and sets
lambda_fixed=1.0 (lambda frozen at 1, lambda_logit removed from training), which
makes w_i = a_i exactly => STANDARD gated attention. a85 vs a86 isolates EXACTLY
the active ingredient: "does an explicit diffuseness prior on the pooling weights
(start diffuse, sharpen only if needed) help / stabilise on this cohort, versus
plain gated attention?".

Permutation- and bag-size-invariant (softmax + uniform are both symmetric means);
deterministic at inference (Dropout off in eval; no sampling).
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


def _load_fibrosis_axis(path: Path, input_dim: int) -> torch.Tensor:
    """Unit fibrosis axis (1280-d) from the seed=2 TRAIN-ONLY prototype cache."""
    assert path.is_file(), (
        f"Axis cache not found: {path}\n"
        f"Run the prototype builder to generate it."
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    v = blob["axis"].float().view(-1)
    assert v.shape[0] == input_dim, f"axis dim {v.shape[0]} != input_dim {input_dim}"
    v = v / v.norm().clamp(min=1e-8)
    return v


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        lambda_init: float = 0.1,        # initial blend weight -> START at near mean-pool
        lambda_fixed: Optional[float] = None,  # a86 ablation sets 1.0 (= plain gated attn)
        axis_weight: float = 0.0,        # >0 -> add warm fibrosis-axis tilt to scores
        prototype_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert 0.0 < float(lambda_init) < 1.0, "lambda_init must be in (0,1)."
        self.lambda_fixed = None if lambda_fixed is None else float(lambda_fixed)
        if self.lambda_fixed is not None:
            assert 0.0 <= self.lambda_fixed <= 1.0, "lambda_fixed must be in [0,1]."

        # Baseline-identical encoder + gated-attention scorer (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Learnable diffuseness blend lambda = sigmoid(lambda_logit), init small.
        # When lambda_fixed is set (ablation) we register a constant buffer instead
        # so NO lambda parameter is trained and w == softmax(scores) exactly.
        if self.lambda_fixed is None:
            li = math.log(float(lambda_init) / (1.0 - float(lambda_init)))
            self.lambda_logit = nn.Parameter(torch.tensor(li, dtype=torch.float32))
        else:
            self.register_buffer(
                "lambda_const", torch.tensor(self.lambda_fixed, dtype=torch.float32)
            )

        # Optional fixed warm-axis tilt (no trainable params; train-only direction).
        self.axis_weight = float(axis_weight)
        if self.axis_weight != 0.0:
            path = Path(prototype_path) if prototype_path else _DEFAULT_AXIS_PATH
            self.register_buffer("fib_axis", _load_fibrosis_axis(path, input_dim))
        else:
            self.fib_axis = None

    def _lambda(self) -> torch.Tensor:
        if self.lambda_fixed is None:
            return torch.sigmoid(self.lambda_logit)
        return self.lambda_const

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        h = self.bottleneck(features)                       # [N, hidden]

        V = self.attention_V(h)                             # [N, hidden]
        U = self.attention_U(h)                             # [N, hidden]
        scores = self.attention_W(V * U).squeeze(-1)        # [N] gated-attention scores

        if self.fib_axis is not None:
            # Tilt scores toward the clinically meaningful fibrosis direction (fixed).
            scores = scores + self.axis_weight * (features @ self.fib_axis)

        a = F.softmax(scores, dim=0)                        # [N] sharp attention
        N = features.size(0)
        u = features.new_full((N,), 1.0 / float(N))         # [N] uniform mean-pool anchor

        lam = self._lambda().to(a.dtype)
        w = (1.0 - lam) * u + lam * a                       # [N] convex blend (sums to 1)

        z = torch.mv(h.t(), w)                              # [hidden] weighted-mean bag rep
        logits = self.classifier(z)                         # [1] RAW logit (no clamp)

        if return_attention:
            return logits, w, None
        return logits, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    lambda_init=0.1,        # a85 main: START diffuse (near mean-pool), sharpen only if needed
    lambda_fixed=None,      # learnable lambda
    axis_weight=0.0,        # purely learned scorer by default; >0 -> warm-axis tilt
    prototype_path=None,    # None -> data/prototypes_virchow2_reti_train_seed2.pt
)
