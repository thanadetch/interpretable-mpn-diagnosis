"""a50 — random-direction score soft-mean + length-norm temperature (H25 ablation, batch 24).

Philosophy bucket: norm_based_salience

Role
----
Ablation companion for a49_learned_direction_score. Identical wiring,
except the salience direction v ∈ R^128 is **frozen** at a random unit
vector drawn from N(0, I/H) with `random_direction_seed=2`.

This ablates the *learned* part of a49's direction scorer while keeping
the structural form (signed projection onto a direction → score-softmax
with √N temperature → weighted mean) identical. Combined with a45
(rank-by-||h||), the batch resolves a clean three-way comparison:

    a45 (rank by ||h||)        — frozen norm scorer (rank-softmax),    164,098 params
    a50 (score by random v)    — frozen random scorer (score-softmax), 164,098 params
    a49 (score by learned v)   — learned direction (score-softmax),    164,226 params

See `a49_learned_direction_score.py` docstring for the full hypothesis,
interpretation matrix, kill criterion, and pathology rationale.

Notes
-----
- Reuses `a49_learned_direction_score.Model` with `learn_direction=False`
  so the implementation cannot drift between main and ablation.
- 164,098 trainable params (bottleneck 163,968 + classifier 129 +
  c_raw 1). v is a buffer with no gradient.
- For an apples-to-apples DE22 cross-comparison: a26 (random-axis
  attention scorer in *raw* 1280-d Virchow2 space) reached val 0.7889 /
  test 0.9441. a50 differs by (a) operating in bottleneck space and
  (b) using length-normalised temperature on the score-softmax.
"""
from __future__ import annotations

from .a49_learned_direction_score import Model as _A49Model


class Model(_A49Model):  # noqa: D401 — thin alias for the trainer's discovery API
    """Frozen-random-direction variant of a49 (ablation)."""


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    c_raw_init=0.5413248,
    length_norm=True,
    learn_direction=False,     # ← the only flip vs a49 KWARGS
    random_direction_seed=2,
    clamp_output=True,
)

