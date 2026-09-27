"""a98 — ABLATION of a97: subspace_dim=1 (single-axis variance only).

NULL companion to a97_scatter_eigenstructure. Imports a97's `Model` UNCHANGED
and flips ONLY the documented flag `subspace_dim` from 8 (multivariate) to 1
(single 1-D variance). Everything else is byte-identical: same train-only
(seed=2) fibrosis axis as column 0 of Q, same QR-built frozen frame
(frame_seed=2), same CENTRED (mean-free) within-bag covariance, same log1p
descriptors, same warm-started 6-param-shaped linear head, same RAW-logit output.

With K=1, Q = [axis] alone, the within-bag covariance C is 1x1, and the
descriptor vector collapses to

    g = [ log1p(var), log1p(var), log1p(0), var/(var+eps), log1p(0) ]
      = [ log1p(var), log1p(var), 0,        ~1,            0        ]

so the head can read ONLY the single-axis variance (a 1-D 2nd moment along the
fibrosis direction — exactly the a64 "spread" term) plus constants. This removes
EXACTLY the active ingredient of a97 — the MULTIVARIATE scatter eigen-structure
(off-axis trace, anisotropy across directions, axis<->neighbourhood
cross-covariance) — while holding everything else identical.

a97 (K=8) vs a98 (K=1) therefore isolates: "does the multivariate covariance
SHAPE of the patch cloud beat the variance of a single 1-D projection?".

Note both a97 and a98 are MEAN-FREE (centroid removed before the covariance), so
neither can collapse to mean-pool; the comparison is purely 2nd-order
multivariate-shape vs 1-D variance.

Param count: Q frozen buffer (1280*1 = 1280 stored floats, 0 trainable).
head Linear(5 -> 1) = 6 trainable parameters (identical head shape to a97).
"""
from __future__ import annotations

# Reuse a97's Model unchanged; only KWARGS differ (subspace_dim=1). Load by file
# path so this works whether the loader imports as a package module or by path.
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_a97_scatter_eigenstructure", _here / "a97_scatter_eigenstructure.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Model = _mod.Model  # type: ignore  # noqa: F401  (re-exported)


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    subspace_dim=1,        # <-- THE ONLY CHANGE vs a97: single-axis variance only (no multivariate shape)
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    frame_seed=2,
    eps=1e-4,
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
