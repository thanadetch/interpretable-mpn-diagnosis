"""a94 — ABLATION of a93: subspace_dim=1 (single-axis variance only).

Identical pipeline to a93 (frozen fibrosis axis, centred covariance, log1p
descriptors, 6-shaped linear head) but with K=1. Then Q = [axis] alone, the
within-bag covariance C is 1x1, and the descriptor vector collapses to

    g = [ log1p(var), log1p(var), log1p(0), var/(var+eps), log1p(0) ]
      = [ log1p(var), log1p(var), 0,        ~1,            0        ]

so the head can read ONLY the single-axis variance (a 1-D 2nd moment along the
fibrosis direction — exactly the a64 "spread" term) plus constants. This removes
EXACTLY the active ingredient of a93 — the MULTIVARIATE scatter eigen-structure
(off-axis trace, anisotropy across directions, axis<->neighbourhood
cross-covariance) — while holding everything else byte-identical.

a93 (K=8) vs a94 (K=1) isolates: "does the multivariate covariance SHAPE of the
patch cloud beat the variance of a single 1-D projection?". Same warm-start, same
frozen-frame construction, same head shape, same forward arithmetic.

Param count: Q frozen buffer (0 trainable). head Linear(5,1) = 6 trainable.
"""
from __future__ import annotations

# Reuse a93's Model unchanged; only KWARGS differ (subspace_dim=1). Load by file
# path so this works whether the loader imports as a package module or by path.
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_a93_scatter_eigenstructure_pool", _here / "a93_scatter_eigenstructure_pool.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Model = _mod.Model  # type: ignore


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    subspace_dim=1,        # ABLATION: single-axis variance only (no multivariate shape)
    warm_start=True,
    prototype_path=None,
    frame_seed=2,
    eps=1e-4,
    clamp_output=False,
)
