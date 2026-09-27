"""a108 — MonoBandDensity ABLATION: free (unconstrained) 16-weight readout.

Imports a107's Model and flips EXACTLY ONE flag: monotone=False. Everything
else — the FROZEN train-only grade axis, the B=16 soft occupancy histogram, the
calibration (m0,v0), the bin centres/sigma, the normalisation, and the parameter
count (theta in R^16, rho, bias => 18 scalars) — is byte-identical (same Model
class, same KWARGS except the one flag).

What the flip changes (the ablated active ingredient = the HARD SHAPE PRIOR):
    main a107 (monotone=True):  w_b = softplus(theta_b) >= 0
                                g_b = cumsum(w_b) / sum(w)   # NON-DEC, non-neg
                                scale = softplus(rho) > 0
    abl  a108 (monotone=False): g_b = theta_b                # FREE: sign- AND
                                                              # order-UNCONSTRAINED
                                scale = rho                  # free-sign too
Both read the IDENTICAL exceedance functional
    y = bias + scale * sum_b g_b * (1 - P_{b-1})
from the IDENTICAL within-bag density histogram p (=> P its CDF). Only the
constraint on the 16 readout weights (and the sign of scale) differs.

This isolates the central claim: "does the hard monotone / non-negative shape
constraint (vs a FREE 16-weight signed linear readout of the SAME density)
prevent the val-overfit?" Prediction: a108 (free weights) recovers the
a93/a99 val-overfit pattern — higher seed=2 val, lower/tanked test, paired
delta <= 0 — because 16 free signed weights CAN fit a fold-specific anti-grade
reweighting of the density profile; a107 (monotone) holds val and test together
=> positive paired delta. Both start from the SAME functional (g_b = b/B linear
ramp at init), so only the constraint, not the init, differs.
"""
from .a107_monobanddensity import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    n_bins=16,
    monotone=False,        # ablation = FREE signed/unordered 16-weight readout
    bin_margin=0.5,
    prototype_path=None,
)
