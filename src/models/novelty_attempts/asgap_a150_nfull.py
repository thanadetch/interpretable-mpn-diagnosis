"""asgap_a150_nfull - ASGAP at alpha=1.50, bags capped to no cap (the cohort's natural ~44 instances).

One cell of the sparsity-vs-bag-size grid; see asgap_scale.py for the design and
for why the cap is applied at both train and eval time.
"""
from __future__ import annotations

from .asgap_scale import Model as _ASGAPScale

KWARGS = dict(input_dim=1280, num_classes=1, alpha=1.5, max_patches=0)


class Model(_ASGAPScale):
    pass
