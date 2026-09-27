"""asgap_a125_n20 - ASGAP at alpha=1.25, bags capped to 20 instances.

One cell of the sparsity-vs-bag-size grid; see asgap_scale.py for the design and
for why the cap is applied at both train and eval time.
"""
from __future__ import annotations

from .asgap_scale import Model as _ASGAPScale

KWARGS = dict(input_dim=1280, num_classes=1, alpha=1.25, max_patches=20)


class Model(_ASGAPScale):
    pass
