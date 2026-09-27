"""asgap_a200_n05 - ASGAP at alpha=2.00, bags capped to 5 instances.

One cell of the sparsity-vs-bag-size grid; see asgap_scale.py for the design and
for why the cap is applied at both train and eval time.
"""
from __future__ import annotations

from .asgap_scale import Model as _ASGAPScale

KWARGS = dict(input_dim=1280, num_classes=1, alpha=2.0, max_patches=5)


class Model(_ASGAPScale):
    pass
