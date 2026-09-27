"""Pluggable, registry-style feature-space augmentations for MIL training.

Mirrors the novelty-aggregator pattern (src/models/novelty_attempts/): one file per
augmentation under ``src/data/augmentations/<name>.py`` exposing:

    KWARGS = dict(...)                 # default hyper-parameters
    class Augmentation(BaseAugmentation):
        requires_regression = bool     # if True, only applied under regression formulation
        def __call__(self, features, label, pool=None) -> Tuple[Tensor, float]:
            '''features: [N, D] bag already on the model's device.
               label:    float grade of this bag.
               pool:     the train Subset (indexable -> (feat, label, ...)) for cross-bag
                         sampling; may be None.
               Returns (possibly-augmented features, possibly-mixed regression target).'''

Select at train time with:  ``--augmentation <name> [--aug_strength <float>]``
Default (no ``--augmentation``) = OFF = behaviour byte-identical to baseline.

Add a NEW augmentation = drop a new ``<name>.py`` file here. No edits to the trainer
needed (exactly like adding a novelty aggregator).
"""
from __future__ import annotations
import importlib
from typing import Optional


class BaseAugmentation:
    """Optional base class. An augmentation maps (features, label) -> (features, target)."""

    requires_regression: bool = False

    def __call__(self, features, label, pool=None):  # pragma: no cover - interface
        raise NotImplementedError


def load_augmentation(name: Optional[str], strength: Optional[float] = None):
    """Load ``src/data/augmentations/<name>.py`` and return an ``Augmentation`` instance.

    Args:
        name: module name (without .py). ``None`` / ``""`` / ``"none"`` / ``"off"`` -> returns
              ``None`` (augmentation OFF, baseline behaviour).
        strength: if not None, overrides the module's ``KWARGS["strength"]``.

    Returns:
        An ``Augmentation`` instance, or ``None`` when augmentation is disabled.
    """
    if name is None or name.lower() in ("", "none", "off"):
        return None
    mod = importlib.import_module(f"data.augmentations.{name}")
    kwargs = dict(getattr(mod, "KWARGS", {}))
    if strength is not None:
        kwargs["strength"] = strength
    return mod.Augmentation(**kwargs)
