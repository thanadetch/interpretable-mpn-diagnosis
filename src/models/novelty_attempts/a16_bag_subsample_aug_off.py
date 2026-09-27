"""a16 — bag-subsampling augmentation OFF (H9 ablation, positive control).
Identical wiring to a15 but K_max set larger than any bag in the dataset,
so subsampling never triggers and the model is bit-for-bit the
ABMIL baseline. Serves as positive control: a16 should
reproduce baseline val_qwk = 0.8182 / test_qwk = 0.9476.
Kill criterion (family-level): if a15 val_qwk < 0.82 at seed=2, abandon
H9 and add the family to section 4 DE. Same 197,250 params as baseline.
"""
from .a15_bag_subsample_aug import Model as _Model
Model = _Model
KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    K_max=10_000,
)
