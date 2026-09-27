"""a466 — group the patches by APPEARANCE, grade each group, let the model learn which group to trust.

THE IDEA (user's, 2026-09-11)
    A pathologist reading a reticulin ROI does not average everything in the field: fat spaces and
    bone trabeculae are not marrow and should not drive the grade ("avoid the dark bone
    trabeculae" -- advisor, 2026-06). Every aggregator in this project instead pools ALL patches
    and hopes attention learns to discount the irrelevant ones. Measured 2026-09-08, it does not:
    attention correlates NEGATIVELY with the fibrosis axis (rho -.32 to +.01).

    So make the structure explicit. Cluster the patches into k appearance groups, read a grade
    from EACH group, and let a learned weight decide how much each group counts:

        c_1..c_k = k-means centroids, TRAIN patches only, frozen           (no labels used)
        q_ig     = softmax_g(-||x_i - c_g||^2 / tau)      soft membership
        z_g      = sum_i q_ig h_i / sum_i q_ig            pooled within group g
        y_g      = head(z_g)                              this group's grade
        y        = sum_g w_g y_g,  w = softmax(logits)    w is READABLE

    `w` is one vector for the whole model, not per bag: it answers "which kind of tissue does the
    model grade from?" in three numbers. Adding a per-bag gate would add capacity, and capacity has
    lost every time it was tried here (a448/a449/a456/a461).

WHY THE GROUPS SHOULD CARRY SIGNAL (measured before building, k-means on TRAIN patches)
    virchow2 k=3: group proportions alone give Spearman(proportion, grade) = +.746 / +.471 / -.805
    titan    k=3: +.711 / -.368 / -.711
    virchow2 k=4: one group reaches +.841 while ANOTHER sits at -.050 -- i.e. ~20% of all patches
    carry no grade information at all. There is something for `w` to learn.

PREDICTION, STATED BEFORE THE RUN
    Accuracy ties ABMIL or loses slightly: pooling inside frozen groups is a constraint, and
    constraints have cost accuracy throughout this project. The point is not accuracy -- it is that
    `w` is directly readable, which no model here has offered (a373's attention slot is a uniform
    placeholder; ABMIL/ASGAP attention does not track fibrosis).

KILL CONDITIONS
    * `w` converges to uniform (~1/k) -> the model is not using the grouping; interpretability
      claim dies even if QWK is fine.
    * test QWK falls more than .02 below ABMIL -> the constraint is too expensive to be worth it.

NO LEAKAGE: centroids come from `patient_split(...)[0]` (the locked 30-patient train split) and are
computed once per (backbone, k) per process. Labels are never used to form the groups.
Permutation- and size-invariant; deterministic at eval.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_CENTROIDS: dict = {}


def _centroids(dim: int, k: int, data_root: Path) -> torch.Tensor:
    """k-means centroids over TRAIN patches only. Cached per (dim, k, root) per process."""
    key = (dim, k, str(data_root))
    if key in _CENTROIDS:
        return _CENTROIDS[key]
    import sys
    src = str(Path(__file__).resolve().parents[2])
    if src not in sys.path:
        sys.path.insert(0, src)
    from sklearn.cluster import KMeans
    from data.bag_dataset import GradingBagDatasetFull
    from train_grading_reti import BACKBONE_CONFIG, patient_split

    bb = {768: "titan", 1280: "virchow2", 1536: "uni2"}[dim]
    ds = GradingBagDatasetFull(data_root / BACKBONE_CONFIG[bb]["feature_dir"])
    tr, _, _ = patient_split(ds, seed=2)
    X = torch.cat([ds[i][0].squeeze(0) for i in tr]).numpy()
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    c = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    print(f"  a466: {bb} centroids k={k} from {X.shape[0]:,} TRAIN patches "
          f"(shares {[round(float((km.labels_ == j).mean()), 3) for j in range(k)]})")
    _CENTROIDS[key] = c
    return c


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, k: int = 3, data_root: Optional[str] = None):
        super().__init__()
        if num_classes != 1:
            raise ValueError("a466 is a scalar-regression head; use --formulation regression")
        self.k = k
        self.data_root = Path(data_root or os.environ.get("MPN_DATA_ROOT", "data"))
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dim, num_classes)      # shared across groups
        self.group_logits = nn.Parameter(torch.zeros(k))    # -> w = softmax, starts uniform
        self.log_tau = nn.Parameter(torch.zeros(()))        # membership sharpness
        self._c = None

    def group_weights(self) -> torch.Tensor:
        """The readable output: how much each appearance group counts."""
        return F.softmax(self.group_logits, dim=0)

    def memberships(self, features: torch.Tensor) -> torch.Tensor:
        if self._c is None:
            self._c = _centroids(int(features.shape[1]), self.k, self.data_root)
        c = self._c.to(features.device, features.dtype)
        d2 = torch.cdist(features, c).pow(2)                       # [N, k]
        tau = torch.exp(self.log_tau) * (d2.mean().detach() + 1e-6)
        return F.softmax(-d2 / tau, dim=1)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        q = self.memberships(features)                             # [N, k]
        h = self.bottleneck(features)                              # [N, H]
        z = (q.t() @ h) / (q.sum(0).unsqueeze(1) + 1e-6)           # [k, H] pooled per group
        y_g = self.head(z).squeeze(-1)                             # [k]
        w = self.group_weights()
        y = torch.dot(w, y_g).view(1)
        if return_attention:
            a = (q * w.unsqueeze(0)).sum(1)                        # per-patch contribution
            return y, a / (a.sum() + 1e-9), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, k=3)
