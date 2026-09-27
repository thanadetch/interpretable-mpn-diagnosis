"""psemix - Pseudo-Bag Mixup augmentation (feature-space, train-only).

Reimplements PseMix: Liu et al., "Pseudo-Bag Mixup Augmentation for Multiple
Instance Learning-Based Whole Slide Image Classification", IEEE Transactions on
Medical Imaging (TMI), 2024 (arXiv:2306.16180).

MECHANISM (faithful to the paper's pseudo-bag-level Mixup):
1. Divide THIS bag's N patches into ``k`` near-equal pseudo-bags via a random partition
   (the paper's "random" pseudo-bag division), using ``torch.randperm``.
2. Sample ONE partner bag from the train pool (ANY grade -> cross-class mixup is the
   intended behaviour). Divide it into ``k`` pseudo-bags the same way.
3. Pseudo-bag-level Mixup with SIZE ALIGNMENT: draw ``lam ~ Beta(strength, strength)``
   (sampled on CPU), set ``m = round(lam * k)`` pseudo-bags to come from the partner,
   and build the mixed bag from ``(k - m)`` of THIS bag's pseudo-bags concatenated with
   ``m`` of the partner's pseudo-bags. This replaces a contiguous "area" of the bag with
   the partner's tissue at pseudo-bag granularity (PseMix's CutMix-like assembly), rather
   than convex-blending individual patch vectors.
4. SOFT LABEL (size/area-aligned Mixup, "semantic alignment"): the label-space mix ratio
   equals the pseudo-bag-count (area) ratio:
       target = ((k - m) / k) * label + (m / k) * partner_label,   clipped to [0, 3].
   With near-equal pseudo-bags the count ratio ~= the patch-area ratio, matching PseMix's
   "size alignment + semantic alignment" coupling between mixed content and mixed label.

``strength`` = Beta concentration governing how aggressively the two bags are mixed
(small ->0: lam near 0/1, near-pure bags; large ->inf: lam near 0.5, ~50/50 mixes).
``k`` = number of pseudo-bags per bag (paper default region count; here default 4).
If ``strength <= 0`` it is a no-op; if the pool is None / has <2 bags, or either bag has
fewer patches than ``k``, it gracefully no-ops (returns the original bag + float label).

DEVIATIONS / APPROXIMATIONS FROM THE ORIGINAL (honest):
- INSTANCE-FEATURE VARIANT: we mix cached foundation-model patch FEATURES, not raw image
  tiles; the augmentation lives in feature space (the MIL features are precomputed here).
- PSEUDO-BAG DIVISION is selectable: ``division="random"`` (this module's default, kept so
  earlier runs stay reproducible) or ``division="proto"``, the paper's phenotype-stratified
  ProtoDiv. Use ``psemix_proto`` for the paper-faithful setting.
- NO "Mixup-instance partial-removal" warm-up / target-bag instance dropping schedule from
  the paper's full recipe; this is the core pseudo-bag Mixup step only, with no training
  schedule (the registry API is stateless per call).
- Partner is sampled uniformly over all bags (cross-class allowed), matching the paper's
  intent; we do not do class-balanced partner selection.

Deterministic given the global torch seed (torch RNG only; Beta sampled on CPU), permutation-
and bag-size-invariant in spirit (random partition of an unordered patch set, normalised by
count), MPS-safe (indexing + concat only, no eigh/cdist/pca), no new dependencies.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


def _partition(n: int, k: int, device) -> List[torch.Tensor]:
    """Random near-equal partition of indices [0, n) into EXACTLY k non-empty index tensors.

    Caller guarantees ``n >= k`` so every chunk has >= 1 element. We compute near-equal
    split sizes by hand (sizes differ by at most one) instead of ``torch.chunk``, which can
    return FEWER than k chunks for some n (e.g. n=5,k=4 -> 3 chunks) and would then break the
    fixed ``range(k)`` slot loop below.
    """
    perm = torch.randperm(n, device=device)
    base, rem = divmod(n, k)
    # First ``rem`` chunks get one extra element so all sizes sum to n and each is >= 1.
    sizes = [base + 1 if i < rem else base for i in range(k)]
    return list(torch.split(perm, sizes))


def _proto_partition(feats: torch.Tensor, k: int, n_pheno: int) -> List[torch.Tensor]:
    """Phenotype-stratified partition - PseMix's own ProtoDiv pseudo-bag division.

    The paper divides a bag by first grouping instances into phenotypes, then drawing each
    pseudo-bag *across* phenotypes, so every pseudo-bag is a miniature of the whole bag
    rather than an arbitrary random subset. ProtoDiv realises this cheaply:

        1. bag prototype = mean instance feature
        2. phenotype score = cosine distance of each instance to that prototype
        3. cut the score into ``n_pheno`` groups (quantile cut = equal-count bins)
        4. inside each phenotype, shuffle and deal instances round-robin to the k pseudo-bags

    Quantile cutting is used rather than the paper's equal-width option because ROI bags here
    hold ~44 instances, where equal-width bins can come out empty; the starting slot of each
    phenotype is rotated at random so no pseudo-bag systematically receives the remainders.
    """
    n = feats.shape[0]
    proto = feats.mean(dim=0, keepdim=True)
    dist = 1.0 - F.cosine_similarity(feats.float(), proto.float(), dim=1)
    order = torch.argsort(dist)

    base, rem = divmod(n, n_pheno)
    sizes = [base + 1 if i < rem else base for i in range(n_pheno)]

    slots: List[List[torch.Tensor]] = [[] for _ in range(k)]
    start = 0
    for size in sizes:
        if size == 0:
            continue
        group = order[start:start + size]
        start += size
        group = group[torch.randperm(size, device=feats.device)]
        off = int(torch.randint(k, (1,)).item())
        for j, idx in enumerate(group):
            slots[(j + off) % k].append(idx)

    empty = torch.empty(0, dtype=torch.long, device=feats.device)
    return [torch.stack(s) if s else empty for s in slots]


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.5, k: int = 4,
                 division: str = "random", n_pheno: int = 4):
        self.strength = float(strength)
        self.k = max(2, int(k))
        # "random"  = the original registry behaviour (kept so old runs stay reproducible)
        # "proto"   = the paper's phenotype-stratified ProtoDiv division; see psemix_proto.py
        self.division = str(division)
        self.n_pheno = max(2, int(n_pheno))

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        k = self.k
        # No-op guards: disabled, no usable pool, or either bag too small to split into k.
        if self.strength <= 0.0 or pool is None or len(pool) < 2 or features.shape[0] < k:
            return features, float(label)

        j = int(torch.randint(len(pool), (1,)).item())
        partner = pool[j]
        feat_b = partner[0].to(features.device, features.dtype)
        label_b = float(partner[1])
        if feat_b.shape[0] < k:
            return features, float(label)

        # Mix ratio -> number of partner pseudo-bags (size-aligned).
        lam = float(torch.distributions.Beta(self.strength, self.strength).sample())
        m = int(round(lam * k))  # number of pseudo-bags taken from the PARTNER bag
        m = max(0, min(k, m))

        if self.division == "proto":
            parts_a = _proto_partition(features, k, self.n_pheno)
            parts_b = _proto_partition(feat_b, k, self.n_pheno)
        else:
            parts_a = _partition(features.shape[0], k, features.device)
            parts_b = _partition(feat_b.shape[0], k, feat_b.device)

        # Keep (k - m) pseudo-bags of THIS bag, take m pseudo-bags of the PARTNER bag.
        # Randomly choose WHICH pseudo-bag slots come from the partner.
        slot_order = torch.randperm(k, device=features.device)
        partner_slots = set(int(s) for s in slot_order[:m].tolist())

        chunks = []
        for slot in range(k):
            if slot in partner_slots:
                chunks.append(feat_b[parts_b[slot]])
            else:
                chunks.append(features[parts_a[slot]])
        mixed = torch.cat(chunks, dim=0)

        # Size/area-aligned soft label. The paper couples the label ratio to the AREA taken
        # from each bag; with phenotype-stratified division pseudo-bags need not be equal in
        # size, so the ratio is computed from the instances actually contributed, not from k.
        n_b = sum(int(parts_b[s].numel()) for s in partner_slots)
        n_total = int(mixed.shape[0])
        frac_b = (n_b / n_total) if n_total else 0.0
        target = (1.0 - frac_b) * float(label) + frac_b * label_b
        target = float(min(3.0, max(0.0, target)))
        return mixed, target
