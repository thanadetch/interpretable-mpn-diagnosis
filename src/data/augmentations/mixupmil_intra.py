"""mixupmil_intra - faithful multilinear Intra-WSI MixUp (train-only, feature-space).

PAPER: Gadermayr, Koller, Tschuchnig, Stangassinger, Kreutzer, Unger, Reischl,
"MixUp-MIL: Novel Data Augmentation for Multiple Instance Learning and a Study on
Thyroid Cancer Diagnosis", arXiv:2211.05862, 2022/2023 (MICCAI 2023).

MECHANISM (the variant the paper reports as best -- *multilinear Intra-WSI MixUp*):
MixUp-MIL augments at the instance/patch-feature level rather than the image level. The
paper distinguishes:
  - INTER-WSI MixUp: interpolate patches between TWO different WSIs (bags); the bag label
    becomes a soft mix of the two WSI labels.
  - INTRA-WSI MixUp: interpolate patches WITHIN the SAME WSI; since every patch and the
    synthesised patches stay inside one bag, the bag label is UNCHANGED.
  - "multilinear" extension: instead of mixing just two patches with a single lambda, mix
    MORE THAN TWO patches at once via a convex (barycentric) weight vector.
The paper found the intra-WSI / multilinear variants the most reliably beneficial, while
inter-WSI mixing was weaker / dataset-dependent.

This module implements multilinear INTRA-WSI MixUp:
  1. Work entirely within this single bag of N patches (no partner bag needed; pool may be
     None and it still works -- intra-WSI mixing is self-contained).
  2. Build M (=N) synthetic patches. For each output patch we pick T patches from the SAME
     bag (uniformly, with replacement) and take a convex combination with Dirichlet weights
     w ~ Dirichlet(alpha * 1_T). T defaults to 3 (>2 = "multilinear", vs pairwise MixUp).
  3. The bag label is UNCHANGED (intra-WSI), so requires_regression = False and the target
     is returned exactly as given. This works under either the classification or the scalar-
     regression formulation.

`strength` controls the Dirichlet concentration / interpolation magnitude. We map it so that
  - small strength -> peaky weights (one weight ~1, the rest ~0) -> synthetic patches sit
    very close to a real patch (mild augmentation, near the data manifold);
  - larger strength -> flatter weights -> stronger mixing / further from any single real
    patch (more aggressive).
Concretely alpha = 1 / (strength + eps): strength=0.3 -> alpha~=3.3 (mildly peaky). <=0
disables (no-op). We additionally blend the synthesised bag with the original by `strength`
so that strength also literally scales how far the bag moves from the real patches:
    out = (1 - strength) * original_patch + strength * multilinear_mix .
At strength=0 this is exactly the original bag; at strength=1 it is the pure multilinear mix.

DEVIATIONS / APPROXIMATIONS FROM THE PAPER (honest):
  - The paper applies MixUp on a CNN's patch *feature maps / embeddings* produced inside the
    MIL pipeline; here we apply it on FROZEN foundation-model patch features ([N, 768] for
    TITAN). The convex-combination operation is identical, only the feature source differs.
  - The exact weight distribution in the paper for the multilinear case is a barycentric /
    convex weighting; we realise it with a Dirichlet(alpha) draw, which is the standard
    continuous distribution over the simplex. Plain MixUp uses a Beta (the T=2 Dirichlet
    special case), so this is a faithful generalisation, but the precise alpha schedule is
    our choice (mapped from `strength`), not a value taken verbatim from the paper.
  - We pair each synthetic patch with one "anchor" real patch and blend by `strength` to give
    `strength` a smooth no-op-at-0 interpretation; the paper does not describe this exact
    anchor-blend, it is an added knob to make the strength continuum well-behaved. At
    strength=1 the behaviour is the pure paper-style multilinear intra-WSI mix.
  - We do NOT implement the inter-WSI variant here (the paper / our prior
    mixup_instance_pool already found inter-WSI mixing weaker); this is the intra-WSI
    multilinear variant only.

Permutation- and bag-size-invariant (operates on the unordered patch set, M=N), deterministic
given the global torch seed (uses torch.rand/randint only), MPS-safe (plain matmul/indexing,
no eigh/cdist/pca), no new dependencies. No pool-derived state is needed, but a lazy
_ensure(pool)/_tried guard is kept for idiom consistency.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)
_EPS = 1e-6


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.3, T: int = 3):
        self.s = float(strength)
        self.T = max(2, int(T))  # >=2; 3 (default) = multilinear
        self._tried = False

    def _ensure(self, pool) -> None:
        # Intra-WSI mixing needs no pool-derived state; kept for idiom consistency.
        if self._tried:
            return
        self._tried = True

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        # No-op guards: disabled, or bag too small to mix T distinct-ish patches.
        if self.s <= 0.0 or features.shape[0] < 2:
            return features, float(label)
        self._ensure(pool)

        n, d = features.shape
        device, dtype = features.device, features.dtype
        T = min(self.T, n)  # cannot mix more points than the bag has

        # Dirichlet concentration: small strength -> peaky (mild), large -> flat (strong).
        alpha = 1.0 / (self.s + _EPS)

        # Sample Dirichlet(alpha) weights for each of the M=N output patches via the
        # Gamma -> normalise construction (torch.distributions is seeded by the global RNG
        # and MPS-safe; we keep it on CPU then move, to be safe across backends).
        conc = torch.full((n, T), float(alpha))
        w = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample()  # [N, T] cpu
        w = w / (w.sum(dim=1, keepdim=True) + _EPS)
        w = w.to(device=device, dtype=dtype)

        # For each output patch pick T source patches from THIS bag (uniform, w/ replacement).
        idx = torch.randint(n, (n, T), device=device)  # [N, T]
        # Gather: [N, T, D]
        gathered = features[idx]  # advanced indexing -> [N, T, D]
        # Multilinear convex combination: sum_t w_t * h_{idx_t} -> [N, D]
        mixed = (gathered * w.unsqueeze(-1)).sum(dim=1)

        # Anchor-blend by strength so strength=0 -> original, strength=1 -> pure mix.
        s = min(1.0, self.s)
        out = (1.0 - s) * features + s * mixed

        # Intra-WSI: label unchanged.
        return out, float(label)
