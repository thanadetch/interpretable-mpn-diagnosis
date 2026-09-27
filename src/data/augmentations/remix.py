"""remix - ReMix latent bag augmentation (reduce + mix-the-bag), train-only, feature-space.

Faithful re-implementation of:
    Yang, J., Chen, H., Zhao, Y., Yang, F., Zhang, Y., He, L., Yao, J.
    "ReMix: A General and Efficient Framework for Multiple Instance Learning based
    Whole Slide Image Classification." MICCAI 2022. arXiv:2207.01805.

MECHANISM (two stages, exactly as in the paper):

  (1) REDUCE. Each bag of N instance features is summarised by C "instance prototypes"
      = the C centroids of a hand-rolled k-means run over the bag's patches (few Lloyd
      iterations, computed on CPU then moved back to the model device). This is ReMix's
      "reduce" step: the bag is represented by its prototypes instead of all N patches.
      If N < C the patches themselves are used as prototypes (no reduction possible).

  (2) MIX-THE-BAG. With probability ``strength`` one of the four latent augmentations the
      paper defines is chosen uniformly at random and applied to the reduced prototypes.
      All four draw from a per-grade prototype BANK (centroids of every train bag of that
      grade, cached once) so they are class-identity preserving -> the bag's grade label is
      unchanged (requires_regression = False):
        (a) append      - concatenate a few same-grade bank prototypes to this bag's set.
        (b) replace     - swap a random subset of this bag's prototypes with same-grade
                          bank prototypes.
        (c) interpolate - move prototypes a fraction lam ~ U(0, strength) toward a randomly
                          paired same-grade bank prototype:  p <- p + lam * (q - p).
        (d) covary      - add zero-mean Gaussian noise scaled by the PER-GRADE prototype
                          standard deviation (the paper's "semantic variation" direction):
                          p <- p + (eps * strength) * sigma_grade,  eps ~ N(0, 1).

  ``strength`` is BOTH the per-bag probability of applying a mix op AND the intensity of the
  interpolate / covary ops. ``C`` (2nd arg, default 8) is the number of prototypes per bag.
  ``strength <= 0`` -> no-op (returns the raw features, unchanged label). ``pool is None`` or a
  bag too small to reduce -> graceful no-op.

DEVIATIONS / APPROXIMATIONS FROM THE ORIGINAL (honest):
  - PROTOTYPE SLOT ALIGNMENT is selectable. ``align="none"`` (this module's default, kept so
    earlier runs stay reproducible) uses a flat same-grade bank and draws partners at random.
    ``align="global"`` reproduces the paper's matched-slot behaviour: all training patches are
    clustered once, prototype k of every bag is that bag's mean over global cluster k, and the
    ops then pair slot k with slot k (covary also gets a per-slot sigma). Use
    ``remix_slot_<op>`` for the paper-faithful setting.
  - The k-means is a short fixed-iteration Lloyd's algorithm (no k-means++ seeding; init = first C
    distinct-ish patches via a random permutation), run on CPU for MPS safety. Empty clusters keep
    their previous centroid. This is an approximation of a fully-converged clustering.
  - With ``align="none"`` the ``sigma_grade`` for op (d) is a single per-grade vector; with
    ``align="global"`` it is estimated per prototype slot, as the paper does.
  - Bank capped at ~20000 prototype rows / grade for memory.

MPS-safe (k-means math on CPU, only matmul/cdist-free distance via expansion-free chunks; here we
use plain pairwise squared-distance with broadcasting on CPU), deterministic given the global torch
seed (torch.rand/randn/randint/randperm only), no new dependencies.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


def _kmeans(x: torch.Tensor, c: int, iters: int = 8) -> torch.Tensor:
    """Hand-rolled Lloyd k-means on CPU. x: [N, D] (cpu float). Returns [c, D] centroids.

    Pure torch, MPS-safe (caller passes CPU tensors). Empty clusters keep their centroid.
    """
    n = x.shape[0]
    if n <= c:
        return x
    # init: c patches chosen by a (seeded) random permutation
    init = torch.randperm(n)[:c]
    centroids = x[init].clone()  # [c, D]
    for _ in range(iters):
        # squared euclidean dist [N, c] via |a|^2 - 2 a.b + |b|^2 (no cdist, MPS-safe on CPU)
        x2 = (x * x).sum(dim=1, keepdim=True)            # [N, 1]
        c2 = (centroids * centroids).sum(dim=1)          # [c]
        d = x2 - 2.0 * (x @ centroids.t()) + c2.unsqueeze(0)  # [N, c]
        assign = d.argmin(dim=1)                          # [N]
        new_centroids = centroids.clone()
        for k in range(c):
            mask = assign == k
            if int(mask.sum()) > 0:
                new_centroids[k] = x[mask].mean(dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def _assign(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid assignment for [N, D] against [C, D]. CPU, MPS-safe."""
    x2 = (x * x).sum(dim=1, keepdim=True)
    c2 = (centroids * centroids).sum(dim=1)
    return (x2 - 2.0 * (x @ centroids.t()) + c2.unsqueeze(0)).argmin(dim=1)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, C: int = 8, max_bank: int = 20000,
                 op: Optional[int] = None, align: str = "none",
                 max_global_patches: int = 40000):
        self.s = float(strength)
        self.C = int(C)
        self.max_bank = int(max_bank)
        # op = None keeps the original behaviour (one of the four ops drawn per bag);
        # 0..3 pins a single op for the whole run, which is how the paper's own
        # `--mode {append,replace,interpolate,cov}` flag is used. See remix_<op>.py.
        self.op = None if op is None else int(op)
        # align = "none"   : bank is a flat same-grade pool, partners drawn at random
        #         "global" : prototypes are SLOT-ALIGNED across bags (the paper's setting) by
        #                    clustering all training patches once and defining prototype k of
        #                    every bag as that bag's mean over global cluster k. Ops then pair
        #                    slot k with slot k, and covary gets a PER-SLOT sigma.
        self.align = str(align)
        self.max_global_patches = int(max_global_patches)
        self._bank: Optional[Dict[int, torch.Tensor]] = None   # grade -> [M, D] prototypes (cpu)
        self._sigma: Optional[Dict[int, torch.Tensor]] = None  # grade -> [D] per-grade std (cpu)
        self._gcent: Optional[torch.Tensor] = None             # [C, D] global centroids (cpu)
        self._tried = False

    def _bag_protos_aligned(self, feat_cpu: torch.Tensor) -> torch.Tensor:
        """Slot-aligned prototypes: slot k = this bag's mean over GLOBAL cluster k.

        Because every bag is projected onto the same global clustering, prototype k of bag A
        and prototype k of bag B describe the same phenotype - which is what lets the paper's
        interpolate/replace ops pair corresponding slots instead of arbitrary ones. A slot the
        bag has no patches in falls back to the global centroid.
        """
        g = self._gcent
        a = _assign(feat_cpu, g)
        out = g.clone()
        for k in range(g.shape[0]):
            m = a == k
            if int(m.sum()) > 0:
                out[k] = feat_cpu[m].mean(dim=0)
        return out

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True

        if self.align == "global":
            self._ensure_aligned(pool)
            return

        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            feat = item[0].float()            # cpu [Ni, D]
            protos = _kmeans(feat, self.C)    # [<=C, D] centroids = "reduce"
            chunks[g].append(protos)
        bank: Dict[int, torch.Tensor] = {}
        sigma: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)      # [M, D]
            if allp.shape[0] > self.max_bank:
                sel = torch.randperm(allp.shape[0])[: self.max_bank]
                allp = allp[sel]
            bank[g] = allp
            if allp.shape[0] >= 2:
                sigma[g] = allp.std(dim=0)    # [D] semantic-variation direction
            else:
                sigma[g] = torch.zeros(allp.shape[1])
        self._bank = bank
        self._sigma = sigma

    def _ensure_aligned(self, pool) -> None:
        """Build the global clustering and a per-grade, PER-SLOT prototype bank."""
        feats = [pool[i][0].float() for i in range(len(pool))]
        allp = torch.cat(feats, dim=0)
        if allp.shape[0] > self.max_global_patches:                 # bound the k-means cost
            allp = allp[torch.randperm(allp.shape[0])[: self.max_global_patches]]
        self._gcent = _kmeans(allp, self.C)                         # [C, D] shared slots

        per_grade: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            g = int(round(float(pool[i][1])))
            per_grade[g].append(self._bag_protos_aligned(feats[i]))  # [C, D]

        bank, sigma = {}, {}
        for g, lst in per_grade.items():
            stack = torch.stack(lst, dim=0)                          # [M, C, D]
            bank[g] = stack
            # per-SLOT semantic variation, replacing the single per-grade vector
            sigma[g] = stack.std(dim=0) if stack.shape[0] >= 2 else torch.zeros_like(stack[0])
        self._bank, self._sigma = bank, sigma

    def _call_aligned(self, features: torch.Tensor, label: float) -> Tuple[torch.Tensor, float]:
        dev, dt = features.device, features.dtype
        protos = self._bag_protos_aligned(features.detach().float().cpu()).to(dev, dt)  # [C, D]
        g = int(round(float(label)))
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 1:
            return protos, float(label)
        if float(torch.rand(1).item()) > self.s:
            return protos, float(label)

        op = int(torch.randint(4, (1,)).item()) if self.op is None else self.op
        C, M = protos.shape[0], bank.shape[0]

        if op == 0:      # append: add one same-grade bag's full slot-aligned prototype set
            j = int(torch.randint(M, (1,)).item())
            protos = torch.cat([protos, bank[j].to(dev, dt)], dim=0)

        elif op == 1:    # replace: swap slots with the SAME slots of same-grade donor bags
            k = max(1, min(C, int(round(self.s * C))))
            dst = torch.randperm(C)[:k]
            src = torch.randint(M, (k,))
            protos = protos.clone()
            protos[dst] = bank[src, dst].to(dev, dt)

        elif op == 2:    # interpolate slot k toward slot k of a donor (the paper's pairing)
            src = torch.randint(M, (C,))
            q = bank[src, torch.arange(C)].to(dev, dt)
            lam = float(torch.rand(1).item()) * self.s
            protos = protos + lam * (q - protos)

        else:            # covary with the PER-SLOT sigma
            sig = self._sigma.get(g)
            if sig is not None and float(sig.abs().sum()) > 0.0:
                eps = torch.randn(protos.shape, device=dev, dtype=dt)
                protos = protos + (self.s * eps) * sig.to(dev, dt)

        return protos, float(label)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 2:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        if self.align == "global":
            return self._call_aligned(features, label)
        dev, dt = features.device, features.dtype

        # (1) REDUCE: this bag -> C prototypes (k-means on CPU, back to device)
        protos = _kmeans(features.detach().float().cpu(), self.C).to(dev, dt)  # [P, D]

        g = int(round(float(label)))
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 1:
            # nothing to mix with; still return the reduced bag (faithful "reduce")
            return protos, float(label)

        # (2) MIX-THE-BAG with probability = strength
        if float(torch.rand(1).item()) > self.s:
            return protos, float(label)

        # 0=append 1=replace 2=interpolate 3=covary; pinned when self.op is set
        op = int(torch.randint(4, (1,)).item()) if self.op is None else self.op
        P = protos.shape[0]
        bsz = bank.shape[0]

        if op == 0:  # append
            k = max(1, min(self.C, bsz))
            src = torch.randint(bsz, (k,))
            extra = bank[src].to(dev, dt)
            protos = torch.cat([protos, extra], dim=0)

        elif op == 1:  # replace
            k = max(1, int(round(self.s * P)))
            k = min(k, P, bsz)
            dst = torch.randperm(P)[:k]
            src = torch.randint(bsz, (k,))
            protos = protos.clone()
            protos[dst] = bank[src].to(dev, dt)

        elif op == 2:  # interpolate toward a same-grade bank prototype
            src = torch.randint(bsz, (P,))
            q = bank[src].to(dev, dt)                       # [P, D] partner prototypes
            lam = float(torch.rand(1).item()) * self.s      # lam ~ U(0, strength)
            protos = protos + lam * (q - protos)

        else:  # op == 3: covary - per-grade std-scaled Gaussian "semantic variation"
            sigma = self._sigma.get(g)
            if sigma is not None and float(sigma.abs().sum()) > 0.0:
                sig = sigma.to(dev, dt)
                eps = torch.randn(protos.shape, device=dev, dtype=dt)
                protos = protos + (self.s * eps) * sig.unsqueeze(0)

        return protos, float(label)
