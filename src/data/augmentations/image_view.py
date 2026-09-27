"""image_view - D4 image-space augmentation, replayed from a pre-extracted feature bank.

This is the only augmentation in the registry that is NOT synthesised in feature space:
each patch is replaced by the frozen-backbone embedding of a dihedrally transformed
version of the SAME patch, computed offline by ``src/data/extract_reti_views.py``.
Because the encoder is frozen, an image-space augmentation is fully equivalent to
sampling from that pre-computed bank - no encoder forward pass is needed at train time,
and no trainer or dataset edits are needed either.

    view 0 identity (the bag the trainer already handed us - never re-read from disk)
    view 1 rot90    2 rot180    3 rot270
    view 4 flip_lr  5 transpose 6 flip_tb  7 transverse

Sampling is PER PATCH and independent, which is what makes a small bank sufficient: a
bag of N=40 patches over 8 views spans 8^40 distinct configurations, so view count is
never the bottleneck. ``strength`` is the per-patch probability of re-drawing a view
(1.0 = every patch draws uniformly from the available views, the standard D4 setting;
0.0 = OFF, byte-identical to no augmentation).

Only transforms that preserve the MF grade are used. Fibre density, thickness and
continuity per unit area ARE the grading criterion, so scale/zoom, elastic warping and
free-angle rotation are excluded by construction; the eight dihedral transforms are exact
pixel permutations and a fibre network has no canonical orientation.

Bags are matched to their bank entry by a value fingerprint of the incoming features,
since the augmentation contract passes features only (no slide_id). The index is built
once, lazily, from the identity feature directory. A bag with no bank entry - or with a
mismatched shape - is passed through unchanged, so a partially extracted bank degrades
gracefully instead of crashing a run.

    Requires: python -m src.data.extract_reti_views --backbone <bb> --views 1-7
    Train with: --augmentation image_view [--aug_strength 1.0]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from . import BaseAugmentation

KWARGS = dict(strength=1.0)

# Feature dimension -> backbone name, so the module needs no --backbone flag.
DIM_TO_BACKBONE = {768: "titan", 1280: "virchow2", 1536: "uni2"}


class Augmentation(BaseAugmentation):
    requires_regression = False  # the label is never changed

    def __init__(
        self,
        strength: float = 1.0,
        data_root: Optional[str] = None,
        max_cached_bags: int = 4096,
    ):
        self.strength = float(strength)
        self.data_root = Path(data_root or os.environ.get("MPN_DATA_ROOT", "data"))
        self.max_cached_bags = int(max_cached_bags)

        # Lazily initialised on the first call, once the feature dim reveals the backbone.
        self._ready = False
        self._backbone: Optional[str] = None
        self._index: Dict[tuple, Path] = {}
        self._view_dirs: list = []
        self._cache: Dict[Path, torch.Tensor] = {}
        self._warned_miss = False

    # ── bank discovery ────────────────────────────────────────────────────
    def _setup(self, dim: int) -> None:
        self._ready = True

        backbone = DIM_TO_BACKBONE.get(dim)
        if backbone is None:
            raise ValueError(
                f"image_view: feature dim {dim} does not match a known backbone "
                f"({DIM_TO_BACKBONE}). Extract the bank first with "
                f"`python -m src.data.extract_reti_views --backbone <bb> --views 1-7`."
            )
        self._backbone = backbone

        identity_root = self.data_root / f"features_{backbone}_reti"
        views_root = self.data_root / f"features_{backbone}_reti_views"

        self._view_dirs = sorted(
            (d for d in views_root.glob("view*") if d.is_dir() and d.name != "view0"),
            key=lambda d: int(d.name.removeprefix("view")),
        )
        if not self._view_dirs:
            raise FileNotFoundError(
                f"image_view: no view directories under {views_root}. Run "
                f"`python -m src.data.extract_reti_views --backbone {backbone} --views 1-7` first."
            )

        # Fingerprint every identity bag so an incoming [N, D] tensor can be traced back
        # to its .pt path without the trainer having to pass a slide id.
        collisions = 0
        for pt_path in sorted(identity_root.rglob("*.pt")):
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            feats = data["feats"] if isinstance(data, dict) else data
            key = self._fingerprint(feats)
            if key in self._index:
                collisions += 1
            self._index[key] = pt_path.relative_to(identity_root)

        if collisions:
            print(f"  ⚠ image_view: {collisions} fingerprint collisions in {identity_root}")

        # A half-extracted bank is the dangerous failure mode: bags whose view files are
        # missing would silently go un-augmented, so the run would compare a mixture of
        # augmented and clean bags against itself. Refuse to start instead.
        expected = len(self._index)
        incomplete = [
            (d.name, n)
            for d, n in ((d, sum(1 for _ in d.rglob("*.pt"))) for d in self._view_dirs)
            if n != expected
        ]
        if incomplete:
            detail = ", ".join(f"{name}: {n}/{expected}" for name, n in incomplete)
            raise RuntimeError(
                f"image_view: incomplete view bank for {backbone} ({detail}). "
                f"Finish extraction first - re-running the same command resumes and only "
                f"fills the gaps:\n"
                f"    python -m src.data.extract_reti_views --backbone {backbone} "
                f"--views 1-3 --dtype fp32 --device cuda"
            )

        print(
            f"  image_view: {self._backbone} | {len(self._index)} bags indexed | "
            f"{len(self._view_dirs)} views ({', '.join(d.name for d in self._view_dirs)}) | "
            f"p={self.strength}"
        )

    @staticmethod
    def _fingerprint(feats: torch.Tensor) -> tuple:
        """Cheap, exact identifier for a bag: shape plus four corner values."""
        n, d = int(feats.shape[0]), int(feats.shape[1])
        f = feats.detach().to(torch.float32)
        return (
            n,
            d,
            float(f[0, 0]),
            float(f[0, -1]),
            float(f[-1, 0]),
            float(f[-1, -1]),
        )

    def _bank(self, rel: Path, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return the stacked non-identity views [V, N, D] for one bag, cached in RAM."""
        cached = self._cache.get(rel)
        if cached is not None:
            return cached

        views = []
        for view_dir in self._view_dirs:
            pt_path = view_dir / rel
            if not pt_path.exists():
                return None
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            feats = data["feats"] if isinstance(data, dict) else data
            views.append(feats.to(dtype))

        stacked = torch.stack(views, dim=0).to(device)

        # Only train bags are ever augmented, so this converges to the training split
        # after the first epoch; the bound is a safety net, not an expected path.
        if len(self._cache) < self.max_cached_bags:
            self._cache[rel] = stacked
        return stacked

    # ── augmentation ──────────────────────────────────────────────────────
    def __call__(
        self, features: torch.Tensor, label: float, pool=None
    ) -> Tuple[torch.Tensor, float]:
        if self.strength <= 0.0:
            return features, float(label)

        if not self._ready:
            self._setup(int(features.shape[1]))

        rel = self._index.get(self._fingerprint(features))
        if rel is None:
            if not self._warned_miss:
                self._warned_miss = True
                print("  ⚠ image_view: bag not found in the view index; passing through unchanged")
            return features, float(label)

        bank = self._bank(rel, features.device, features.dtype)
        if bank is None or bank.shape[1:] != features.shape:
            if not self._warned_miss:
                self._warned_miss = True
                print("  ⚠ image_view: incomplete/mismatched bank entry; passing through unchanged")
            return features, float(label)

        n_patches = features.shape[0]
        n_views = bank.shape[0]

        # 0 = keep identity, 1..V = take view v-1 from the bank. Drawing over V+1 options
        # means p=1.0 reproduces the standard "uniform over all eight D4 transforms".
        choice = torch.randint(n_views + 1, (n_patches,), device=features.device)
        redraw = torch.rand(n_patches, device=features.device) < self.strength
        choice = torch.where(redraw, choice, torch.zeros_like(choice))

        out = features.clone()
        for v in range(1, n_views + 1):
            mask = choice == v
            if mask.any():
                out[mask] = bank[v - 1][mask]

        return out, float(label)
