"""two_branch_transmil — TransMIL over TWO views of the same ROI (base module for a380-a385).

THE TWO VIEWS (both already extracted, no new data)
  patch view  `features_{bb}_reti`          ~44 tiles cut from the ROI at native resolution,
                                            NOT resized. High detail, no field context.
  field view  `features_{bb}_reti_no_patch`  the WHOLE ROI downsized to 224 px, one vector.
                                            Field context (fibre continuity, intersections),
                                            no detail.

Why two branches rather than one. The WHO/EUMNET criterion separating MF-2/3 uses "extensive
intersections", a property of the fibre network ACROSS the field; once the ROI is cut into 224 px
tiles no tile can see it. Measured on this cohort (2026-08-08): a field-only linear model reaches
test QWK .875-.894 on its own, and its errors are complementary to the patch model's — it is
stronger on G0/G1, much weaker on G2/G3 — with an oracle union of 87.6-92.7% accuracy against
76.8-88.4% for the patch model alone.

MODES (`fuse`)
  "late"   a380 — the literal two-branch design. Branch A = TransMIL over the patch tokens;
                  branch B = an MLP over the field vector; the two bag descriptors are summed
                  through separate heads. Branch B's head is ZERO-INITIALISED, so the model
                  starts bit-identical to single-branch TransMIL.
  "token"  a381 — transformer-native fusion: the field vector becomes an extra TOKEN in the same
                  sequence as [CLS] and the patches, carrying a learned modality embedding so the
                  model can tell it apart. Self-attention then lets every patch attend to the
                  field and the field attend to every patch, instead of the two views only
                  meeting at the very end.
  "cross"  a382 — the field is a QUERY that cross-attends over the contextualised patch tokens;
                  the resulting descriptor is fused with the [CLS] descriptor.

PPEG is applied to the PATCH tokens only; the [CLS] and field tokens are held out of the
squarified grid, which would otherwise mix a non-spatial token into a spatial convolution.

CONTROLS
  `field="mean"`    the field vector is replaced by the mean patch embedding: same architecture,
                    same parameter count, but NO information the patch bag did not already
                    contain. Answers "does the extra token need to carry new information?".
  `field="shuffle"` the field vector of a DIFFERENT ROI, chosen deterministically from the bag's
                    fingerprint. Same marginal distribution, correspondence destroyed. Answers
                    "does the extra token need to be THIS ROI's field?" — the control that
                    falsified a340-a361.
  Legacy note: the mean control alone Any two-branch gain that does
  not survive this control is not coming from the field view. `field="patient"` uses the
  leave-one-out mean field view of the same patient's other ROIs (transductive at the patient
  level; images only, never labels).

HOW TO READ THE RESULT. Report ROI-level AND patient-level metrics
(`scripts/patient_level_eval.py`). ROI labels are constant within a patient, so a mechanism that
merely makes a patient's ROIs agree raises ROI-QWK without changing any clinical decision; that
is exactly what happened to the a340-a361 field family.

Deterministic at eval. Field vectors are looked up by a value fingerprint of the incoming bag
(same device as `field_mil.py`), so this stays a drop-in `--novelty_id` with zero trainer edits.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .field_mil import DIM_TO_BACKBONE, _bank, _fingerprint  # noqa: F401

FUSE_MODES = ("late", "token", "cross", "cls", "clsadd", "clsboth",
              "film", "multi", "clsppeg")
FIELD_SOURCES = ("roi", "mean", "shuffle", "patient")


class _TransLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 8, zero_init: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.attn(h, h, h, need_weights=False)[0]


class _PPEG(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is ALREADY squarified to S*S tokens by the caller (TransMIL's reference pads the
        patch sequence up front, before the first attention layer, and keeps the repeats)."""
        n, D = x.shape[1], x.shape[2]
        S = int(round(math.sqrt(n)))
        g = x.transpose(1, 2).reshape(1, D, S, S)
        g = g + self.proj(g) + self.proj1(g) + self.proj2(g)
        return g.flatten(2).transpose(1, 2)


class _PPEG1D(nn.Module):
    """Depthwise convolution along the SEQUENCE instead of a squarified grid.

    Measured on this cohort: in the ceil(sqrt(N)) grid the HORIZONTAL neighbour of a token is its
    true spatial neighbour 89.1% of the time, the VERTICAL neighbour only 3.3% of the time. The
    2-D form therefore spends two of its three axes mixing tokens that are not neighbours at all.
    This keeps the axis that is real and drops the one that is not. Kernels are unchanged (7/5/3),
    so the receptive field is 7 of ~40 sequence positions (17.5%) rather than the whole grid.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv1d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv1d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.transpose(1, 2)                                  # [1, D, N]
        g = g + self.proj(g) + self.proj1(g) + self.proj2(g)
        return g.transpose(1, 2)


class _PPEGSmall(nn.Module):
    """PPEG with the kernel pyramid rescaled from gigapixel bags to ROI bags.

    TransMIL's 7/5/3 pyramid was designed for N ~ 10^4 patches, where the grid is ~100 wide and a
    7-kernel spans 7% of it -- genuinely coarse-to-fine positional context. Measured here: the
    median ROI has N=40, so S=7, and 91.0% of the 1330 ROIs have S <= 7. At that size a 7-kernel
    spans the ENTIRE grid, and 5 is nearly as wide, so the pyramid collapses into three copies of
    a bag-level average. 5/3/1 restores an actual coarse-to-fine ordering at S=7 (71% / 43% / 14%).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, D = x.shape[1], x.shape[2]
        S = int(round(math.sqrt(n)))
        g = x.transpose(1, 2).reshape(1, D, S, S)
        g = g + self.proj(g) + self.proj1(g) + self.proj2(g)
        return g.flatten(2).transpose(1, 2)


class _TokenFFN(nn.Module):
    """CONTROL: same parameter budget as PPEG, but NO mixing across tokens whatsoever.

    If this matches PPEG, the benefit of PPEG is depth and capacity, not neighbourhood mixing --
    and the whole positional-encoding framing for this data is finished. hidden=42 gives 43,562
    parameters against PPEG's 44,032.
    """

    def __init__(self, dim: int, hidden: int = 42):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _PPEGBoth(nn.Module):
    """Both mixing geometries at once: the squarified 2-D grid AND the raw 1-D sequence.

    Measured separately (a445 vs a458), the two win on different backbones -- the 2-D form on
    virchow2 (+.0098) and titan (+.0035), the 1-D form on uni2 (+.0140). This asks whether a model
    can have both. Each branch keeps its own kernels and both are added to the residual, so the
    2-D path is bit-identical to PPEG at init and the 1-D path is pure addition.

    Prediction, stated first: this loses. Every combination tried in this project has been at best
    equal to its parts -- a448 and a449 (two ways of having the global view twice) lost 2/3 and
    3/3, and the wider augmentation stacks never exceeded their components. If it loses, that is
    the fourth independent confirmation that these mechanisms do not compose on 50 patients.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.g2 = _PPEG(dim)
        self.g1 = _PPEG1D(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g2(x) + self.g1(x) - x       # both residuals, x counted once


class _ROIScale(nn.Module):
    """Purpose-built replacement for PPEG at ROI scale. Permutation-invariant, grid-free.

    PPEG's coarse/medium/fine pyramid is defined by KERNEL WIDTH on a squarified grid. On this
    cohort that grid is fabricated -- ceil(sqrt(N)) matches the true ROI width for almost no bag,
    a token's vertical neighbour there is a true neighbour only 3.3% of the time, and the operator
    changes behaviour discontinuously when N crosses a square (N=49 -> S=7, N=50 -> S=8). It also
    forces padding that duplicates 14% of the sequence, and it breaks permutation invariance for a
    signal that was measured to be absent (patch coordinates carry no grade information here).

    This keeps the ONE property that the ablations showed actually matters -- mixing each token
    with a broad, long-range summary rather than with immediate neighbours -- and defines the
    pyramid by BAG FRACTION and FEATURE SIMILARITY instead of by pixel geometry:

        fine    the token itself
        medium  mean of its k most similar tokens, k = N/4        (content neighbourhood)
        coarse  mean of the whole bag                             (field density)

    Each scale gets a learned per-channel gate, so the model can fall back to identity. All three
    terms are permutation-invariant and defined for any N without padding, so the operator behaves
    identically at N=13 and N=112. 2,560 parameters against PPEG's 44,032.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w_self = nn.Parameter(torch.zeros(dim))
        self.w_near = nn.Parameter(torch.zeros(dim))
        self.w_bag = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.use_near = True
        self.grid_free = True          # needs no squarified grid -> caller skips the padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)                                       # [1, N, D]
        N = h.shape[1]
        bag = h.mean(dim=1, keepdim=True).expand_as(h)         # coarse
        if self.use_near and N > 2:
            k = max(2, int(round(N / 4)))
            u = torch.nn.functional.normalize(h, dim=-1)
            idx = (u @ u.transpose(1, 2)).topk(k, dim=-1).indices          # [1, N, k]
            near = h.squeeze(0)[idx.squeeze(0)].mean(dim=1).unsqueeze(0)   # medium
        else:
            near = bag
        return x + self.w_self * h + self.w_near * near + self.w_bag * bag


POS_MODES = ("ppeg", "ppeg1d", "ppeg_small", "ffn", "both", "roiscale", "roibag", "none")


def _make_pos(mode: str, dim: int):
    if mode == "ppeg":
        return _PPEG(dim)
    if mode == "ppeg1d":
        return _PPEG1D(dim)
    if mode == "ppeg_small":
        return _PPEGSmall(dim)
    if mode == "ffn":
        return _TokenFFN(dim)
    if mode == "both":
        return _PPEGBoth(dim)
    if mode == "roiscale":
        return _ROIScale(dim)
    if mode == "roibag":                       # control: drop the similarity scale
        m = _ROIScale(dim); m.use_near = False
        return m
    return None


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 512,
                 heads: int = 8, dropout: float = 0.25, fuse: str = "late",
                 field: str = "roi", use_ppeg: bool = True, sep_proj: bool = False,
                 pos_mode: str = "ppeg", n_layers: int = 2, field_backbone: Optional[str] = None,
                 reinject: bool = False, data_root: Optional[str] = None):
        super().__init__()
        if fuse not in FUSE_MODES:
            raise ValueError(f"fuse must be one of {FUSE_MODES}")
        if field not in FIELD_SOURCES:
            raise ValueError(f"field must be one of {FIELD_SOURCES}")
        self.fuse, self.field, self.dim = fuse, field, hidden_dim
        from pathlib import Path
        import os
        self.data_root = Path(data_root or os.environ.get("MPN_DATA_ROOT", "data"))

        # shared projection: both views come from the SAME frozen encoder, so they share it
        self._fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        # `sep_proj` gives the field view its OWN projection instead of reusing the patch one.
        # The two views come from the same frozen encoder but are not the same kind of input --
        # a patch is a 224 px crop at native resolution, the field is the whole ROI downsized to
        # 224 px -- so one shared matrix has to be a compromise. Costs +input_dim*hidden_dim
        # params, which is why it is OFF by default: the shared form keeps a445's parameter count
        # at the plain-TransMIL number, and that is what removes the capacity confound.
        self._fc1g = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()) if sep_proj else None
        # `field_backbone`: take the whole-ROI vector from a DIFFERENT encoder than the patches.
        # Motivation: TITAN is a slide-level encoder built to summarise a field; Virchow2/UNI2 are
        # patch encoders. a445 makes one encoder do both jobs. The field vector then has that
        # encoder's width, so it needs its own projection -- this is forced by the dimension
        # mismatch, not a design choice, and it is the a456 confound (sep_proj) by necessity.
        self.field_backbone = field_backbone
        if field_backbone is not None:
            _FIELD_DIM = {"titan": 768, "virchow2": 1280, "uni2": 1536}
            self._fc1g = nn.Sequential(nn.Linear(_FIELD_DIM[field_backbone], hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.layer1 = _TransLayer(hidden_dim, heads)
        if pos_mode not in POS_MODES:
            raise ValueError(f"pos_mode must be one of {POS_MODES}")
        if n_layers not in (1, 2):
            raise ValueError("n_layers must be 1 or 2")
        self.n_layers = n_layers
        # n_layers=1: a single round of attention from the readout token, then read out. PPEG
        # only ever edits PATCH tokens between the two rounds, so with one round it could not
        # reach the readout at all -- it is dropped rather than left allocated and inert.
        self.pos_layer = _make_pos(pos_mode, hidden_dim) if (use_ppeg and n_layers == 2) else None
        self.layer2 = _TransLayer(hidden_dim, heads) if n_layers == 2 else None
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, num_classes)

        if fuse == "late":
            self.field_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.field_head = nn.Linear(hidden_dim, num_classes, bias=False)
            nn.init.zeros_(self.field_head.weight)      # starts as single-branch TransMIL
        elif fuse == "token":
            self.modality = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        elif fuse in ("cls", "clsadd"):
            pass                                         # field replaces/biases CLS; nothing extra
        elif fuse == "clsboth":
            self.modality = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        elif fuse == "film":
            # global conditions EVERY patch token (scale+shift) before any attention. Never tried
            # in TransMIL; the ABMIL-family FiLM variants (a341/a344/a345/a349) died to the shuffle
            # control, so the same control is mandatory here.
            self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # starts = a373
        elif fuse == "multi":
            # the single global vector expanded into K tokens with distinct learned offsets, so the
            # sequence carries several "views" of the same field instead of one.
            self.multi = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)
        elif fuse == "clsppeg":
            pass
        else:                                            # cross
            self.cross = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
            nn.init.zeros_(self.cross.out_proj.weight)
            nn.init.zeros_(self.cross.out_proj.bias)
            self.cross_head = nn.Linear(hidden_dim, num_classes, bias=False)
            nn.init.zeros_(self.cross_head.weight)

        # `reinject`: add the field vector back onto the readout token BETWEEN the two attention
        # rounds. Motivation is measured, not assumed -- the readout's attention is markedly more
        # concentrated after layer 1 than after layer 2 (effective-N / N = .515/.630/.261 vs
        # .828/.896/.345), i.e. the field-initialised readout drifts back toward uniform as the
        # second round mixes it. The gate starts at ZERO, so training begins bit-identical to a445
        # and the model can keep it at zero if re-anchoring is not wanted. One parameter.
        self.reinject = reinject
        self.reinject_gate = nn.Parameter(torch.zeros(())) if reinject else None

        self._bank_obj = None
        self._hit = 0
        self._miss = 0
        self._warned = False

    # ── field vector ─────────────────────────────────────────────────────
    def _field(self, features: torch.Tensor) -> torch.Tensor:
        if self.field == "mean":
            return features.mean(dim=0)
        if self._bank_obj is None:
            self._bank_obj = _bank(int(features.shape[1]), self.data_root, self.field_backbone)
        key = _fingerprint(features)
        if self.field == "shuffle":
            # DECISIVE CONTROL. The field token still carries a real whole-ROI embedding, with the
            # same marginal distribution and the same scale — but taken from a DIFFERENT ROI,
            # picked deterministically from this bag's own fingerprint (stable across epochs and
            # at eval). Correspondence between the patch bag and the field view is destroyed while
            # everything else is held fixed. This is the control that falsified the a340-a361
            # FC-MIL family; a381 must beat it or the field token is only extra capacity.
            stack = self._bank_obj.stack
            self._hit += 1
            return stack[abs(hash(key)) % stack.shape[0]].to(features.device, features.dtype)
        vec = self._bank_obj.table.get(key)
        if vec is None:
            self._miss += 1
            if not self._warned:
                self._warned = True
                print("  ⚠ two_branch_transmil: bag missing from the field bank — using the "
                      "patch mean. Expected only when a feature-space augmentation is active.")
            return features.mean(dim=0)
        self._hit += 1
        if self.field == "patient":
            ctx = self._bank_obj.context(key)
            if ctx is not None:
                vec = ctx
        return vec.to(features.device, features.dtype)

    def field_hit_rate(self) -> float:
        n = self._hit + self._miss
        return float(self._hit) / n if n else 0.0

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        N = features.shape[0]
        g = self._field(features)

        patches = self._fc1(features).unsqueeze(0)                    # [1, N, D]
        # squarify UP FRONT, before the first attention layer, exactly as TransMIL's reference
        # does — this is what makes a380/a382 bit-identical to single-branch TransMIL at init.
        if not getattr(self.pos_layer, "grid_free", False):
            # TransMIL squarifies up front so PPEG can reshape to a grid. A grid-free position
            # operator has no use for it, and the padding repeats the head of the sequence --
            # which is what breaks permutation invariance and over-weights the first patches.
            if self.fuse == "clsppeg":
                # the field token joins the grid, so the PATCH block must be S*S-1 long
                S = int(math.ceil(math.sqrt(N + 1)))
                pad = S * S - 1 - N
            else:
                S = int(math.ceil(math.sqrt(N)))
                pad = S * S - N
            if pad:
                patches = torch.cat([patches, patches[:, :pad, :]], dim=1)
        proj_g = self._fc1g if self._fc1g is not None else self._fc1
        fvec = proj_g(g.unsqueeze(0)).unsqueeze(0)                    # [1, 1, D]
        cls = self.cls_token.to(patches.dtype)

        if self.fuse == "token":
            seq = torch.cat([cls, fvec + self.modality, patches], dim=1)
            n_head = 2                                                # CLS + field
        elif self.fuse == "clsboth":
            # a381 + a445 together: the field is BOTH the readout token (position 0, a445) and an
            # attendable element in the sequence with its own modality marker (position 1, a381).
            # The two roles are different -- one is what gets read out, the other is one of the
            # things being read -- so the same vector appearing twice is not a duplicate.
            seq = torch.cat([fvec, fvec + self.modality, patches], dim=1)
            n_head = 2
        elif self.fuse == "clsadd":
            # The learned CLS is KEPT and merely biased by this ROI's whole-view embedding, so the
            # readout token has a component that is constant across bags and a component that is
            # not. a445 ("cls") replaces it outright; this one mixes.
            seq = torch.cat([cls + fvec, patches], dim=1)
            n_head = 1
        elif self.fuse == "film":
            # patches are modulated by the field; the readout stays the learned CLS (as in a373)
            g_, b_ = self.film(fvec).chunk(2, dim=-1)
            patches = patches * (1 + g_) + b_
            seq = torch.cat([cls, patches], dim=1)
            n_head = 1
        elif self.fuse == "multi":
            seq = torch.cat([cls, fvec + self.multi, patches], dim=1)
            n_head = 5                                    # CLS + 4 field views
        elif self.fuse == "clsppeg":
            # a445, but the field token is NOT held out of PPEG. PPEG was measured to act as
            # bag-level smoothing rather than positional encoding at ROI scale, so excluding the
            # field from it is no longer obviously right.
            seq = torch.cat([fvec, patches], dim=1)
            n_head = 0
        elif self.fuse == "cls":
            # The field IS the readout token: instead of starting the aggregation from a learned
            # constant that is identical for every bag, start it from this ROI's own whole-view
            # embedding. The field stops being a thing the model READS (a381) and becomes the
            # thing that DOES the reading.
            # self.cls_token is left allocated but unused (no gradient): allocated count equals
            # plain TransMIL, effective count is 512 lower.
            seq = torch.cat([fvec, patches], dim=1)
            n_head = 1
        else:
            seq = torch.cat([cls, patches], dim=1)
            n_head = 1

        seq = self.layer1(seq)
        if self.pos_layer is not None:
            seq = torch.cat([seq[:, :n_head], self.pos_layer(seq[:, n_head:])], dim=1)
        if self.reinject_gate is not None:
            seq = torch.cat([seq[:, :1] + self.reinject_gate * fvec, seq[:, 1:]], dim=1)
        if self.layer2 is not None:
            seq = self.layer2(seq)

        z = self.norm(seq)[:, 0]                                      # [1, D] CLS descriptor
        y = self._fc2(z)

        if self.fuse == "late":
            y = y + self.field_head(self.field_mlp(fvec.squeeze(1)))
        elif self.fuse == "cross":
            ctx = self.cross(fvec, seq[:, n_head:], seq[:, n_head:], need_weights=False)[0]
            y = y + self.cross_head(ctx.squeeze(1))

        y = y.view(-1)
        if return_attention:
            a = torch.full((N,), 1.0 / N, device=features.device)
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, fuse="late", field="roi")
