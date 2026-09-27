"""field_mil — Field-Conditioned MIL (FC-MIL): the base module for the a340-a349 family.

MOTIVATION (grading-specific, not a generic MIL trick)
------------------------------------------------------
The WHO/EUMNET reticulin criterion is not only *how much* fibre is present but how it is
ARRANGED: "diffuse and dense increase ... with extensive intersections" separates MF-2/3
from MF-1. Intersections, continuity and network structure are FIELD-level properties. Once
an ROI is cut into 224 px patches, every patch sees fibre density but no patch sees the
network. Standard MIL cannot recover it either: it pools a set of exchangeable instances, so
whatever was destroyed by patching is gone before the aggregator runs.

This project already extracted the missing view — ``features_{bb}_reti_no_patch`` holds the
same frozen encoder applied to the WHOLE ROI downsized to 224 px, one vector per ROI. On its
own it is a much weaker grader than patch MIL (test QWK 0.71-0.79 vs 0.95) but it FAILS
DIFFERENTLY: G1 recall 90-98% where patch MIL is weakest, G0 recall 0-50% where patch MIL is
strong. Two views with the same label and complementary error profiles is exactly the setting
where conditioning one on the other should pay.

MECHANISM
---------
Let X in R^{NxD} be the patch bag and g in R^D the whole-ROI (field) embedding of the SAME
ROI. Both go through the SAME bottleneck (weight-shared, so the field costs no extra encoder
parameters and lands in the same space as the patches):

    h_i = Bottleneck(x_i)        q = Bottleneck(g)

``attn`` — how the field enters the attention:
  gated    e_i = W( tanh(V h_i) (.) sigmoid(U h_i) )                     ABMIL/ASGAP, no field
  film     e_i = W( (( 1 + gamma(q) ) (.) tanh(V h_i) + beta(q)) (.) sigmoid(U h_i) )
           the field RE-SCALES the attention feature space per ROI: "given this field, these
           are the directions worth attending to".
  query    e_i = <W_q q, h_i> / sqrt(H)
           the sharpest statement of the idea. ABMIL's attention query ``W`` is a SINGLE
           parameter shared by every bag — the aggregator asks the same question of every
           ROI. Here the query is computed from the ROI's own field view, so the aggregator
           asks what *this* field suggests is worth looking for.

``pool`` — entmax (alpha=1.5, ASGAP) or softmax (ABMIL). Kept separate from the field
mechanism so FC-MIL can be compared against either baseline on equal terms.

``readout`` — how the field enters the prediction:
  none     y = C(z)                                   field only steers attention
  concat   y = C([z ; q])                             field also contributes evidence
  gate     y = s * C(z) + (1 - s) * C_f(q),  s = sigmoid(w.q)
           per-ROI arbitration between the two magnifications.

All field-dependent paths are ZERO-INITIALISED, so at step 0 the model is bit-identical to
its fieldless baseline (ASGAP for pool="entmax", ABMIL for pool="softmax") and any gain is
something training had to find, not a different starting point. ``attn="query"`` is the one
exception — it has no fieldless limit by construction.

CONTROLS (the claim is only meaningful with these)
--------------------------------------------------
``field="mean"``     q = Bottleneck(mean_i x_i). Same architecture, same parameter count,
                     same per-bag adaptivity — but NO information the bag did not already
                     contain. Isolates "is the gain the un-patched FIELD VIEW, or merely
                     having any bag-adaptive query?"
``field="shuffle"``  q taken from a DIFFERENT ROI's field view, chosen deterministically from
                     the bag's own fingerprint. Same marginal distribution of q, correspondence
                     destroyed. Isolates "is the gain this ROI's field, or just an extra
                     plausible vector?"

PATIENT CONTEXT
---------------
``field="patient"``  q from the LEAVE-ONE-OUT mean of the field views of the SAME PATIENT's
                     OTHER ROIs. Motivated by a measured property of this cohort: within-
                     patient prediction spread is 44% of the between-patient spread, i.e. the
                     ROIs of one patient disagree substantially, yet the label — and the
                     clinical decision — is per patient, and a pathologist grading a case sees
                     every field, not one. Leave-one-out means the context can never contain
                     the ROI's own field view, so the model cannot shortcut to ``field="roi"``.
``field="ownpat"``   both: q = W_q(own field) + W_c(patient context). Only meaningful with
                     ``attn="query"``.

NOT LABEL LEAKAGE, BUT TRANSDUCTIVE — state this explicitly in any write-up. The context is
built from IMAGES only, never from labels, and patients never cross the split boundary. It
does assume the other ROIs of the case are available at inference, which is true in practice
(they are cut from the same slide) but makes the setting transductive at the patient level.

IMPLEMENTATION NOTE
-------------------
The trainer's model contract passes ``features`` only, with no slide id, so the field vector
is looked up by a value fingerprint of the incoming bag — the same device used by the
``image_view`` augmentation. This keeps FC-MIL a drop-in ``--novelty_id`` with ZERO trainer or
dataset edits. A bag that is not in the bank (e.g. a run with feature-space augmentation
enabled, which perturbs the fingerprint) falls back to the patch mean and is counted; the
count is printed once so a silently-degraded run cannot pass unnoticed.

Deterministic at eval, permutation-invariant, bag-size-invariant.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect

DIM_TO_BACKBONE = {768: "titan", 1280: "virchow2", 1536: "uni2"}

ATTN_MODES = ("gated", "film", "query", "query2", "queryk")
POOL_MODES = ("entmax", "softmax")
READOUT_MODES = ("none", "concat", "gate")
FIELD_SOURCES = ("roi", "mean", "shuffle", "patient", "ownpat", "resid", "static", "random")


# ── field bank ───────────────────────────────────────────────────────────────
def _fingerprint(feats: torch.Tensor) -> tuple:
    """Cheap exact identifier for a bag: shape plus four corner values."""
    f = feats.detach().to(torch.float32)
    return (
        int(feats.shape[0]),
        int(feats.shape[1]),
        float(f[0, 0]),
        float(f[0, -1]),
        float(f[-1, 0]),
        float(f[-1, -1]),
    )


class _FieldBank:
    """fingerprint(patch bag) -> whole-ROI embedding, built once per process."""

    def __init__(self, dim: int, data_root: Path, field_backbone: Optional[str] = None):
        backbone = DIM_TO_BACKBONE.get(dim)
        if backbone is None:
            raise ValueError(
                f"field_mil: feature dim {dim} matches no known backbone ({DIM_TO_BACKBONE})."
            )
        self.backbone = backbone
        # `field_backbone` pairs this encoder's PATCH bags with another encoder's whole-ROI
        # vectors for the same ROI (same {Class}/{Patient}/{Img}.pt path under both roots).
        self.field_backbone = field_backbone or backbone
        patch_root = data_root / f"features_{backbone}_reti"
        field_root = data_root / f"features_{self.field_backbone}_reti_no_patch"
        if not field_root.is_dir():
            raise FileNotFoundError(
                f"field_mil: no field bank at {field_root}. Extract it first:\n"
                f"    python -m src.data.extract_{backbone}_reti_no_patch"
            )

        self.table: Dict[tuple, torch.Tensor] = {}
        self.order: list = []           # insertion order, for the deterministic shuffle control
        self.patient: Dict[tuple, str] = {}
        psum: Dict[str, torch.Tensor] = {}
        pcount: Dict[str, int] = {}
        missing = 0
        for pt in sorted(patch_root.rglob("*.pt")):
            gp = field_root / pt.relative_to(patch_root)
            if not gp.exists():
                missing += 1
                continue
            pd = torch.load(pt, map_location="cpu", weights_only=False)
            gd = torch.load(gp, map_location="cpu", weights_only=False)
            feats = pd["feats"] if isinstance(pd, dict) else pd
            gvec = (gd["feats"] if isinstance(gd, dict) else gd).reshape(-1).float()
            key = _fingerprint(feats)
            pid = pt.parent.name                       # .../{Class}/{PatientID}/{ImgID}.pt
            self.table[key] = gvec
            self.patient[key] = pid
            self.order.append(gvec)
            psum[pid] = gvec.clone() if pid not in psum else psum[pid] + gvec
            pcount[pid] = pcount.get(pid, 0) + 1
        if not self.table:
            raise RuntimeError(f"field_mil: field bank for {backbone} indexed 0 bags.")
        if missing:
            print(f"  ⚠ field_mil: {missing} patch bags had no field counterpart")
        self.stack = torch.stack(self.order, dim=0)      # [M, D]
        self.psum, self.pcount = psum, pcount
        print(
            f"  field_mil: {backbone} field bank | {len(self.table)} ROIs | "
            f"{len(pcount)} patients (median {sorted(pcount.values())[len(pcount)//2]} ROIs/patient)"
        )

    def context(self, key: tuple) -> Optional[torch.Tensor]:
        """Leave-one-out mean field view of the SAME patient's other ROIs."""
        pid = self.patient.get(key)
        if pid is None or self.pcount[pid] < 2:
            return None
        return (self.psum[pid] - self.table[key]) / (self.pcount[pid] - 1)


_BANKS: Dict[tuple, _FieldBank] = {}


def _bank(dim: int, data_root: Path, field_backbone: Optional[str] = None) -> _FieldBank:
    key = (dim, str(data_root), field_backbone)
    if key not in _BANKS:
        _BANKS[key] = _FieldBank(dim, data_root, field_backbone)
    return _BANKS[key]


# ── model ────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        attn: str = "film",
        pool: str = "entmax",
        readout: str = "concat",
        field: str = "roi",
        alpha: float = 1.5,
        heads: int = 4,
        data_root: Optional[str] = None,
    ):
        super().__init__()
        self.n_query = 1
        for name, val, allowed in (
            ("attn", attn, ATTN_MODES),
            ("pool", pool, POOL_MODES),
            ("readout", readout, READOUT_MODES),
            ("field", field, FIELD_SOURCES),
        ):
            if val not in allowed:
                raise ValueError(f"field_mil: {name}={val!r} not in {allowed}")
        self.attn, self.pool, self.readout, self.field = attn, pool, readout, field
        self.alpha = float(alpha)
        self.data_root = Path(data_root or os.environ.get("MPN_DATA_ROOT", "data"))
        self.hidden_dim = int(hidden_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        if attn == "film":
            # zero-init -> gamma=0, beta=0 -> attention identical to the fieldless baseline
            self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        elif attn in ("query", "query2", "queryk"):
            self.n_query = int(heads) if attn == "queryk" else 1
            self.to_query = nn.Linear(hidden_dim, hidden_dim * self.n_query)
            if field == "ownpat":
                self.to_query_ctx = nn.Linear(hidden_dim, hidden_dim * self.n_query)
            if attn == "query2":
                # second-round query: the first pooled descriptor refines the field query
                self.refine = nn.Linear(hidden_dim, hidden_dim)
                nn.init.zeros_(self.refine.weight)
                nn.init.zeros_(self.refine.bias)

        z_dim = hidden_dim * (self.n_query if attn == "queryk" else 1)
        self.classifier = nn.Linear(z_dim, num_classes)
        if readout == "concat":
            # a separate zero-initialised head for the field half, so the sum starts at C(z)
            self.field_head = nn.Linear(hidden_dim, num_classes, bias=False)
            nn.init.zeros_(self.field_head.weight)
        elif readout == "gate":
            self.field_head = nn.Linear(hidden_dim, num_classes)
            self.gate = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 4.0)   # sigmoid(4) ~ 0.982 -> starts as patch-only

        if field == "static":
            self.static_src = nn.Parameter(torch.randn(input_dim) * 0.02)

        self._bank_obj: Optional[_FieldBank] = None
        self._miss = 0
        self._hit = 0
        self._warned = False

    # ── field vector for one bag ─────────────────────────────────────────
    def _field(self, features: torch.Tensor
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (query source, patient-context source or None)."""
        if self.field == "mean":
            return features.mean(dim=0), None
        if self.field == "static":
            # One LEARNED query shared by every bag: dot-product attention with NO conditioning
            # at all. Separates "query attention instead of gated attention" from "the query
            # depends on the bag".
            return self.static_src, None
        if self.field == "random":
            # A fixed pseudo-random query per bag, scale-matched to the bag but carrying no
            # information whatsoever. Separates "a query that is not the bag centroid" from
            # "a query derived from the field view".
            key = _fingerprint(features)
            gen = torch.Generator().manual_seed(abs(hash(key)) % (2 ** 31))
            m = features.mean(dim=0)
            r = torch.randn(features.shape[1], generator=gen)
            r = r / r.norm().clamp(min=1e-6) * m.norm().to("cpu")
            return r.to(features.device, features.dtype), None
        if self._bank_obj is None:
            self._bank_obj = _bank(int(features.shape[1]), self.data_root)
        bank = self._bank_obj
        key = _fingerprint(features)
        if self.field == "shuffle":
            # deterministic pseudo-random *other* ROI: stable across epochs and at eval
            idx = abs(hash(key)) % bank.stack.shape[0]
            self._hit += 1
            return bank.stack[idx].to(features.device, features.dtype), None
        vec = bank.table.get(key)
        if vec is not None and self.field in ("patient", "ownpat"):
            ctx = bank.context(key)
            if ctx is None:                       # single-ROI patient: fall back to its own field
                ctx = vec
            ctx = ctx.to(features.device, features.dtype)
            self._hit += 1
            if self.field == "patient":
                return ctx, None
            return vec.to(features.device, features.dtype), ctx
        if vec is None:
            self._miss += 1
            if not self._warned:
                self._warned = True
                print(
                    "  ⚠ field_mil: bag not found in the field bank — falling back to the "
                    "patch mean. This is expected only if a feature-space augmentation is "
                    "active (it perturbs the fingerprint); otherwise the bank is stale."
                )
            return features.mean(dim=0), None
        self._hit += 1
        vec = vec.to(features.device, features.dtype)
        if self.field == "resid":
            # Remove everything the bag already knows: project the field view off the patch
            # mean. Whatever is left is information no patch carried, so a gain here cannot
            # be explained by "the query is just a bag-adaptive vector" (the a347 control).
            m = features.mean(dim=0)
            mhat = m / m.norm().clamp(min=1e-6)
            vec = vec - (vec @ mhat) * mhat
        return vec, None

    def field_hit_rate(self) -> float:
        n = self._hit + self._miss
        return float(self._hit) / n if n else 0.0

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, features, return_attention=False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        g, gctx = self._field(features)                            # [D], [D] or None
        h = self.bottleneck(features)                              # [N, H]
        q = self.bottleneck(g.unsqueeze(0)).squeeze(0)             # [H]  (shared weights)

        if self.attn in ("query", "query2", "queryk"):
            qv = self.to_query(q)
            if gctx is not None:
                qc = self.bottleneck(gctx.unsqueeze(0)).squeeze(0)
                qv = qv + self.to_query_ctx(qc)
            scale = math.sqrt(self.hidden_dim)
            if self.attn == "queryk":
                Q = qv.view(self.n_query, self.hidden_dim)             # [K, H]
                E = (h @ Q.t()) / scale                                # [N, K]
                cols = [torch.softmax(E[:, k], dim=0) if self.pool == "softmax"
                        else entmax_bisect(E[:, k], self.alpha) for k in range(self.n_query)]
                A = torch.stack(cols, dim=1)                           # [N, K]
                z = (h.t() @ A).t().reshape(-1)                        # [K*H]
                y = self.classifier(z)
                if self.readout == "concat":
                    y = y + self.field_head(q)
                elif self.readout == "gate":
                    s = torch.sigmoid(self.gate(q))
                    y = s * y + (1.0 - s) * self.field_head(q)
                a = A.mean(dim=1)
                return (y.view(-1), a, None) if return_attention else (y.view(-1), None, None)
            e = (h @ qv) / scale
            if self.attn == "query2":
                a1 = torch.softmax(e, dim=0) if self.pool == "softmax" else entmax_bisect(e, self.alpha)
                e = (h @ (qv + self.refine(torch.mv(h.t(), a1)))) / scale
        else:
            v = self.attention_V(h)
            u = self.attention_U(h)
            if self.attn == "film":
                gamma, beta = self.film(q).chunk(2, dim=-1)
                v = (1.0 + gamma) * v + beta
            e = self.attention_W(v * u).squeeze(-1)                # [N]

        a = torch.softmax(e, dim=0) if self.pool == "softmax" else entmax_bisect(e, self.alpha)
        z = torch.mv(h.t(), a)                                     # [H]

        y = self.classifier(z)
        if self.readout == "concat":
            y = y + self.field_head(q)
        elif self.readout == "gate":
            s = torch.sigmoid(self.gate(q))
            y = s * y + (1.0 - s) * self.field_head(q)
        y = y.view(-1)

        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, attn="film", pool="entmax",
              readout="concat", field="roi")
