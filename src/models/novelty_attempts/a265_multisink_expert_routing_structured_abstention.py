"""a265 - Multi-sink expert routing with structured abstention. NEW family: ATTENTION-SINK /
LEARNED-NULL-TOKEN pooling (register/null tokens that compete in the SAME softmax as real patches
so the pool can dump probability mass onto NOTHING). This is NOT an attention-shape/temperature/
sparsity/de-concentration tweak: those families (entmax a215, size-temp a238, James-Stein a248,
entropy-gated rank-cap a252, variance-reg a254, robust-MAD a255, content-gate a243, mean-blend a237)
all RENORMALISE all probability mass onto the real patches and only reshape HOW that full unit mass is
distributed. a265 instead lets a learned softmax LEAK mass off the real bag entirely.

WHY THIS IS A CATEGORICALLY-DISTINCT MECHANISM. The de-concentration lane keeps sum_i a_i = 1 over the N
real patches and argues about the SHAPE of that simplex. a265 appends K learned SINK EMBEDDINGS that live
in the same 128-d space as the patches and join the SAME joint softmax. Real patches and sinks therefore
compete jointly; the softmax can assign arbitrary probability to the sinks, so the mass that lands on the
N real patches, p_real = a_aug[:N], can SUM TO LESS THAN ONE. The kept bag vector z = h^T p_real is built
ONLY from real-patch mass and is deliberately NOT renormalised: mass routed to a sink is simply DROPPED.
This is "structured abstention" -- the pool can down-weight the entire bag when no patch is relevant,
rather than being forced to pick a relatively-best patch. Distinct from a260 capsule routing (no null
output capsule there) and from a263 MoE gating (which routed among real experts and collapsed; here the
"experts" being competed for are NULL absorbers, not output heads).

WHY MULTIPLE SINKS + A GATE (the genuinely-new content, in VECTORS not a lone scalar). A single sink is a
lone scalar knob and would go inert like a215/a237/a238. Instead a265 uses K=4 distinct sink embeddings
(K*128 learnable params, each a direction in feature space that absorbs a different kind of irrelevant
patch) and reads the SINK-MASS DISTRIBUTION p_sinks in R^K as a data-driven per-bag descriptor. A small
MLP turns p_sinks into a scalar ignore_w = sigmoid(MLP(p_sinks)) that gates the kept vector:
z <- z * ignore_w. The gate's learnable content is the MLP WEIGHT MATRICES reading a K-vector, never a
single free shape-scalar, so the absorb-the-bag behaviour is driven by which sinks fired, not by an inert
global temperature. The two sink mechanisms are complementary: (a) the joint softmax already lets the
KEPT vector shrink in magnitude when much mass leaks to sinks (z = h^T p_real has small p_real); (b) the
gate lets the model further, NONLINEARLY, suppress z as a function of the PATTERN of which sinks absorbed
the mass.

MECHANISM (matmul / softmax / sigmoid only; no [N,K,D] tensor, no MPS-forbidden op).
  1. h = bottleneck(features) = Dropout(ReLU(Linear(input_dim, 128)))                       -> [N,128]
  2. sink_embed = nn.Parameter(randn(K,128) * 0.01)  (K=4 learned null embeddings)          -> [K,128]
  3. h_aug = cat([h, sink_embed], dim=0)                                                    -> [N+K,128]
  4. gated-attention score over ALL rows (Ilse 2018 gated attention head):
        e_aug = attn_W( tanh(attn_V(h_aug)) * sigmoid(attn_U(h_aug)) ).squeeze(-1)          -> [N+K]
  5. a_aug = softmax(e_aug)  (real patches AND the K sinks compete in ONE joint softmax)    -> [N+K]
  6. split  p_real = a_aug[:N]  [N],   p_sinks = a_aug[N:]  [K]   (p_real may sum < 1)
  7. KEPT bag vector from real-patch mass ONLY, NOT renormalised:  z = h^T @ p_real         -> [128]
  8. per-bag abstention gate from the sink-mass pattern:  ignore_w = sigmoid(MLP(p_sinks))  -> scalar
        z <- z * ignore_w
  9. y = classifier(z).view(-1),  classifier = Linear(128, num_classes)                     -> (1,)

HONEST NOTES / APPROXIMATIONS / RISKS.
  - The sink mechanism only CHANGES THE SCALE/PRESENCE of the bag vector; it does NOT add a new direction.
    The kept z is still a weighted mean of real patch embeddings (a first-moment pool), so the novelty is
    purely in the routing/abstention, not in capturing higher-order distribution shape. If the val<->test
    QWK anti-correlation is cohort-driven (as ~284 priors suggest), nothing about abstention is guaranteed
    to break it -- this is completeness-coverage of the attention-sink family, not a promised winner.
  - Because the gate ignore_w in (0,1) multiplies z and the FINAL classifier is linear, a constant gate
    is partly absorbable into the classifier bias/scale; the gate only earns its keep when it varies
    ACROSS bags as a function of p_sinks. It is fed a K-vector through an MLP (real matrices), so it is not
    a lone inert scalar, but it could still learn to stay near-constant -- that would just degrade it to a
    plain leaky-softmax gated-attention pool, which is the conservative fallback, not a collapse.
  - The K sinks could all collapse to absorbing near-zero mass (then p_real ~ unit-sum and a265 reduces to
    ordinary gated attention). This is a benign degeneration, not the MoE-style collapse (a263) where a
    single expert dominated and accuracy fell to chance -- here every real patch is always pooled.
  - sink_embed init is small (*0.01) so at init the sinks sit near the origin and absorb little mass; the
    model STARTS close to a standard gated-attention pool and learns to leak mass only if useful.
  - Concept-free: no zero-shot text prompts, no bone/fibrosis labels. NOT norm-weighted: ||h|| is never
    used as a relevance signal; all weights come from the learned gated-attention head over h_aug.

CONSTRAINTS satisfied. Self-contained single nn.Module; torch / torch.nn / torch.nn.functional only; no
external files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, tanh, sigmoid, softmax, matmul,
cat, elementwise mul -- NO linalg.solve/eigh/svd, NO cdist, NO torch.median, NO eig/solve.
Permutation-invariant over real patches: a_aug is a softmax over per-row scores (each row scored
independently), and z = h^T p_real sums per-patch contributions, so reordering the N patches permutes
both h and p_real identically and leaves z, p_sinks, ignore_w, y unchanged; the K sinks are always the
last K rows (fixed order, independent of patch order). Bag-size-invariant: there is NO division by N, NO
std, NO /(N-1); the joint softmax self-normalises over N+K rows, and the kept-mass leak naturally adapts
to bag size. Deterministic in eval() (Dropout off; softmax/sigmoid/matmul deterministic). n=1 SAFE: with
one real patch h_aug is [1+K,128], softmax over 1+K rows is well-defined, z = h^T p_real is a single
scaled patch vector, no div-by-zero (softmax denominator >= K sink terms > 0), no /N anywhere. Capacity is
modest: bottleneck (input_dim*128), gated-attention V/U/W heads (128*att + 128*att + att), K*128 sinks
(K=4 -> 512), a tiny gate MLP (K->K->1), classifier (128->1) -- well under 197K with input_dim=1280
(dominant term ~1280*128 = 164K). All learnable content is in VECTORS/matrices/heads; no lone learnable
shape-scalar.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, att_dim=64,
                 dropout=0.5, K=4):
        super().__init__()
        assert K >= 1, "need at least one sink"
        self.K = K
        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # K learned SINK / null embeddings living in the same 128-d space as patches.
        # Small init => start near a standard gated-attention pool; sinks absorb little mass at first.
        self.sink_embed = nn.Parameter(torch.randn(K, hidden_dim) * 0.01)   # [K,128]
        # Ilse-style gated attention head, scored over real patches AND sinks jointly.
        self.attn_V = nn.Linear(hidden_dim, att_dim)
        self.attn_U = nn.Linear(hidden_dim, att_dim)
        self.attn_W = nn.Linear(att_dim, 1)
        # Per-bag abstention gate: reads the K-dim sink-mass distribution -> scalar in (0,1).
        # Learnable content is in MATRICES (K->K->1), not a lone scalar.
        self.gate = nn.Sequential(
            nn.Linear(K, K), nn.ReLU(inplace=True), nn.Linear(K, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                   # [N,128]
        N = h.shape[0]
        h_aug = torch.cat([h, self.sink_embed], dim=0)                  # [N+K,128] patches + sinks
        # gated-attention scoring over ALL rows (real patches + sinks).
        e_aug = self.attn_W(torch.tanh(self.attn_V(h_aug)) *
                            torch.sigmoid(self.attn_U(h_aug))).squeeze(-1)   # [N+K]
        a_aug = F.softmax(e_aug, dim=0)                                 # [N+K] ONE joint softmax
        p_real = a_aug[:N]                                              # [N]  may sum < 1
        p_sinks = a_aug[N:]                                            # [K]  mass dumped onto NOTHING
        # KEPT bag vector from real-patch mass ONLY; mass on sinks is DROPPED (NOT renormalised).
        z = h.t() @ p_real                                             # [128]
        # data-driven per-bag abstention gate from the sink-mass pattern.
        ignore_w = torch.sigmoid(self.gate(p_sinks)).view(())          # scalar in (0,1)
        z = z * ignore_w                                               # soft bag down-weighting
        y = self.classifier(z).view(-1)                                # shape (1,)
        if return_attention:
            return y, p_real, None                                     # real-patch attention (un-renormalised)
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
