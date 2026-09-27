"""a261 - Patch ISAB (Induced-point Self-Attention Block, Set Transformer, Lee 2019) with a PMA
(Pooling-by-Multihead-Attention) readout. NEW family: induced-point low-rank SELF-attention AMONG
patches mediated by a small set of learned inducing points, then a learned-query pooling readout.

WHY THIS IS A CATEGORICALLY-DISTINCT FAMILY (vs the exhausted lanes and vs a259).
  - The exhausted first-moment / attention-shape lane (softmax/entmax/temperature/sparsity/James-Stein/
    variance-reg/MAD/content-gate/mean-blend) is ONE relevance-weighted mean of the patches: there is no
    interaction BETWEEN patches, only a re-weighting of each patch independently. a261 is fundamentally
    different: it performs patch<->patch interaction. Inducing points first AGGREGATE information from all
    patches (I attends to X), then the patches are RE-EXPRESSED as a low-rank combination of those
    information-carrying inducing points (X attends back to I). So each patch's updated representation
    depends on the WHOLE bag through the M-dimensional inducing bottleneck -- a context-mixing step that a
    single weighted mean cannot express.
  - vs a259 (plain multi-query cross-attention readout / Perceiver): a259 only does a fixed set of learned
    QUERIES reading the patches once (queries -> patches), i.e. a pooling readout with no patch<->patch
    mediation. ISAB is distinct: the inducing points are an intermediate low-rank channel through which the
    patches interact with EACH OTHER (X -> I -> X), and only THEN is a PMA readout applied. The defining new
    mechanism here is the I-mediated patch<->patch self-attention (steps 3-4), absent from a259.

MECHANISM (matmul + softmax only; no [N,N] full attention is needed, all attentions are [M,N] or [N,M]
or [M,M]; M is small so this is genuinely low-rank).
  1. Bottleneck:  X = Dropout(ReLU(Linear(input_dim, H)))(features)                         -> [N, H]
  2. Inducing points  I in R^{M x H} = nn.Parameter, init randn * H**-0.5  (M small)
  3. I attends to patches (ISAB half 1, "MAB(I, X)"):
        A_IX = softmax( (I @ X^T) / sqrt(H), dim=1 )                                         -> [M, N]
        I_h  = A_IX @ X                                                                       -> [M, H]
        I1   = LayerNorm( I + FF(I_h) ),  FF = Linear(H,H) -> ReLU -> Linear(H,H)             -> [M, H]
     (I1 are the inducing points after absorbing bag-wide context.)
  4. patches attend back to I1 (ISAB half 2, "MAB(X, I1)") -> low-rank patch<->patch self-attention:
        A_XI = softmax( (X @ I1^T) / sqrt(H), dim=1 )                                         -> [N, M]
        X_I  = A_XI @ I1                                                                       -> [N, H]
        X1   = LayerNorm( X + X_I )                                                            -> [N, H]
     (each patch is now re-expressed via the M context-carrying inducing points; the bag has interacted
      with itself through the M-dim bottleneck. This is the ISAB output set.)
  5. PMA readout (Set-Transformer pooling-by-attention with learned seed queries S in R^{M x H}):
        A_SX = softmax( (S @ X1^T) / sqrt(H), dim=1 )                                         -> [M, N]
        z    = A_SX @ X1                                                                       -> [M, H]
        z    = LayerNorm( z + FF2(z) )                                                         -> [M, H]
     bag = z.reshape(-1)  (concat over the FIXED M seed rows -> a richer fixed-size descriptor)        [M*H]
  6. y = Linear(M*H, num_classes)(bag).view(-1)                                              -> (1,)

HONEST NOTES / APPROXIMATIONS.
  - Single-head attention throughout (no multi-head split). Multi-head was omitted on purpose to keep
    capacity modest for the 214-ROI cohort; it is an honest simplification of the original ISAB/PMA, which
    use multi-head attention. The inducing/seed sets (M rows) provide the analogous expressivity here.
  - No explicit per-head value/key/query projections beyond the bottleneck: I, I1, X, X1 share the same
    H-dim space (queries=keys=values are the raw H-dim vectors, scaled-dot-product directly). This is the
    "tied projection" low-capacity variant; it trades the standard separate W_q/W_k/W_v matrices for fewer
    parameters. Stated plainly as a deliberate capacity choice, not an oversight.
  - LayerNorm (not the standard MAB rFF+residual with separate norms per the paper's exact ordering) is
    used for the residual blocks; ordering is residual-then-norm (post-norm). This is a standard, MPS-safe
    stabiliser and does not change the family.
  - Concept-free: no zero-shot text prompts, no bone/fibrosis labels. NOT norm-weighted: ||h|| is never
    used as a relevance signal; all weights come from learned scaled-dot-product attention.
  - Capacity is deliberately modest: H=64 bottleneck, M=4 inducing/seed points. The learnable content lives
    in VECTORS/MATRICES (I [M,H], S [M,H], the FF/FF2 linears, the bottleneck, the head) -- there is NO lone
    learnable shape-scalar, so the a215/a237/a238 inert-scalar failure mode is avoided.

CONSTRAINTS satisfied. Self-contained nn.Module; torch / torch.nn / torch.nn.functional only; no external
files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, LayerNorm, softmax, matmul, add, reshape --
NO linalg.solve, NO linalg.eigh, NO linalg.svd, NO cdist, NO torch.median.
Permutation-invariant: patches enter ONLY through softmax-over-patches attentions ((I@X^T) and (S@X1^T)
softmax along the patch axis, then @X / @X1) and through (X@I1^T) which produces one independent row per
patch; reordering patches reorders the [N,...] intermediates identically and leaves the [M,H] inducing/
seed aggregates (hence z, bag, y) unchanged. The final bag is a concat over the FIXED M seed rows, never
over the variable N patches.
Bag-size-invariant: EVERY aggregation over patches is a softmax-weighted convex combination (attention
rows sum to 1), never a raw sum, never /N, never /(N-1), never std.
Deterministic in eval(): only Dropout is stochastic and it is disabled in eval(); LayerNorm uses running-
free per-sample statistics (deterministic). All other ops deterministic.
n=1 SAFE: X=[1,H]; A_IX=softmax((I@X^T)=[M,1], dim=1)=[M,1] (all 1.0) -> I_h=[M,H]; A_XI=softmax((X@I1^T)=
[1,M], dim=1)=[1,M] -> X_I=[1,H]; A_SX=softmax((S@X1^T)=[M,1], dim=1)=[M,1] -> z=[M,H]; bag=[M*H] well
defined. No div-by-zero, no std, no /(N-1). LayerNorm over the H feature dim is per-row and fine for N=1.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=64, M=4, dropout=0.5):
        super().__init__()
        assert M <= 8, "keep inducing/seed point count modest for this small cohort"
        self.H = hidden_dim
        self.M = M
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # 1. bottleneck input_dim -> H
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

        # 2. learned inducing points I [M,H] and PMA seed queries S [M,H]
        self.inducing = nn.Parameter(torch.randn(M, hidden_dim) * (hidden_dim ** -0.5))  # I
        self.seeds = nn.Parameter(torch.randn(M, hidden_dim) * (hidden_dim ** -0.5))     # S (PMA)

        # 3. FF block applied to the inducing points after they absorb bag context
        self.ff_ind = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.ln_ind = nn.LayerNorm(hidden_dim)

        # 4. LayerNorm for the patch-side ISAB output
        self.ln_x = nn.LayerNorm(hidden_dim)

        # 5. FF block + LayerNorm for the PMA readout
        self.ff_pma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.ln_pma = nn.LayerNorm(hidden_dim)

        # 6. head reads the concatenated M seed rows
        self.classifier = nn.Linear(M * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        X = self.bottleneck(features)                                   # [N,H]

        # --- ISAB half 1: inducing points attend to patches (I -> X) ---
        A_IX = F.softmax((self.inducing @ X.t()) * self.scale, dim=1)   # [M,N] softmax over patches
        I_h = A_IX @ X                                                  # [M,H] bag-wide context per inducing pt
        I1 = self.ln_ind(self.inducing + self.ff_ind(I_h))             # [M,H] context-carrying inducing pts

        # --- ISAB half 2: patches attend back to I1 (X -> I1) low-rank patch<->patch self-attn ---
        A_XI = F.softmax((X @ I1.t()) * self.scale, dim=1)              # [N,M] softmax over inducing pts
        X_I = A_XI @ I1                                                 # [N,H] each patch re-expressed via I1
        X1 = self.ln_x(X + X_I)                                         # [N,H] ISAB output set

        # --- PMA readout: learned seed queries pool the ISAB output set (S -> X1) ---
        A_SX = F.softmax((self.seeds @ X1.t()) * self.scale, dim=1)     # [M,N] softmax over patches
        z = A_SX @ X1                                                   # [M,H] pooled per seed
        z = self.ln_pma(z + self.ff_pma(z))                            # [M,H]

        bag = z.reshape(-1)                                            # [M*H] concat over FIXED M seeds
        y = self.classifier(bag).view(-1)                             # (1,)

        if return_attention:
            # per-patch relevance proxy: mean PMA attention mass over the M seed rows -> [N]
            return y, A_SX.mean(dim=0), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
