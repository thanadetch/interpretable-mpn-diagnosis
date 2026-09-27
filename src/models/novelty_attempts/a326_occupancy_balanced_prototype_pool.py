"""a326 - Occupancy-BALANCED prototype feature pool. The one prototype/occupancy variant NOT on disk.

WHAT IS GENUINELY NEW (vs the existing prototype family, verified on disk):
  * a116 (SimplexOccupancyPool): reads the bag's occupancy distribution over FROZEN grade prototypes
    through fixed monotone anchors -> a grade-expectation scalar. It never builds a pooled FEATURE vector.
  * a264 (monotone-prototype soft-assign): descriptor = occupancy-WEIGHTED prototype mixture
    z = sum_k w_k * P_k  (the dominant tissue mode still dominates, by construction).
  * a256 / a183 (NetVLAD): per-cluster SUM-OF-RESIDUALS descriptor.
  None of them do an explicit MODE-BALANCED average of per-cluster FEATURE means, i.e. down-weighting
  whichever tissue region simply happens to contain the most patches. That exact mechanism is the gap.

WHY IT MATCHES THE GRADING PRINCIPLE.
  The pathologist reads the OVERALL/holistic reticulin density across the WHOLE marrow space. A plain
  attention/mean pool (and a264's occupancy-weighted mixture) lets the single largest tissue region drive
  the descriptor purely because it has more patches. a326 soft-assigns patches to K learnable prototypes
  (feature-space tissue modes), takes the per-cluster feature MEAN, then averages the cluster-means with a
  CONCAVE occupancy weight w_k ~ occupancy_k^p (p=0.5): p=1 -> plain weighted mean (mode-imbalanced),
  p=0 -> fully balanced (every occupied mode counts equally). p=0.5 partially de-biases the dominant mode
  toward a whole-marrow density read.

HONEST DISCLOSURE (user's never-stop "หา novelty ต่อ / คิดมาให้ได้", seed-2 exploratory fish).
  - Family-adjacent to a116/a264/a256; distinct ONLY in the occupancy-balanced FEATURE-mean readout.
  - HONEST PRIOR: very likely a TIE. The cohort's val<->test QWK anti-correlation (Pearson -0.76..-0.98 on
    10 val / 10 test patients) caps single-seed gains; ~40 session candidates + 336 disk modules all tie/
    fail. A seed-2 pass here would NOT be a win until it survives a multi-seed audit + a clean reproduction.
  - KNOWN FLAW (disclosed): p<1 also up-weights RARE modes, some of which are bone/artefact junk (anti-
    grading). p=0.5 is the compromise; bone-suppression is deliberately NOT added (keep one new variable).
  - ZERO learnable SHAPE params: p and the cosine temperature tau are FIXED (learnable shape -- alpha/
    lambda/temperature -- was proven inert x3 on this cohort). Prototypes are the only added params.
  - No ||h|| salience (cosine assign is scale-free). Self-contained, permutation/size-invariant,
    deterministic at eval, MPS-safe, no new deps.

z = sum_k w_k m_k ;  m_k = (sum_i a_ik h_i)/(sum_i a_ik) ;  a_ik = softmax_k(<h_i,c_k>/tau) ;
w_k = occ_k^p / sum_j occ_j^p ;  occ_k = mean_i a_ik .
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
                 num_proto=8, tau=0.5, balance_p=0.5):
        super().__init__()
        self.tau = float(tau)
        self.balance_p = float(balance_p)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # K learnable tissue-mode prototypes in bottleneck space.
        self.prototypes = nn.Parameter(torch.randn(num_proto, hidden_dim) * 0.02)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                          # [N,H]
        N = h.shape[0]
        hn = h / (h.norm(dim=1, keepdim=True) + 1e-6)                          # unit dirs (scale-free assign)
        cn = self.prototypes / (self.prototypes.norm(dim=1, keepdim=True) + 1e-6)
        logits = (hn @ cn.t()) / self.tau                                      # [N,K] cosine / tau
        a = F.softmax(logits, dim=1)                                           # [N,K] rows sum to 1
        occ = a.mean(0)                                                        # [K] occupancy (sums to 1)
        denom = a.sum(0).clamp(min=1e-6)                                       # [K]
        m = (a.t() @ h) / denom.unsqueeze(1)                                   # [K,H] per-cluster feature mean
        w = occ.clamp(min=1e-8) ** self.balance_p                              # concave occupancy de-bias
        w = w / w.sum().clamp(min=1e-8)                                        # [K] balanced cluster weights
        z = w @ m                                                              # [H] mode-balanced descriptor
        y = self.classifier(z).view(-1)
        if return_attention:
            # report per-patch responsibility to the most-occupied cluster as a pseudo-attention
            return y, a.max(dim=1).values, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
