"""a120 — bone-suppressing / fibrosis-weighting gated-attention MIL (USER idea).

Pathology prior: grade = diffuse fibrosis density; AVOID dark bone trabeculae.
The faithfulness test showed the baseline gated-attention does NOT avoid bone
(Spearman(attn, bone)=+0.30, same as fibrosis +0.32) — a real faithfulness gap.

a120 = the baseline ABMIL + TWO interpretable scalars that nudge the
attention toward fibrosis and away from BONE-SPECIFIC patches. The per-patch
semantic bone & fibrosis scores (TITAN zero-shot concepts, patch-index aligned)
are carried as the trailing 2 dims of the 1282-d input (built offline; point the
trainer at --data_root data_bonefib --backbone virchow2). Because bone and
fibrosis scores correlate ~0.79, we suppress only the bone-SPECIFIC residual
relu(bone_z - fib_z) so fibrosis evidence is preserved.

    attn_logits_i = gated_score_i + softplus(w_f)*fib_z_i - softplus(w_b)*relu(bone_z_i - fib_z_i)

Only +2 learnable params over the baseline (197,252 total) -> stays at the U-shape
capacity optimum; the active ingredient is the bone/fibrosis attention bias.
Ablation a121 (suppress=False) -> exactly the baseline gated attention.
forward reads ONLY 'features'. RAW logit. Deterministic at inference.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Global per-patch score stats (all 1330 reti bags) — fixed normalisation constants
# (TITAN zero-shot concept cosines; NOT label-derived).
_BONE_MEAN, _BONE_STD = -0.0486, 0.0193
_FIB_MEAN, _FIB_STD = 0.0045, 0.0281


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        suppress: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = input_dim          # virchow2 features = first `input_dim` cols
        self.suppress = suppress           # trailing 2 cols = [bone, fibrosis] scores
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # two interpretable scalars: fibrosis boost, bone-specific suppression (>=0 via softplus)
        self.w_fib = nn.Parameter(torch.tensor(0.0))
        self.w_bone = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        gated = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)  # [N]
        if self.suppress and features.shape[1] >= D + 2:
            bone = (features[:, D] - _BONE_MEAN) / _BONE_STD
            fib = (features[:, D + 1] - _FIB_MEAN) / _FIB_STD
            bone_specific = F.relu(bone - fib)           # bone beyond fibrosis
            logit = gated + F.softplus(self.w_fib) * fib - F.softplus(self.w_bone) * bone_specific
        else:
            logit = gated
        attn = F.softmax(logit, dim=0)
        agg = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        out = self.classifier(agg)
        if return_attention:
            return out, attn, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, suppress=True)
