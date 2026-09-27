"""a07 — length-normalised soft-mean (parameter-free temperature, H4).

Hint targeted (§7 of NOVELTY_NOTES.md):
    H4_length_normalised_softmean. Mechanism: replace the gated-attention
    softmax with a *length-normalised* softmax over a single attention
    score, with temperature  tau = c / sqrt(N)  where c is a learnable
    scalar and N is the bag size. Addresses bag-size sensitivity (BH3):
    larger bags get FLATTER attention (less peaky), smaller bags get
    sharper attention, so the effective number of contributing patches
    stays roughly constant across bag sizes. Re-uses the ABMIL
    bottleneck + V/U/W attention scorer; only the softmax temperature
    changes.

Why H4 may help after H6 failed:
    - DE12 showed that bounded heads break gradient flow under
      SmoothL1Loss. H4 keeps the head architecture identical to baseline
      (single Linear(128->1) classifier), so gradient flow is unchanged.
    - The pathology motivation is direct: G0 bags routinely have 60+
      patches (mean 53.4, max 112) while G1/G2/G3 cap at ~48
      (NOVELTY_NOTES §5.5). Without normalisation, softmax attention on
      a 100-patch G0 bag concentrates all weight on 1-2 outlier patches,
      which is exactly the "G0 looks like G1 because the model fixates
      on the one fibrous-looking patch" failure pattern.

Mechanism:
    h_i        = bottleneck(features)                            # same as baseline
    z_i        = W(V(h_i) ⊙ U(h_i)).squeeze(-1)                  # raw attention logit
    tau        = c / sqrt(N), with c = softplus(c_raw)           # 1 learnable scalar
    alpha_i    = softmax(z_i * tau)                              # length-normalised
    bag        = sum_i alpha_i * h_i
    y          = clamp(Linear(bag), 0, 3)

Initialisation: c_raw is set so that softplus(c_raw) ~= sqrt(40), i.e.
tau at the dataset's median bag size N=40 is roughly 1.0 -- making the
initial attention distribution identical to the baseline's. Training is
then free to scale the temperature up or down.

Ablation companion: a08_constant_temperature.py -- identical wiring but
    tau = softplus(c_raw) (no sqrt(N) normalisation). If a07 beats a08,
    the bag-size adaptive scaling is the active ingredient. If both
    fail, the family is dead.

Kill criterion: abandon if a07 val_qwk < 0.81 at seed=2.

Param count: baseline 197,250 + 1 scalar (c_raw) = 197,251.
"""
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        clamp_output: bool = True,
        # tau = c / sqrt(N); we want tau ~ 1.0 at the median bag size N=40,
        # so c_init = sqrt(40) approx 6.324.
        c_init: float = 6.324555,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 1 learnable scalar; c = softplus(c_raw) keeps c > 0.
        self.c_raw = nn.Parameter(torch.tensor(_inv_softplus(c_init)))

    def _temperature(self, n: int) -> torch.Tensor:
        """tau = c / sqrt(N) -- length-normalised."""
        c = F.softplus(self.c_raw)
        return c / math.sqrt(max(n, 1))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)  # [N, hidden]
        n = h.shape[0]

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_logits = self.attention_W(V * U).squeeze(-1)         # [N]

        tau = self._temperature(n)
        attention = F.softmax(attn_logits * tau, dim=0)            # [N]

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)
        y = self.classifier(bag)

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attention, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    c_init=6.324555,
)

