"""a100 — Ablation of a99: the leading centered-graph SPECTRAL mode turned OFF.

NULL companion to a99_spectral_lowfreq. It imports a99's `Model` UNCHANGED and
flips ONLY the documented flag `gamma_fixed=0.0`. With gamma_fixed=0.0 the model
registers a constant `gamma_const=0` buffer instead of a learnable `gamma_logit`
parameter, and the forward short-circuits the entire spectral branch
(use_spec = (gamma_fixed != 0.0) and n>1 => False), so

    h_i = Dropout(ReLU(Linear(1280->128) f_i))
    z   = mean_i h_i                 # plain DIFFUSE mean-pool, NO spectral mode
    y   = Linear(128 -> 1)(z)        # raw grade logit (same warm-started head)

which is EXACTLY the diffuse mean-pool + linear head with z = z_dc only. The
encoder, the head, the warm-start along the train-only (seed=2) fibrosis axis, and
dropout=0.5 are all byte-identical (a100 only flips the flag). This removes EXACTLY
the active ingredient of a99 — the blend z = z_dc + gamma * z_spec of the leading
NON-TRIVIAL centered-graph spectral mode — and NOTHING else.

a99 (gamma learnable, init 0.1) vs a100 (gamma_fixed=0.0) therefore isolates the
single question: "does adding the leading centered-graph spectral mode beat the
diffuse mean?" If a99 <= a100 on the paired cross-fold audit, the spectral mode adds
nothing over the diffuse pool (the honest null this session keeps surfacing).

Param count: a100 has 164,097 trainable (the gamma_logit scalar becomes a frozen
gamma_const buffer => one fewer trainable param than a99's 164,098). The spectral
readout machinery is never invoked, so it contributes zero params either way.
"""
from __future__ import annotations

from .a99_spectral_lowfreq import Model  # noqa: F401  (re-exported, unchanged)

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    power_iters=8,         # inert when gamma_fixed=0.0 (spectral branch skipped)
    gamma_init=0.1,        # inert when gamma_fixed is set (no gamma_logit created)
    gamma_fixed=0.0,       # <-- THE ONLY CHANGE vs a99: spectral mode OFF => z = z_dc
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
