"""a134 — ablation of a133: Virchow2-ONLY slice of the fused bags (backbones=('virchow2',)).
Same gated-MIL head, same data_fused input, but ignores the UNI2-h + TITAN columns.
Reproduces the single-backbone baseline on data_fused so a133 vs a134 isolates the
fusion (adding UNI2-h + TITAN) as the ONLY difference.
"""
from __future__ import annotations
from .a133_fusion_concat_gated import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
              backbones=("virchow2",))
