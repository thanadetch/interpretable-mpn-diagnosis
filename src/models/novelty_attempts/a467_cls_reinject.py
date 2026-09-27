"""a467 — a445, with the whole-ROI vector re-anchored onto the readout BETWEEN the two attention rounds.

WHY (measured, not assumed)
    a445 puts the ROI's global view at position 0 once, before any attention, and reads position 0
    at the end. Extracting the readout's attention from the trained a445 checkpoints shows it is
    markedly MORE concentrated after layer 1 than after layer 2:

        effective-N / N   layer 1   layer 2
        virchow2            .515      .828
        uni2                .630      .896
        titan               .261      .345

    The second round spreads a selection the first round already made — i.e. the field-initialised
    readout drifts back toward a uniform average of the patches, losing the anchor that makes a445
    different from plain TransMIL. This adds the field vector back on before layer 2:

        seq = layer1(seq)
        seq[:, 0] += g * fvec          <- g is a single scalar, initialised to 0
        seq = layer2(seq)

    At g=0 the model is bit-identical to a445, so any gain has to be learned. ONE extra parameter.

NOT a449. a449 places the field at two POSITIONS of the input sequence (duplication in space);
this places it at the same position at two TIMES (re-anchoring in depth). a449 lost 3/3.

PREDICTION, STATED BEFORE THE RUN: it loses. Fourteen variants around a445 have now been tried and
none beat it consistently, and every attempt to combine mechanisms in this project has been at best
equal to its parts (a448, a449, a461, and the augmentation stacks).

KILL CONDITION: if the learned `g` converges to ~0, the model does not want re-anchoring and the
dilution hypothesis is dead regardless of QWK — the same way learnable alpha converged to 1.506 and
a466's group weights stayed at 1/3. Report `g` alongside the metrics; it is the informative output.

Gate: val QWK AND test QWK above a445 on the same backbone.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", reinject=True)
