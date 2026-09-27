"""a465 - a445 whose whole-ROI readout comes from TITAN while the patches come from another encoder.

a445 uses ONE encoder for two different jobs: embedding 224 px crops at native resolution, and
embedding the whole ROI downsampled to 224 px. The second job is out-of-distribution for a patch
encoder. TITAN is a slide-level encoder -- summarising a field is what it was built for -- and its
whole-ROI vectors are already extracted (`features_titan_reti_no_patch`, 1330 ROIs, same paths).

So: patches from --backbone (virchow2 / uni2), readout token from TITAN, paired by ROI path.

CONFOUND, STATED FIRST: the TITAN vector is 768-d and the patches are 1280/1536-d, so the field
needs its own projection. That is exactly a456 (separate projection), which lost 2/3. A win here
must therefore beat a456 as well as a445; a loss cannot separate "wrong encoder" from "extra
projection". Report against both.

PREDICTION: no prior in this project covers cross-encoder fusion, so none is offered. This is the
one untried axis that is about WHAT the readout is made of, rather than where it goes.

Gate: val QWK AND test QWK above a445 on the same patch backbone. titan+titan is a445 itself.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", field_backbone="titan")
