"""a352_patient_query — Field-Conditioned MIL variant.

PATIENT CONTEXT AS QUERY. The attention query comes from the LEAVE-ONE-OUT mean field view
of the SAME patient's OTHER ROIs — never this ROI's own field. Motivated by a measured
property of the cohort: within-patient prediction spread is 44% of the between-patient
spread, while the label and the clinical decision are per patient. Transductive at the
patient level (images only, never labels); see field_mil.py.

    attn='query'  pool='entmax'  readout='concat'  field='patient'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="query",
    pool="entmax",
    readout="concat",
    field="patient",
)
