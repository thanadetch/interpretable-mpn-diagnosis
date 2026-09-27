"""a96x — a96 (warm fibrosis-axis diffuse projection, free-affine readout), made
BACKBONE-AGNOSTIC for the seed-2 cross-backbone comparison.

a96 hardcodes the Virchow2 seed-2 prototype path, so on UNI2/TITAN its axis dim
mismatches and it silently falls back to a RANDOM projection (not the warm
fibrosis-axis one). a96x picks the matching seed-2 train-only prototype file by
input_dim so the SAME mechanism runs on virchow2 (1280) / uni2 (1536) / titan
(768). Mechanism is otherwise byte-identical to a96: diffuse pool z=mean_i f_i ->
scalar s=<z, v> (v warm-started at the backbone's seed-2 train fibrosis axis) ->
free affine head y=w_s*s+c_s. forward reads ONLY 'features'. RAW logit.
"""
from __future__ import annotations

from pathlib import Path

from .a95_a93_prototype_barycentre import Model as _Base  # noqa: F401

_DATA = Path(__file__).resolve().parents[3] / "data"
_PROTO_BY_DIM = {
    1280: "prototypes_virchow2_reti_train_seed2.pt",
    1536: "prototypes_uni2_reti_train_seed2.pt",
    768: "prototypes_titan_reti_train_seed2.pt",
}


class Model(_Base):
    def __init__(self, input_dim: int = 1280, **kw) -> None:
        kw.pop("prototype_path", None)
        fname = _PROTO_BY_DIM.get(int(input_dim))
        path = str(_DATA / fname) if fname else None
        super().__init__(input_dim=input_dim, prototype_path=path, **kw)


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    num_grades=4,
    rho_max=0.5,
    tau_init=4.0,
    use_anchors=False,    # a96 ablation = free affine readout on the diffuse projection
    warm_start=True,
    clamp_output=False,
)
