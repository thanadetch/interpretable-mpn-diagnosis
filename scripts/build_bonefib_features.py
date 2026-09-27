"""Build an offline Virchow2 feature dir with per-patch bone+fibrosis scores appended.

Each output bag has feats = [N, 1282] = [virchow2 1280 | bone_score | fibrosis_score],
patch-index aligned (TITAN and Virchow2 share the exact patch order — verified).
Point the trainer at it with --data_root data_bonefib --backbone virchow2; the
a120 module slices the trailing 2 dims as the bone/fibrosis signal.

No training. No label use. Mirrors data/features_virchow2_reti/ layout.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "features_virchow2_reti"
SCORES = ROOT / "data" / "patch_concept_scores_reti.pt"
OUT = ROOT / "data_bonefib" / "features_virchow2_reti"


def main():
    scores = torch.load(SCORES, map_location="cpu", weights_only=False)
    files = sorted(glob.glob(str(SRC / "*" / "*" / "*.pt")))
    print(f"[build] {len(files)} virchow2 bags -> {OUT}")
    n_ok = n_skip = 0
    for f in files:
        rel = "/".join(Path(f).parts[-3:])
        d = torch.load(f, map_location="cpu", weights_only=False)
        feats = d["feats"].float() if isinstance(d, dict) else torch.as_tensor(d).float()
        if rel not in scores:
            n_skip += 1; continue
        bone = scores[rel]["bone"].float().view(-1, 1)
        fib = scores[rel]["fibrosis"].float().view(-1, 1)
        if feats.shape[0] != bone.shape[0]:
            n_skip += 1; continue
        aug = torch.cat([feats, bone, fib], dim=1)  # [N, 1282]
        out_path = OUT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nd = dict(d) if isinstance(d, dict) else {}
        nd["feats"] = aug
        torch.save(nd, out_path)
        n_ok += 1
    print(f"[build] wrote {n_ok} bags, skipped {n_skip}. example dim check:")
    ex = torch.load(sorted(glob.glob(str(OUT / "*" / "*" / "*.pt")))[0], map_location="cpu", weights_only=False)
    print(f"  feats shape = {tuple(ex['feats'].shape)}  (expect [N,1282])")


if __name__ == "__main__":
    main()
