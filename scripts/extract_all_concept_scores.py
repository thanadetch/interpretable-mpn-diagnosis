"""Extract PER-PATCH scores for ALL pathology concepts (for the faithfulness audit).

Generalises extract_patch_concept_scores.py to every concept group (cellularity,
open_adipose, fibrosis_stroma, bone_trabecular, megakaryocyte_*). Per-patch score
= mean cosine over the group's prompts (TITAN zero-shot). Patch-index aligned to
the Virchow2 bags (same grid).

Output: data/patch_concept_scores_all_reti.pt
    { "<Class>/<Patient>/<ImgID>.pt": {group: Tensor[N], ..., "n": int}, ... }
"""
from __future__ import annotations
import glob
from collections import defaultdict
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
from tools.concept_probe_titan import CONCEPT_PROMPTS, encode_concepts  # noqa: E402

TITAN_DIR = ROOT / "data" / "features_titan_reti"
OUT = ROOT / "data" / "patch_concept_scores_all_reti.pt"


def load_feats(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        for k in ("feats", "features", "x"):
            if k in d and torch.is_tensor(d[k]):
                return d[k].float()
        for v in d.values():
            if torch.is_tensor(v) and v.dim() == 2:
                return v.float()
    return torch.as_tensor(d).float()


def main():
    prompts = [c["text"] for c in CONCEPT_PROMPTS]
    groups = [c["group"] for c in CONCEPT_PROMPTS]
    gidx = defaultdict(list)
    for i, g in enumerate(groups):
        gidx[g].append(i)
    print("groups:", {g: len(ix) for g, ix in gidx.items()})

    tf = encode_concepts(prompts, torch.device("cpu"))  # [C,768] normalised
    files = sorted(glob.glob(str(TITAN_DIR / "*" / "*" / "*.pt")))
    print(f"[extract] {len(files)} bags")
    out = {}
    for i, f in enumerate(files):
        feats = load_feats(f)
        sim = torch.nn.functional.normalize(feats, dim=-1) @ tf.T  # [N,C]
        rec = {g: sim[:, ix].mean(dim=1).contiguous() for g, ix in gidx.items()}
        rec["n"] = feats.shape[0]
        out["/".join(Path(f).parts[-3:])] = rec
        if (i + 1) % 300 == 0 or i + 1 == len(files):
            print(f"  {i+1}/{len(files)}")
    torch.save(out, OUT)
    print(f"[extract] saved -> {OUT}")
    # per-concept global mean + per-patch cross-correlations
    cats = list(gidx.keys())
    stacks = {g: torch.cat([v[g] for v in out.values()]) for g in cats}
    print("\nper-concept global mean (per-patch):")
    for g in cats:
        print(f"  {g:22s}: mean {stacks[g].mean():+.4f} std {stacks[g].std():.4f}")
    print("\nper-patch correlation with fibrosis_stroma (entanglement check):")
    fib = stacks["fibrosis_stroma"]
    for g in cats:
        if g == "fibrosis_stroma":
            continue
        c = torch.corrcoef(torch.stack([stacks[g], fib]))[0, 1]
        print(f"  corr({g:22s}, fibrosis) = {c:+.3f}")


if __name__ == "__main__":
    main()
