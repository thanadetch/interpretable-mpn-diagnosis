"""Extract PER-PATCH semantic bone + fibrosis scores for every reticulin ROI.

Reuses the TITAN/CONCH zero-shot concept probe (concept_probe_titan.py): encode
the 'bone_trabecular' and 'fibrosis_stroma' text concepts, then for each
features_titan_reti bag compute the per-patch cosine similarity and average
within each concept group -> one bone score and one fibrosis score per patch.

Because features_titan_reti and features_virchow2_reti share the SAME patch grid
(identical N per ROI), these per-patch scores align index-wise onto the Virchow2
bags and can be used by a bone-suppressing / fibrosis-weighting aggregator OR to
test whether the baseline gated attention already avoids bone (faithfulness).

Output: data/patch_concept_scores_reti.pt
    { "<Class>/<Patient>/<ImgID>.pt": {"bone": Tensor[N], "fibrosis": Tensor[N], "n": int}, ... }
No training. Read-only over frozen features. TITAN text encoder is cached locally.
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tools.concept_probe_titan import CONCEPT_PROMPTS, encode_concepts  # noqa: E402

TITAN_DIR = ROOT / "data" / "features_titan_reti"
OUT = ROOT / "data" / "patch_concept_scores_reti.pt"
GROUPS = ["bone_trabecular", "fibrosis_stroma"]


def load_feats(pt_path):
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        for k in ("feats", "features", "x"):
            if k in d and torch.is_tensor(d[k]):
                return d[k].float()
        # fall back to first tensor value
        for v in d.values():
            if torch.is_tensor(v) and v.dim() == 2:
                return v.float()
        raise ValueError(f"no feature tensor in {pt_path}")
    return torch.as_tensor(d).float()


def main():
    # which prompt indices belong to each group
    prompts = [c["text"] for c in CONCEPT_PROMPTS]
    groups = [c["group"] for c in CONCEPT_PROMPTS]
    group_idx = {g: [i for i, gg in enumerate(groups) if gg == g] for g in GROUPS}
    for g in GROUPS:
        assert group_idx[g], f"no prompts for group {g}"
        print(f"  group {g}: {[prompts[i] for i in group_idx[g]]}")

    device = torch.device("cpu")
    text_features = encode_concepts(prompts, device)  # [C,768] normalised

    files = sorted(glob.glob(str(TITAN_DIR / "*" / "*" / "*.pt")))
    print(f"[extract] {len(files)} TITAN reti bags")
    out = {}
    for i, f in enumerate(files):
        feats = load_feats(f)  # [N,768]
        fn = torch.nn.functional.normalize(feats, dim=-1)
        sim = fn @ text_features.T  # [N,C] cosine
        rec = {}
        for g in GROUPS:
            rec["bone" if g == "bone_trabecular" else "fibrosis"] = sim[:, group_idx[g]].mean(dim=1).contiguous()
        rec["n"] = feats.shape[0]
        rel = "/".join(Path(f).parts[-3:])
        out[rel] = rec
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f"  {i+1}/{len(files)}")
    torch.save(out, OUT)
    # quick summary
    allb = torch.cat([v["bone"] for v in out.values()])
    allf = torch.cat([v["fibrosis"] for v in out.values()])
    print(f"[extract] saved -> {OUT}")
    print(f"  bone score:     mean {allb.mean():.4f} std {allb.std():.4f} min {allb.min():.4f} max {allb.max():.4f}")
    print(f"  fibrosis score: mean {allf.mean():.4f} std {allf.std():.4f} min {allf.min():.4f} max {allf.max():.4f}")
    print(f"  corr(bone,fibrosis) per-patch = {torch.corrcoef(torch.stack([allb,allf]))[0,1]:.3f}")


if __name__ == "__main__":
    main()
