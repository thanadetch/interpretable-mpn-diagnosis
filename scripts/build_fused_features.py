"""Build an offline FUSED feature dir = per-patch concat of all 3 frozen FMs.

Each output bag has feats = [N, 3584] = [virchow2 1280 | uni2 1536 | titan 768],
patch-index aligned (all three backbones share the EXACT patch grid AND order —
verified: 1330 bags each, per-ROI patch counts match; order checked via patch_paths
when available, else trusted from the deterministic shared preprocessing).

Layout offsets (a fusion Model hardcodes these):
    virchow2 = feats[:, 0:1280]
    uni2     = feats[:, 1280:2816]
    titan    = feats[:, 2816:3584]

Point the trainer at it with --data_root data_fused --backbone virchow2; the trainer
passes input_dim=1280 (virchow2 config) regardless, so a fusion Model must infer the
true width (3584) from features.shape[1] and lazy-init its first layer.

No training. No label use. Mirrors data/features_virchow2_reti/ layout exactly.
"""
from __future__ import annotations
import glob
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
DIRS = {
    "virchow2": ROOT / "data" / "features_virchow2_reti",
    "uni2": ROOT / "data" / "features_uni2_reti",
    "titan": ROOT / "data" / "features_titan_reti",
}
ORDER = ["virchow2", "uni2", "titan"]  # concat order -> [1280 | 1536 | 768] = 3584
OUT = ROOT / "data_fused" / "features_virchow2_reti"


def load_bag(p: Path):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        feats = d["feats"].float() if "feats" in d else None
        if feats is None:
            for v in d.values():
                if torch.is_tensor(v) and v.dim() == 2:
                    feats = v.float(); break
        pp = d.get("patch_paths") or d.get("paths")
        return feats, pp, d
    return torch.as_tensor(d).float(), None, {}


def main():
    files = sorted(glob.glob(str(DIRS["virchow2"] / "*" / "*" / "*.pt")))
    print(f"[build-fused] {len(files)} virchow2 bags -> {OUT}")
    n_ok = n_skip = 0
    order_checked = order_match = 0
    dim_seen = None
    for f in files:
        rel = "/".join(Path(f).parts[-3:])
        bags = {}
        ok = True
        for bb in ORDER:
            p = DIRS[bb] / rel
            if not p.exists():
                ok = False; break
            bags[bb] = load_bag(p)
        if not ok:
            n_skip += 1; continue
        feats = {bb: bags[bb][0] for bb in ORDER}
        ns = {bb: feats[bb].shape[0] for bb in ORDER}
        if len(set(ns.values())) != 1:
            n_skip += 1; continue
        # order check via patch_paths when all three carry them
        pps = {bb: bags[bb][1] for bb in ORDER}
        if all(pps[bb] is not None and len(pps[bb]) == ns["virchow2"] for bb in ORDER):
            order_checked += 1
            same = all(list(pps[bb]) == list(pps["virchow2"]) for bb in ORDER)
            order_match += int(same)
            if not same:
                n_skip += 1; continue  # refuse to fuse misaligned patches
        fused = torch.cat([feats[bb] for bb in ORDER], dim=1)  # [N, 3584]
        dim_seen = fused.shape[1]
        out_path = OUT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base = bags["virchow2"][2]
        nd = dict(base) if isinstance(base, dict) else {}
        nd["feats"] = fused
        nd["fusion_offsets"] = {"virchow2": [0, 1280], "uni2": [1280, 2816], "titan": [2816, 3584]}
        torch.save(nd, out_path)
        n_ok += 1
        if n_ok % 300 == 0:
            print(f"  {n_ok} written ...")
    print(f"[build-fused] wrote {n_ok} bags, skipped {n_skip}.")
    print(f"  patch-order verified via patch_paths on {order_checked} bags; "
          f"{order_match}/{order_checked} matched"
          + ("" if order_checked else " (no patch_paths stored; trusting shared deterministic preprocessing)"))
    if dim_seen is not None:
        print(f"  fused feats width = {dim_seen}  (expect 3584)")
    # sanity re-load
    outs = sorted(glob.glob(str(OUT / "*" / "*" / "*.pt")))
    print(f"  re-load check: {len(outs)} fused bags on disk")
    if outs:
        ex = torch.load(outs[0], map_location="cpu", weights_only=False)
        print(f"  example feats shape = {tuple(ex['feats'].shape)}  offsets = {ex.get('fusion_offsets')}")


if __name__ == "__main__":
    main()
