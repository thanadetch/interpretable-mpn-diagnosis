"""Build a FUSED reticulin feature dir by concatenating per-patch embeddings
from two (or more) frozen backbones, WITHOUT touching the trainer.

The trainer builds its feature path as ``{data_root}/{cfg['feature_dir']}``
(src/train_grading_reti.py:857) and forces ``input_dim`` from the backbone
config (L1017). So we write the fused bags into a parallel data root using the
SAME relative layout and the SAME ``feature_dir`` name as the primary backbone:

    data_fused/features_virchow2_reti/<Class>/.../<slide>.pt   (feats = [N, D1+D2])

Then run:  --backbone virchow2 --data_root data_fused
and use a Model that self-detects / overrides the feature dim (a68/a69), since
the trainer will still force input_dim=1280.

Patch correspondence is verified per-slide (same N, and same `rc` grid when
present). Any mismatch aborts that slide with a loud warning (never silently
truncates).

Usage (defaults fuse virchow2 + uni2 -> data_fused):
    python scripts/fuse_features.py
    python scripts/fuse_features.py --backbones virchow2 uni2 titan --out_root data_fused3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

# feature_dir names mirror BACKBONE_CONFIG in src/train_grading_reti.py
FEATURE_DIRS = {
    "virchow2": "features_virchow2_reti",
    "uni2": "features_uni2_reti",
    "titan": "features_titan_reti",
}


def _feats(d) -> torch.Tensor:
    return d["feats"] if isinstance(d, dict) else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument(
        "--backbones",
        nargs="+",
        default=["virchow2", "uni2"],
        help="Backbones to concatenate (order = concat order). First one is the layout/template.",
    )
    ap.add_argument(
        "--out_root",
        default="data_fused",
        help="Fused bags go to {out_root}/{feature_dir-of-first-backbone}/<same rel paths>.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only process first N slides (debug).")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    primary = args.backbones[0]
    src_dirs = [data_root / FEATURE_DIRS[b] for b in args.backbones]
    for b, sd in zip(args.backbones, src_dirs):
        if not sd.is_dir():
            raise SystemExit(f"missing feature dir for {b}: {sd}")
    out_dir = Path(args.out_root) / FEATURE_DIRS[primary]
    out_dir.mkdir(parents=True, exist_ok=True)

    template = src_dirs[0]
    pt_files = sorted(template.rglob("*.pt"))
    if args.limit:
        pt_files = pt_files[: args.limit]
    print(f"Fusing {args.backbones} -> {out_dir}   ({len(pt_files)} slides from {primary})")

    ok, skipped = 0, 0
    dims_seen = set()
    for i, pt in enumerate(pt_files):
        rel = pt.relative_to(template)
        try:
            d0 = torch.load(pt, map_location="cpu", weights_only=False)
            f0 = _feats(d0)
            N = f0.shape[0]
            parts = [f0]
            mismatch = False
            for sd in src_dirs[1:]:
                other = sd / rel
                if not other.is_file():
                    print(f"  [skip] no match in {sd.name}: {rel}")
                    mismatch = True
                    break
                dk = torch.load(other, map_location="cpu", weights_only=False)
                fk = _feats(dk)
                if fk.shape[0] != N:
                    print(f"  [skip] N mismatch {rel}: {primary}={N} {sd.name}={fk.shape[0]}")
                    mismatch = True
                    break
                # verify patch order via rc grid when both expose it
                if isinstance(d0, dict) and isinstance(dk, dict) and "rc" in d0 and "rc" in dk:
                    if not torch.equal(d0["rc"], dk["rc"]):
                        print(f"  [skip] rc (patch order) mismatch: {rel}")
                        mismatch = True
                        break
                parts.append(fk)
            if mismatch:
                skipped += 1
                continue

            fused = torch.cat([p.float() for p in parts], dim=1)  # [N, sum(D)]
            dims_seen.add(fused.shape[1])
            out = {"feats": fused}
            if isinstance(d0, dict):
                for k in ("rc", "metrics", "patch_paths"):
                    if k in d0:
                        out[k] = d0[k]
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out, dest)
            ok += 1
            if (i + 1) % 200 == 0:
                print(f"  ...{i + 1}/{len(pt_files)}  (fused dim={fused.shape[1]})")
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {rel}: {e}")
            skipped += 1

    print(f"DONE: wrote {ok} fused bags, skipped {skipped}. fused dims seen = {sorted(dims_seen)}")
    print(f"Run with:  --backbone {primary} --data_root {args.out_root}")


if __name__ == "__main__":
    main()
