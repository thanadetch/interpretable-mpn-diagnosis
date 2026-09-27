"""
Per-grade, per-split bag statistics for the reticulin fibrosis task.

Reports:
  - Bag count, patch count distribution (min / mean / median / max) per grade × split.
  - Scale-bar (µm) distribution per grade × split, joined from
    ``results/scalebar_results.csv``.

Run from repo root:
    python scripts/bag_stats_reti.py [--backbone virchow2|uni2|titan]

Writes a Markdown summary to ``results/reports/bag_stats_reti_<backbone>.md``
and prints it to stdout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from train_grading_reti import patient_split  # noqa: E402

GRADE_NAMES = {0: "G0", 1: "G1", 2: "G2", 3: "G3"}


def load_scale_map(csv_path: Path) -> dict[tuple[str, str], str]:
    """Returns {(patient_folder, roi_stem): scalebar_micron_str}."""
    out: dict[tuple[str, str], str] = {}
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            patient = row["patient"]
            roi_stem = Path(row["filename"]).stem  # reti10.tif -> reti10
            out[(patient, roi_stem)] = (row.get("scalebar_micron") or "unknown").strip() or "unknown"
    return out


def fmt_dist(values: list[int]) -> str:
    if not values:
        return "—"
    return f"n={len(values):3d}  min={min(values):3d}  med={int(median(values)):3d}  mean={mean(values):5.1f}  max={max(values):3d}  sum={sum(values):5d}"


def fmt_scale_dist(scales: list[str]) -> str:
    if not scales:
        return "—"
    cnt: dict[str, int] = defaultdict(int)
    for s in scales:
        cnt[s] += 1
    total = len(scales)
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}µm: {v} ({100*v/total:4.1f}%)" for k, v in items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="virchow2", choices=["virchow2", "uni2", "titan"])
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    feat_root = REPO / "data" / f"features_{args.backbone}_reti"
    if not feat_root.exists():
        raise SystemExit(f"Feature dir not found: {feat_root}")

    ds = GradingBagDatasetFull(feat_root)
    train_idx, val_idx, test_idx = patient_split(ds, seed=args.seed)
    split_of: dict[int, str] = {}
    for i in train_idx:
        split_of[i] = "train"
    for i in val_idx:
        split_of[i] = "val"
    for i in test_idx:
        split_of[i] = "test"

    scale_map = load_scale_map(REPO / "results" / "scalebar_results.csv")

    # Aggregators: {(split, grade): list of patch counts}
    patches: dict[tuple[str, int], list[int]] = defaultdict(list)
    scales: dict[tuple[str, int], list[str]] = defaultdict(list)
    # Patient bookkeeping for sanity
    patients_per_split_grade: dict[tuple[str, int], set[str]] = defaultdict(set)

    for idx, (pt_path, label) in enumerate(ds.samples):
        split = split_of.get(idx, "unassigned")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        feats = data["feats"] if isinstance(data, dict) else data
        n_patches = int(feats.shape[0])
        patient = pt_path.parent.name
        roi_stem = pt_path.stem
        scale = scale_map.get((patient, roi_stem), "unknown")

        patches[(split, label)].append(n_patches)
        scales[(split, label)].append(scale)
        patients_per_split_grade[(split, label)].add(patient)

    # ── Report ─────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# Reticulin bag stats — backbone={args.backbone}, seed={args.seed}")
    lines.append("")
    lines.append(f"- Feature root: `{feat_root.relative_to(REPO)}`")
    lines.append(f"- Total bags: {len(ds.samples)}")
    lines.append(f"- Train/Val/Test bag counts: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    lines.append("")

    # Patch-count table per split × grade
    lines.append("## Patch count per bag (split × grade)")
    lines.append("")
    lines.append("| split | grade | patients | bags | min | med | mean | max | total patches |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for split in ["train", "val", "test"]:
        for g in [0, 1, 2, 3]:
            v = patches[(split, g)]
            pts = len(patients_per_split_grade[(split, g)])
            if not v:
                lines.append(f"| {split} | {GRADE_NAMES[g]} | {pts} | 0 | — | — | — | — | — |")
                continue
            lines.append(
                f"| {split} | {GRADE_NAMES[g]} | {pts} | {len(v)} | {min(v)} | "
                f"{int(median(v))} | {mean(v):.1f} | {max(v)} | {sum(v)} |"
            )
    lines.append("")

    # Scale distribution per split × grade
    lines.append("## Scale-bar (µm) distribution (split × grade)")
    lines.append("")
    lines.append("| split | grade | bags | distribution |")
    lines.append("|---|---|---:|---|")
    for split in ["train", "val", "test"]:
        for g in [0, 1, 2, 3]:
            s = scales[(split, g)]
            if not s:
                lines.append(f"| {split} | {GRADE_NAMES[g]} | 0 | — |")
                continue
            lines.append(f"| {split} | {GRADE_NAMES[g]} | {len(s)} | {fmt_scale_dist(s)} |")
    lines.append("")

    # Cross-cut: overall scale distribution per grade (all splits)
    lines.append("## Scale-bar (µm) distribution per grade (all splits combined)")
    lines.append("")
    lines.append("| grade | bags | distribution |")
    lines.append("|---|---:|---|")
    for g in [0, 1, 2, 3]:
        s_all = scales[("train", g)] + scales[("val", g)] + scales[("test", g)]
        lines.append(f"| {GRADE_NAMES[g]} | {len(s_all)} | {fmt_scale_dist(s_all)} |")
    lines.append("")

    # Cross-cut: patch count per grade (all splits)
    lines.append("## Patch count per grade (all splits combined)")
    lines.append("")
    lines.append("| grade | bags | min | med | mean | max | total patches |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for g in [0, 1, 2, 3]:
        v_all = patches[("train", g)] + patches[("val", g)] + patches[("test", g)]
        if not v_all:
            continue
        lines.append(
            f"| {GRADE_NAMES[g]} | {len(v_all)} | {min(v_all)} | "
            f"{int(median(v_all))} | {mean(v_all):.1f} | {max(v_all)} | {sum(v_all)} |"
        )
    lines.append("")

    report = "\n".join(lines)
    out_path = REPO / "results" / "reports" / f"bag_stats_reti_{args.backbone}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(report)
    print(f"\n[saved] {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

