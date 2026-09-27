#!/usr/bin/env python3
"""Recover the true physical scale (um per pixel) of every reticulin ROI.

`read_scalebar.py` OCRs the LABEL ("20 um", "50 um", ...) but never measures the bar, so the
project knows how the scale bar is annotated and not how much tissue a pixel covers. Those are
different questions: the acquisition software picks a bar length to suit the field, so two ROIs
labelled 100 um and 200 um can render the SAME bar in pixels, which would mean their um/px differ
by exactly 2x. Without um/px it is impossible to say whether this cohort really mixes
magnifications -- and therefore whether patches cut at a fixed 224 px cover comparable tissue.

The annotation is a white box in the bottom-right corner holding a dark horizontal bar with end
ticks, and the label underneath. Detection:

  1. crop the bottom-right corner
  2. the longest dark horizontal run is the box's top border; the matching run ~20-40 rows below
     is its bottom border
  3. inside the box, above the text, the longest dark run is the BAR
  4. um/px = labelled micron / bar length in pixels

Every measurement is kept with its bar length so the result can be audited; a clean detector
should make um/px fall into a few tight clusters, one per objective.

    python -m src.tools.measure_scalebar_px --out results/scale_um_per_px.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DARK = 110


def longest_run(row: np.ndarray) -> int:
    if not row.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
    return int((idx[1::2] - idx[::2]).max())


def measure(path: Path) -> tuple[float, float, float]:
    """(bar length px, box height px, box border width px); nan when the box is not found.

    The annotation box has two long horizontal borders 15-60 rows apart, the bar sitting just
    below the top one and the label text below that. An earlier version assumed the LONGEST dark
    run in the crop was the top border, which is wrong whenever the label text runs into the
    bottom border and makes that row longer -- the search for a border below then falls off the
    end of the crop and the ROI is discarded. That single mistake accounted for 201 of the 233
    failures on ROIs that do carry a readable label.

    So instead of trusting the argmax, score every candidate (top, bottom) pair and keep the best
    one. A pair only qualifies if the borders are comparably long, are 15-60 rows apart, and
    enclose a bar of 30 px or more that does not exceed its own box -- the same guards as before,
    since darkly stained marrow produces long dark runs of its own.
    """
    with Image.open(path) as im:
        a = np.array(im.convert("L"))
    h, w = a.shape
    c = a[int(h * 0.86):, int(w * 0.55):] < DARK
    L = np.array([longest_run(r) for r in c])
    if L.max() < 20:
        return float("nan"), float("nan"), float("nan")

    best = None
    for top in range(len(L)):
        if L[top] < 30:                                      # too short to be a box border
            continue
        for bot in range(top + 15, min(top + 61, len(L))):
            if L[bot] < 0.85 * L[top]:                       # borders must be comparably long
                continue
            inner = L[top + 2: top + 2 + max(1, (bot - top) // 2)]   # the bar sits above the label
            if not len(inner):
                continue
            bar = float(inner.max())
            if not (30 <= bar <= L[top]):                    # a bar cannot exceed its own box
                continue
            score = L[top] + L[bot]                          # prefer the longest, best-matched pair
            if best is None or score > best[0]:
                best = (score, bar, float(bot - top), float(L[top]))
    if best is None:
        return float("nan"), float("nan"), float("nan")
    return best[1], best[2], best[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/scalebar_results.csv")
    ap.add_argument("--out", default="results/scale_um_per_px.csv")
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N (debugging)")
    a = ap.parse_args()

    d = pd.read_csv(ROOT / a.csv)
    d["stem"] = d.filename.str.replace(r"\.tif$", "", regex=True)
    raw = {}
    for p in glob.glob(str(ROOT / "data/raw/**/*.tif"), recursive=True):
        P = Path(p)
        raw[(P.parent.name, P.stem)] = p
    d["path"] = [raw.get((p, s)) for p, s in zip(d.patient, d.stem)]
    if a.limit:
        d = d.head(a.limit)

    rows = []
    for r in d.itertuples():
        if r.path is None:
            continue
        bar, bh, bw = measure(Path(r.path))
        with Image.open(r.path) as im:
            W, H = im.size
        um = float(r.scalebar_micron) if str(r.scalebar_micron) != "unknown" else float("nan")
        rows.append(dict(patient=r.patient, stem=r.stem, grade=r.grade,
                         label_um=r.scalebar_micron, bar_px=bar, box_h=bh, box_w=bw,
                         img_w=W, img_h=H, um_per_px=um / bar if bar and bar == bar else float("nan")))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / a.out, index=False)

    k = out[out.um_per_px.notna()]
    print(f"\nmeasured {len(k)} of {len(out)} ROIs\n")
    print("bar length in pixels, by printed label")
    print(k.groupby("label_um").bar_px.agg(["count", "median", "min", "max"]).to_string(), "\n")
    print("um per pixel, by printed label   <- if these differ, the cohort really mixes scales")
    print(k.groupby("label_um").um_per_px.agg(["count", "median", "std", "min", "max"])
          .round(4).to_string(), "\n")
    v = k.um_per_px.round(2).value_counts().head(10)
    print("most common um/px values (a clean detector gives a few tight clusters)")
    print(v.to_string())
    print(f"\nfull range {k.um_per_px.min():.3f} - {k.um_per_px.max():.3f} um/px "
          f"= {k.um_per_px.max() / k.um_per_px.min():.1f}x")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
