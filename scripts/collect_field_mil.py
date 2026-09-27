#!/usr/bin/env python3
"""Collect the Field-Conditioned MIL (a340-a351) results next to their seed-2 baselines.

Usage:
    python scripts/collect_field_mil.py [--date 20260808] [--grades]

Prints one row per (backbone, method) with val/test QWK, macro recall and per-grade recall,
plus the dual-gate verdict against the matching fieldless baseline (ASGAP for the entmax
variants, ABMIL for the softmax ones).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

# Locked seed-2, no-augmentation, mps reference runs (see PROJECT_STATUS.md).
BASELINE = {
    ("titan", "entmax"): ("ASGAP", 0.7968, 0.9600, 87.1),
    ("virchow2", "entmax"): ("ASGAP", 0.7976, 0.9588, 87.2),
    ("uni2", "entmax"): ("ASGAP", 0.7703, 0.9262, 72.3),
    ("titan", "softmax"): ("ABMIL", 0.7902, 0.9584, 82.9),
    ("virchow2", "softmax"): ("ABMIL", 0.8182, 0.9476, 81.9),
    ("uni2", "softmax"): ("ABMIL", 0.7743, 0.9418, 79.8),
}

POOL_OF = {  # which baseline each variant must be judged against
    "a349": "softmax", "a350": "softmax", "a351": "softmax",
}


def rows(date: str | None):
    out = []
    pattern = f"{date}/*/config.json" if date else "*/*/config.json"
    for cf in EXP.glob(pattern):
        try:
            c = json.load(open(cf))
        except Exception:
            continue
        nid = c.get("novelty_id") or ""
        if not nid.startswith(("a34", "a35")):
            continue
        d = cf.parent
        try:
            v = json.load(open(d / "val_metrics.json"))
            t = json.load(open(d / "test_metrics.json"))
        except Exception:
            continue
        out.append((c["backbone"], nid, v["val_qwk"], t["test_qwk"],
                    t["test_macro_recall"], t["test_recall_per_class"], d.name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--grades", action="store_true")
    a = ap.parse_args()

    data = rows(a.date)
    if not data:
        print("no FC-MIL runs found")
        return

    order = {"virchow2": 0, "uni2": 1, "titan": 2}
    data.sort(key=lambda r: (order.get(r[0], 9), r[1]))

    hdr = f"{'backbone':9s} {'method':28s} {'val':>7s} {'test':>7s} {'mRec':>6s}  gate"
    if a.grades:
        hdr += "   G0    G1    G2    G3"
    print(hdr)
    print("-" * len(hdr))

    seen = set()
    for bb, nid, val, test, mrec, per, run in data:
        pool = POOL_OF.get(nid[:4], "entmax")
        bname, bval, btest, bmrec = BASELINE[(bb, pool)]
        if (bb, pool) not in seen:
            seen.add((bb, pool))
            line = f"{bb:9s} {bname + ' (baseline)':28s} {bval:7.4f} {btest:7.4f} {bmrec:6.1f}  --"
            print(line)
        gate = ("PASS" if val > bval and test > btest else
                "val✓" if val > bval else "test✓" if test > btest else "✗")
        line = f"{bb:9s} {nid:28s} {val:7.4f} {test:7.4f} {mrec:6.1f}  {gate}"
        if a.grades:
            line += "".join(f" {per[g]:5.1f}" for g in ("G0", "G1", "G2", "G3"))
        print(line)


if __name__ == "__main__":
    main()
