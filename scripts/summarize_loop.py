#!/usr/bin/env python3
"""Build a sorted leaderboard CSV/Markdown summarising the loop_runner sweep."""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
EXP = ROOT / "experiments"
RESULTS = ROOT / "results"

TARGET_VAL = 0.813
TARGET_TEST = 0.908

rows = []

# Baseline
b = EXP / "reti_mean_pool_uni2_s2_20260506_232602"
v = json.loads((b / "val_metrics.json").read_text())
t = json.loads((b / "test_metrics.json").read_text())
rows.append(
    {
        "model": "BASELINE_mean_pool",
        "novelty_id": "—",
        "val_qwk": v["val_qwk"],
        "test_qwk": t["test_qwk"],
        "val_acc": v["val_accuracy"],
        "test_acc": t["test_accuracy"],
        "val_macroRec": v["val_macro_recall"],
        "test_macroRec": t["test_macro_recall"],
        "exp_dir": str(b.relative_to(ROOT)),
    }
)

# Novelty attempts
for d in sorted(EXP.glob("reti_novelty_attempt_uni2_a*_2026*")):
    vp = d / "val_metrics.json"
    tp = d / "test_metrics.json"
    if not (vp.exists() and tp.exists()):
        continue
    v = json.loads(vp.read_text())
    t = json.loads(tp.read_text())
    cfg = json.loads((d / "config.json").read_text())
    nid = cfg.get("novelty_id", "?")
    rows.append(
        {
            "model": nid,
            "novelty_id": nid,
            "val_qwk": v["val_qwk"],
            "test_qwk": t["test_qwk"],
            "val_acc": v["val_accuracy"],
            "test_acc": t["test_accuracy"],
            "val_macroRec": v["val_macro_recall"],
            "test_macroRec": t["test_macro_recall"],
            "exp_dir": str(d.relative_to(ROOT)),
        }
    )

# Sort by val_qwk desc, but keep baseline first
baseline_row = rows[0]
attempts = sorted(rows[1:], key=lambda r: -r["val_qwk"])
final_rows = [baseline_row] + attempts

# CSV
RESULTS.mkdir(exist_ok=True)
with open(RESULTS / "loop_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
    w.writeheader()
    for r in final_rows:
        w.writerow(r)

# Pretty print
hdr = (
    f"{'model':<27}{'val_qwk':>10}{'test_qwk':>10}"
    f"{'val_acc':>9}{'test_acc':>10}{'v>0.813':>10}{'t>0.908':>10}"
)
print(hdr)
print("-" * len(hdr))
for r in final_rows:
    flag_v = "YES" if r["val_qwk"] > TARGET_VAL else "no"
    flag_t = "YES" if r["test_qwk"] > TARGET_TEST else "no"
    star = "  ⭐" if (flag_v == "YES" and flag_t == "YES") else ""
    print(
        f"{r['model']:<27}{r['val_qwk']:>10.4f}{r['test_qwk']:>10.4f}"
        f"{r['val_acc']:>9.2f}{r['test_acc']:>10.2f}{flag_v:>10}{flag_t:>10}{star}"
    )

print(f"\nSaved -> {RESULTS / 'loop_summary.csv'}")
beat = [r for r in attempts if r["val_qwk"] > TARGET_VAL and r["test_qwk"] > TARGET_TEST]
print(f"\nNumber of attempts that beat baseline on BOTH metrics: {len(beat)}")
print("\nVal-only winners (val_qwk > 0.813):")
for r in [x for x in attempts if x["val_qwk"] > TARGET_VAL]:
    print(f"  {r['model']:<25} val={r['val_qwk']:.4f}  test={r['test_qwk']:.4f}")
print("\nTest-closest novelties (sorted by test_qwk desc):")
for r in sorted(attempts, key=lambda x: -x["test_qwk"])[:5]:
    print(f"  {r['model']:<25} test={r['test_qwk']:.4f}  val={r['val_qwk']:.4f}")

