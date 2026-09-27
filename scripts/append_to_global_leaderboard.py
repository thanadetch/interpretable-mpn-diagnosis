"""Append v1+v2+v3 novelty attempt rows (and recent baselines) to results/leaderboard.csv
if they are not already present. Idempotent.
"""
import csv
import json
import os
import glob

LB = "results/leaderboard.csv"
HEADER = [
    "timestamp", "run_name", "experiment_dir", "model_type", "backbone", "seed",
    "split_seed", "formulation", "main_metric", "best_epoch",
    "val_qwk", "val_mae", "val_accuracy", "val_f1_macro", "val_macro_recall",
    "val_loss", "checkpoint",
]

existing = set()
if os.path.isfile(LB):
    with open(LB, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(row["experiment_dir"])

new_rows = []
# Cover both legacy flat layout (experiments/reti_*) and the new
# date-bucketed layout (experiments/<YYYYMMDD>/reti_*).
patterns = [
    "experiments/reti_novelty_attempt_uni2_a*_s2_*",
    "experiments/*/reti_novelty_attempt_uni2_a*_s2_*",
    "experiments/reti_mean_pool_uni2_s2_*",
    "experiments/*/reti_mean_pool_uni2_s2_*",
]
seen_dirs = set()
for pat in patterns:
    for d in sorted(glob.glob(pat)):
        abs_d = os.path.abspath(d)
        if abs_d in seen_dirs:
            continue
        seen_dirs.add(abs_d)
        if abs_d in existing:
            continue
        cp = os.path.join(d, "config.json")
        vp = os.path.join(d, "val_metrics.json")
        if not (os.path.isfile(cp) and os.path.isfile(vp)):
            continue
        cfg = json.load(open(cp))
        v = json.load(open(vp))
        # checkpoint
        ckpts = glob.glob(os.path.join(d, "best_*.pth"))
        ckpt = os.path.abspath(ckpts[0]) if ckpts else ""
        new_rows.append([
            cfg.get("timestamp", ""),
            cfg.get("run_name", ""),
            abs_d,
            cfg.get("model_type", ""),
            cfg.get("backbone", ""),
            cfg.get("seed", ""),
            cfg.get("split_seed", cfg.get("seed", "")),
            cfg.get("formulation", ""),
            cfg.get("main_metric", ""),
            v.get("best_epoch", ""),
            f'{v.get("val_qwk", float("nan")):.6f}',
            f'{v.get("val_mae", float("nan")):.6f}',
            f'{v.get("val_accuracy", float("nan")):.6f}',
            f'{v.get("val_f1_macro", float("nan")):.6f}',
            f'{v.get("val_macro_recall", float("nan")):.6f}',
            f'{v.get("val_loss", float("nan")):.6f}',
            ckpt,
        ])

if not new_rows:
    print("No new rows to append.")
else:
    file_exists = os.path.isfile(LB)
    with open(LB, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(HEADER)
        for r in new_rows:
            w.writerow(r)
    print(f"Appended {len(new_rows)} rows to {LB}")

