#!/usr/bin/env python3
"""Does sparse attention actually resist patch-level contamination? Measured on the LOCKED split.

WHY THIS EXISTS
    The claim "ASGAP tolerates patch contamination that breaks softmax attention" has been
    circulating with numbers that came from a DIFFERENT experiment: models TRAINED with CutMix
    augmentation and evaluated on a clean test set. That measures augmentation tolerance at
    training time, not robustness at inference time. This script runs the experiment the claim
    actually describes.

PROTOCOL (inference only; no retraining, no trainer edit)
    For every bag in the locked seed-2 TEST split (259 ROIs), append k = frac * N patches drawn
    from TRAIN-split bags of OTHER patients, then re-predict. Bags are paired: the same diluted
    bag is shown to every model, so differences cannot come from the draw. Reported per model
    x backbone x dilution level:

        QWK            test QWK after dilution (frac = 0 reproduces the run's logged number,
                       which doubles as the check that this script's forward matches the
                       trainer's)
        |drift|        mean |prediction after - prediction before| on the raw scalar
        support        # patches with non-zero attention on the CLEAN bag
        support@2x     the same after appending 2N distractors

KILL CONDITION, stated before the numbers
    alpha-entmax gives EXACT invariance only while the appended patches score below the
    threshold tau -- i.e. only while `support` stays flat as N grows. If support@2x is
    materially larger than support, the invariance theorem does not bind on this data and the
    robustness claim must be dropped regardless of what the QWK column says.

Locked-protocol notes: single split (857/214/259, seed 2) only -- no CV. Test set is read once
per model, which is already the case for these finished runs. Analysis only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from data.bag_dataset import GradingBagDatasetFull  # noqa: E402
from train_grading_reti import BACKBONE_CONFIG, patient_split  # noqa: E402
from cardinality_invariance import build, predict, support_size  # noqa: E402  (shared helpers)

DILUTE = [0.0, 0.25, 0.5, 1.0, 2.0]

# Canonical no-aug seed-2 runs, pinned by the val/test pair reported throughout the project.
# Pinning by number (not by max-test) keeps this honest: these are the runs the thesis quotes.
CANON = {
    ("ABMIL", "virchow2"): (0.8182, 0.9476),
    ("ABMIL", "uni2"):     (0.7743, 0.9418),
    ("ABMIL", "titan"):    (0.7902, 0.9584),
    ("ASGAP", "virchow2"): (0.7976, 0.9588),
    ("ASGAP", "uni2"):     (0.7703, 0.9262),
    ("ASGAP", "titan"):    (0.7968, 0.9600),
}


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3]))


def grade(raw):
    return int(max(0, min(3, round(raw))))


def find_runs():
    """Locate the checkpoint whose logged val/test match the canonical pair, per (model, backbone)."""
    out = {}
    for cf in list(ROOT.glob("experiments/*/*/config.json")) + list(ROOT.glob("experiments/*/*/*/config.json")):
        try:
            c = json.loads(cf.read_text())
        except Exception:
            continue
        if c.get("seed") != 2 or c.get("formulation") != "regression":
            continue
        if c.get("augmentation") or c.get("n_folds"):
            continue
        nid = c.get("novelty_id") or ""
        name = "ABMIL" if c.get("model_type") == "simple" else ("ASGAP" if nid == "a215_learnable_entmax" else None)
        bb = c.get("backbone")
        if name is None or (name, bb) not in CANON:
            continue
        d = cf.parent
        ck = list(d.glob("best_*.pth"))
        vm, tm = d / "val_metrics.json", d / "test_metrics.json"
        if not (ck and vm.exists() and tm.exists()):
            continue
        v = json.loads(vm.read_text())["val_qwk"]
        t = json.loads(tm.read_text())["test_qwk"]
        want_v, want_t = CANON[(name, bb)]
        if abs(v - want_v) < 5e-4 and abs(t - want_t) < 5e-4:
            out[(name, bb)] = (ck[0], c, t)
    return out


def main() -> None:
    dev = torch.device("cpu")
    runs = find_runs()
    missing = [k for k in CANON if k not in runs]
    if missing:
        print("!! ไม่พบ checkpoint ที่ตรงกับตัวเลข canonical:", missing, file=sys.stderr)
    print(f"pinned {len(runs)}/6 canonical runs\n", file=sys.stderr)

    res = defaultdict(dict)
    supp = {}
    for (name, bb), (ck, cfg, logged) in sorted(runs.items()):
        ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[bb]["feature_dir"])
        tr, _, te = patient_split(ds, seed=2)
        model = build(cfg, BACKBONE_CONFIG[bb]["dim"]).to(dev)
        state = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()

        rng = np.random.default_rng(0)                       # same draw for every model
        pool = torch.cat([ds[i][0].squeeze(0) for i in
                          rng.choice(tr, size=min(40, len(tr)), replace=False)], 0)

        ys, preds = defaultdict(list), defaultdict(list)
        drift = defaultdict(list)
        s_clean, s_2x = [], []
        for i in te:
            x = ds[i][0].squeeze(0).to(dev)
            y = ds.samples[i][1]
            base = predict(model, x)
            s = support_size(model, x)
            if np.isfinite(s):
                s_clean.append(s)
            r2 = np.random.default_rng(1000 + int(i))        # bag-specific, model-independent
            for f in DILUTE:
                k = int(round(f * len(x)))
                xx = x if k == 0 else torch.cat(
                    [x, pool[r2.choice(len(pool), size=k, replace=False)]], 0)
                p = predict(model, xx)
                ys[f].append(y)
                preds[f].append(grade(p))
                drift[f].append(abs(p - base))
                if f == 2.0:
                    s2 = support_size(model, xx)
                    if np.isfinite(s2):
                        s_2x.append(s2)
        for f in DILUTE:
            res[(name, bb)][f] = (qwk(ys[f], preds[f]), float(np.mean(drift[f])))
        supp[(name, bb)] = (float(np.mean(s_clean)) if s_clean else float("nan"),
                            float(np.mean(s_2x)) if s_2x else float("nan"))
        print(f"  done {name} {bb} (logged test {logged:.4f}, reproduced {res[(name,bb)][0.0][0]:.4f})",
              file=sys.stderr)

    print("\n" + "=" * 92)
    print("TEST QWK ภายใต้การเติม distractor patch จาก TRAIN patient คนอื่น (inference only)")
    print("=" * 92)
    hdr = "".join(f"{'+'+str(int(f*100))+'%':>10}" for f in DILUTE)
    for bb in ["virchow2", "uni2", "titan"]:
        print(f"\n--- {bb} ---")
        print(f"{'model':<8}{hdr}    {'Δ@+100%':>9}")
        for name in ["ABMIL", "ASGAP"]:
            if (name, bb) not in res:
                continue
            r = res[(name, bb)]
            row = "".join(f"{r[f][0]:>10.4f}" for f in DILUTE)
            print(f"{name:<8}{row}    {r[1.0][0]-r[0.0][0]:>+9.4f}")
        print(f"{'|drift|':<8}", end="")
        for name in ["ABMIL", "ASGAP"]:
            if (name, bb) in res:
                r = res[(name, bb)]
                print(f"  {name}: " + " ".join(f"{r[f][1]:.3f}" for f in DILUTE), end="")
        print()

    print("\n" + "=" * 92)
    print("KILL CONDITION — support ต้องนิ่งเมื่อ N โต ไม่งั้นทฤษฎีบท invariance ไม่ผูก")
    print("=" * 92)
    print(f"{'model':<8}{'backbone':<11}{'support (clean)':>17}{'support (+200%)':>18}{'growth':>10}")
    for bb in ["virchow2", "uni2", "titan"]:
        for name in ["ABMIL", "ASGAP"]:
            if (name, bb) not in supp:
                continue
            a, b = supp[(name, bb)]
            g = "—" if not np.isfinite(a) or not np.isfinite(b) else f"{b/a:.2f}x"
            print(f"{name:<8}{bb:<11}{a:>17.1f}{b:>18.1f}{g:>10}")


if __name__ == "__main__":
    main()
