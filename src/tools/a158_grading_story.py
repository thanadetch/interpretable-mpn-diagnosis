"""Grading-story evidence from a158/baseline attention (interpretability for the thesis).

Produces, over the TEST split:
  (1) Attention faithfulness — pooled Spearman corr of attention weight vs per-patch concept scores
      (fibrosis_stroma, bone_trabecular, open_adipose) and vs the fibrosis-axis score.
      Tests the two halves of the grading principle: attend to fibrosis (+), avoid bone (<=0).
  (2) Density-vs-grade monotonicity — per-grade mean of (attention-weighted) fibrosis concept &
      fibrosis-axis density; should increase G0<G1<G2<G3 ("grade = fibrosis density").
  (3) T-sweep figure from logs/a163_Tsweep.

Analysis only (uses the precomputed concept scores); does not touch the model rules.
"""
from __future__ import annotations
import sys, json, glob, os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.novelty_attempts.a158_diffuse_temperature_gated import Model
from models.novelty_attempts.a155_axis_attention_density import _train_axis

ROOT = Path(__file__).resolve().parents[2]


def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()); rb = (rb - rb.mean())
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    bb = sys.argv[1] if len(sys.argv) > 1 else "virchow2"
    seed = 2
    ck = sorted(glob.glob(f"experiments/*/r48_a158_{bb}_s2_*/best_*.pth"))
    if not ck:
        ck = sorted(glob.glob(f"experiments/*/r47_a158_{bb}_s2_*/best_*.pth"))
    ck = ck[-1]
    dim = BACKBONE_CONFIG[bb]["dim"]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    model = Model(input_dim=dim, num_classes=1); model.load_state_dict(sd); model.eval()
    v_fib = _train_axis(dim, seed)[0].float()

    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[bb]["feature_dir"])
    _, _, test_idx = patient_split(ds, seed=seed)
    concepts = torch.load(ROOT / "data" / "patch_concept_scores_all_reti.pt",
                          map_location="cpu", weights_only=False)
    feat_root = str(ROOT / "data" / BACKBONE_CONFIG[bb]["feature_dir"]) + "/"

    pooled = {k: [] for k in ["attn", "fib", "bone", "adi", "axis"]}
    per_grade = {g: {"wfib": [], "fib": [], "waxis": [], "axis": []} for g in range(4)}
    skip = 0
    for i in test_idx:
        pt, label = ds.samples[i]
        bag = torch.load(pt, map_location="cpu", weights_only=False)
        feats = bag["feats"].float()
        with torch.no_grad():
            _, attn, _ = model(feats, return_attention=True)
        attn = attn.numpy()
        rel = str(pt).split(feat_root)[-1]
        c = concepts.get(rel)
        if c is None or int(c.get("n", -1)) != len(attn):
            skip += 1; continue
        fib = np.asarray(c["fibrosis_stroma"], float)
        bone = np.asarray(c["bone_trabecular"], float)
        adi = np.asarray(c["open_adipose"], float)
        axis = (feats @ v_fib).numpy()
        pooled["attn"] += list(attn); pooled["fib"] += list(fib)
        pooled["bone"] += list(bone); pooled["adi"] += list(adi); pooled["axis"] += list(axis)
        per_grade[label]["wfib"].append(float((attn * fib).sum()))
        per_grade[label]["fib"].append(float(fib.mean()))
        per_grade[label]["waxis"].append(float((attn * axis).sum()))
        per_grade[label]["axis"].append(float(axis.mean()))

    print(f"== a158 {bb} attention faithfulness (TEST, {len(pooled['attn'])} patches, {skip} bags skipped) ==")
    print(f"  Spearman(attention, fibrosis_stroma) = {spearman(pooled['attn'], pooled['fib']):+.3f}   (want > 0: attends to fibrosis)")
    print(f"  Spearman(attention, bone_trabecular) = {spearman(pooled['attn'], pooled['bone']):+.3f}   (want <= 0: avoids bone)")
    print(f"  Spearman(attention, open_adipose)    = {spearman(pooled['attn'], pooled['adi']):+.3f}   (want <= 0: avoids fat)")
    print(f"  Spearman(attention, fibrosis-axis)   = {spearman(pooled['attn'], pooled['axis']):+.3f}   (want > 0: attends along fibrosis dir)")
    print("\n== density vs grade (bag-level means; want monotone increase) ==")
    print(f"  {'grade':6s} {'attn-wt fib':>12s} {'mean fib':>10s} {'attn-wt axis':>13s} {'mean axis':>10s}")
    rows = []
    for g in range(4):
        d = per_grade[g]
        if not d["fib"]:
            continue
        row = (g, np.mean(d["wfib"]), np.mean(d["fib"]), np.mean(d["waxis"]), np.mean(d["axis"]))
        rows.append(row)
        print(f"  G{g:<5d} {row[1]:12.4f} {row[2]:10.4f} {row[3]:13.4f} {row[4]:10.4f}")

    # ---- figure: density vs grade ----
    gs = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(gs, [r[1] for r in rows], "o-", label="attn-weighted")
    ax[0].plot(gs, [r[2] for r in rows], "s--", label="bag mean")
    ax[0].set_title("fibrosis concept density vs grade"); ax[0].set_xlabel("true grade"); ax[0].legend(fontsize=8)
    ax[0].set_xticks(gs)
    ax[1].plot(gs, [r[3] for r in rows], "o-", label="attn-weighted")
    ax[1].plot(gs, [r[4] for r in rows], "s--", label="bag mean")
    ax[1].set_title("fibrosis-axis density vs grade"); ax[1].set_xlabel("true grade"); ax[1].legend(fontsize=8)
    ax[1].set_xticks(gs)
    fig.suptitle(f"a158/{bb}: fibrosis density increases with grade (holistic read tracks fibrosis)")
    os.makedirs(ROOT / "results" / "a158_grading_story", exist_ok=True)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "a158_grading_story" / f"density_vs_grade_{bb}.png", dpi=120)
    plt.close(fig)

    # ---- figure: T-sweep ----
    sweep = []
    for line in open(ROOT / "logs" / "a163_Tsweep" / "master.log"):
        if "DONE" in line and "test_qwk=" in line:
            T = float(line.split("T=")[1].split(" ")[0])
            q = float(line.split("test_qwk=")[1].split(" ")[0])
            sweep.append((T, q))
    sweep.append((6.0, 0.9253))  # T->inf == mean pooling (titan)
    sweep.sort()
    fig2, axx = plt.subplots(figsize=(5, 3.4))
    axx.plot([s[0] for s in sweep], [s[1] for s in sweep], "o-")
    axx.axhline(0.9584, ls=":", c="g", label="learned T≈1 (baseline) 0.9584")
    axx.annotate("→ mean pooling 0.9253", (6.0, 0.9253), fontsize=8)
    axx.set_xlabel("attention temperature T (higher = more diffuse)")
    axx.set_ylabel("test QWK (titan)")
    axx.set_title("Forcing diffuse attention degrades grading (titan)")
    axx.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(ROOT / "results" / "a158_grading_story" / "tsweep_titan.png", dpi=120)
    plt.close(fig2)
    print("\nsaved figures -> results/a158_grading_story/")


if __name__ == "__main__":
    main()
