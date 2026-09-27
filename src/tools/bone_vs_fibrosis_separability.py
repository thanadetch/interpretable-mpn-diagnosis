"""Diagnostic: are BONE and FIBROSIS patches separable in the frozen feature space?

Answers "can we separate bone from fibrosis at all?" using concept-labelled patches:
  - bone exemplars     = high bone_trabecular & low fibrosis_stroma
  - fibrosis exemplars = high fibrosis_stroma & low bone_trabecular
Then:
  (1) linear separability — fit the mean-difference direction on a TRAIN patch split, report
      test AUC of bone-vs-fibrosis (high AUC => the features DO separate them).
  (2) geometry vs the fibrosis grading axis v_fib=(μ_G3−μ_G0): cosine(bone→fib dir, v_fib), and
      where bone vs fibrosis exemplars project on v_fib (does bone sit high on the grading axis?
      that would explain why a grade-axis read cannot avoid bone).
Analysis only.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import torch

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from data.bag_dataset import GradingBagDatasetFull
from train_grading_reti import patient_split, BACKBONE_CONFIG
from models.novelty_attempts.a155_axis_attention_density import _train_axis

ROOT = Path(__file__).resolve().parents[2]


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, float); ranks[order] = np.arange(len(scores))
    pos = labels == 1; n_pos = pos.sum(); n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0: return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main():
    bb = sys.argv[1] if len(sys.argv) > 1 else "virchow2"
    seed = 2
    dim = BACKBONE_CONFIG[bb]["dim"]
    ds = GradingBagDatasetFull(ROOT / "data" / BACKBONE_CONFIG[bb]["feature_dir"])
    v_fib = _train_axis(dim, seed)[0].float().numpy()
    concepts = torch.load(ROOT / "data" / "patch_concept_scores_all_reti.pt", map_location="cpu", weights_only=False)
    feat_root = str(ROOT / "data" / BACKBONE_CONFIG[bb]["feature_dir"]) + "/"

    F, BONE, FIB = [], [], []
    for i in range(len(ds.samples)):
        pt, _ = ds.samples[i]
        rel = str(pt).split(feat_root)[-1]
        c = concepts.get(rel)
        bag = torch.load(pt, map_location="cpu", weights_only=False)
        feats = bag["feats"].float().numpy()
        if c is None or int(c.get("n", -1)) != feats.shape[0]:
            continue
        F.append(feats); BONE.append(np.asarray(c["bone_trabecular"], float)); FIB.append(np.asarray(c["fibrosis_stroma"], float))
    F = np.concatenate(F); BONE = np.concatenate(BONE); FIB = np.concatenate(FIB)
    print(f"pooled patches: {len(F)}")

    # z-score the two concepts, define clean exemplars
    bz = (BONE - BONE.mean()) / BONE.std(); fz = (FIB - FIB.mean()) / FIB.std()
    bone_ex = (bz > np.quantile(bz, 0.75)) & (fz < np.quantile(fz, 0.50))
    fib_ex = (fz > np.quantile(fz, 0.75)) & (bz < np.quantile(fz, 0.50))
    print(f"bone exemplars: {bone_ex.sum()} | fibrosis exemplars: {fib_ex.sum()}")

    idx = np.where(bone_ex | fib_ex)[0]
    y = fib_ex[idx].astype(int)  # 1 = fibrosis, 0 = bone
    X = F[idx]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(idx)); cut = int(0.7 * len(idx))
    tr, te = perm[:cut], perm[cut:]
    # mean-difference direction (fibrosis - bone) on TRAIN
    d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    d = d / (np.linalg.norm(d) + 1e-8)
    s_te = X[te] @ d
    print(f"\n(1) LINEAR SEPARABILITY (test AUC, fibrosis-vs-bone) = {auc(s_te, y[te]):.3f}")
    print("    1.0=perfectly separable, 0.5=indistinguishable")

    # (2) geometry vs grading axis
    cos = float(d @ v_fib / (np.linalg.norm(v_fib) + 1e-8))
    print(f"\n(2) cosine(bone→fibrosis direction, fibrosis grading axis v_fib) = {cos:+.3f}")
    proj_bone = (F[bone_ex] @ v_fib).mean(); proj_fib = (F[fib_ex] @ v_fib).mean(); proj_all = (F @ v_fib).mean()
    print(f"    mean projection on v_fib:  bone={proj_bone:+.2f}  fibrosis={proj_fib:+.2f}  (all={proj_all:+.2f})")
    print("    -> if bone projects ~as high as fibrosis on v_fib, a grade-axis read CANNOT avoid bone;")
    print("       if bone projects low/negative, bone is separable from the grading axis.")


if __name__ == "__main__":
    main()
