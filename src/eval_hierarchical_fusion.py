"""
Evaluation script for 2-stage hierarchical probabilistic fusion.

Combines two frozen binary MIL models into a 3-class MPN classifier:
    Stage 1 (PMF vs non-PMF):  softmax -> [p_nonpmf, p_pmf]
    Stage 2 (ET vs PV):        softmax -> [p_et, p_pv]

Fusion:
    P(ET)  = p_nonpmf * p_et_given_nonpmf
    P(PV)  = p_nonpmf * p_pv_given_nonpmf
    P(PMF) = p_pmf
    pred   = argmax([P(ET), P(PV), P(PMF)])

Usage:
    python -m src.eval_hierarchical_fusion \
        --backbone titan \
        --ckpt_stage1 experiments/.../best_pmf_vs_nonpmf_simple_titan.pth \
        --ckpt_stage2 experiments/.../best_et_vs_pv_simple_titan_topk10.pth \
        --save_csv
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from tqdm import tqdm

from core.config import CLASS_MAP, SEED
from data.bag_dataset import MPNBagDatasetFull
from models.clam import CLAM_SB
from models.dtfd_mil import DTFDMIL
from models.explicit_mil import ExplicitMetricsMIL
from models.hybrid_mil import HybridMIL
from models.residual_metric_mil import ResidualMetricMIL
from models.simple_mil import SimpleGatedMIL

# ── backbone configuration ───────────────────────────────────────────────
BACKBONE_CONFIG: Dict[str, Dict] = {
    "titan": {
        "dim": 768,
        "feature_dir": "features_titan",
        "display_name": "TITAN",
    },
    "uni2": {
        "dim": 1536,
        "feature_dir": "features_uni2",
        "display_name": "UNI2-h",
    },
    "virchow2": {
        "dim": 1280,
        "feature_dir": "features_virchow2",
        "display_name": "Virchow2",
    },
}

# Final 3-class output order
FUSED_CLASS_NAMES = ["ET", "PV", "PMF"]

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


# ── helpers ──────────────────────────────────────────────────────────────
def log(msg: str, log_file: Optional[Path] = None) -> None:
    """Print to console and optionally append to a log file."""
    print(msg)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


# ── collate ──────────────────────────────────────────────────────────────
def collate_bags(batch):
    """Collate for full-label bags (no label remapping)."""
    features_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    metrics_list = (
        [item[3] for item in batch] if len(batch[0]) > 3 else [{} for _ in batch]
    )
    return features_list, labels, slide_ids, metrics_list


# ── model builder ────────────────────────────────────────────────────────
def build_model_from_ckpt(
    ckpt: dict,
    input_dim: int,
    device: torch.device,
) -> nn.Module:
    """Rebuild a binary MIL model from checkpoint metadata and load weights."""
    model_type = ckpt["model_type"]
    ckpt_args = ckpt["args"]
    num_classes = 2  # both stages are binary

    if model_type == "simple":
        model = SimpleGatedMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=ckpt_args.get("topk", 0),
        )
    elif model_type == "clam_sb":
        model = CLAM_SB(
            input_dim=input_dim,
            num_classes=num_classes,
        )
    elif model_type == "explicit":
        model = ExplicitMetricsMIL(
            input_dim=input_dim,
            num_classes=num_classes,
        )
    elif model_type == "hybrid":
        k = ckpt_args.get("topk", 0)
        k = k if k > 0 else 5
        model = HybridMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            topk=k,
        )
    elif model_type == "residual_metric":
        model = ResidualMetricMIL(
            input_dim=input_dim,
            num_classes=num_classes,
        )
    elif model_type == "dtfd":
        model = DTFDMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            num_pseudo_bags=ckpt_args.get("num_pseudo_bags", 8),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ── forward helper ───────────────────────────────────────────────────────
@torch.no_grad()
def forward_binary_probs(
    model: nn.Module,
    model_type: str,
    features: torch.Tensor,
    metrics: dict = None,
    attention_bias: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a binary model and return (logits, softmax probs).

    Returns:
        logits: [2] raw logits.
        probs:  [2] softmax probabilities.
    """
    if model_type == "dtfd":
        logits, _, _ = model(features)
    elif model_type in ("simple", "explicit", "residual_metric"):
        m = metrics if (attention_bias or model_type in ("explicit", "residual_metric")) else None
        logits, _, _ = model(features, metrics=m)
    else:
        logits, _, _ = model(features)

    probs = torch.softmax(logits, dim=-1)
    return logits, probs


# ── fusion evaluation ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_hierarchical_fusion(
    model_stage1: nn.Module,
    model_stage2: nn.Module,
    test_loader: DataLoader,
    ckpt_stage1: dict,
    ckpt_stage2: dict,
    device: torch.device,
) -> Tuple[List[int], List[int], np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Run hierarchical fusion on the full 3-class test set.

    Returns:
        all_labels:    true labels (flat: ET=0, PV=1, PMF=2)
        all_preds:     predicted labels
        all_fused:     fused probabilities [N, 3]
        all_slide_ids: slide identifiers
        all_s1_probs:  stage1 probs [N, 2]
        all_s2_probs:  stage2 probs [N, 2]
    """
    model_type1 = ckpt_stage1["model_type"]
    model_type2 = ckpt_stage2["model_type"]
    attn_bias1 = ckpt_stage1["args"].get("attention_bias", False)
    attn_bias2 = ckpt_stage2["args"].get("attention_bias", False)

    all_labels = []
    all_preds = []
    all_fused = []
    all_slide_ids = []
    all_s1_probs = []
    all_s2_probs = []

    for features_list, labels, slide_ids, metrics_list in tqdm(
        test_loader, desc="  Eval ", leave=False
    ):
        for i, features in enumerate(features_list):
            features = features.to(device)
            label = labels[i].item()
            metrics = metrics_list[i]

            # Stage 1: PMF vs non-PMF
            _, probs_s1 = forward_binary_probs(
                model_stage1, model_type1, features,
                metrics=metrics, attention_bias=attn_bias1,
            )
            p_nonpmf = probs_s1[0].item()
            p_pmf = probs_s1[1].item()

            # Stage 2: ET vs PV
            _, probs_s2 = forward_binary_probs(
                model_stage2, model_type2, features,
                metrics=metrics, attention_bias=attn_bias2,
            )
            p_et_given_nonpmf = probs_s2[0].item()
            p_pv_given_nonpmf = probs_s2[1].item()

            # Fuse: [P(ET), P(PV), P(PMF)]
            p_et = p_nonpmf * p_et_given_nonpmf
            p_pv = p_nonpmf * p_pv_given_nonpmf
            fused = [p_et, p_pv, p_pmf]
            pred = int(np.argmax(fused))

            all_labels.append(label)
            all_preds.append(pred)
            all_fused.append(fused)
            all_slide_ids.append(slide_ids[i])
            all_s1_probs.append([p_nonpmf, p_pmf])
            all_s2_probs.append([p_et_given_nonpmf, p_pv_given_nonpmf])

    all_fused = np.array(all_fused)
    all_s1_probs = np.array(all_s1_probs)
    all_s2_probs = np.array(all_s2_probs)

    return all_labels, all_preds, all_fused, all_slide_ids, all_s1_probs, all_s2_probs


# ── argument parsing ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 2-stage hierarchical probabilistic fusion for MPN classification."
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--ckpt_stage1",
        required=True,
        help="Path to stage 1 (PMF vs non-PMF) checkpoint.",
    )
    parser.add_argument(
        "--ckpt_stage2",
        required=True,
        help="Path to stage 2 (ET vs PV) checkpoint.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers."
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save per-sample predictions to CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/fusion_eval",
        help="Base directory for outputs (default: results/fusion_eval).",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output directory & log file ───────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"fusion_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "fusion_eval.log"

    # ── Load checkpoints ──────────────────────────────────────────────
    ckpt_stage1 = torch.load(args.ckpt_stage1, map_location=device, weights_only=False)
    ckpt_stage2 = torch.load(args.ckpt_stage2, map_location=device, weights_only=False)

    # ── Safety checks ─────────────────────────────────────────────────
    backbone = ckpt_stage1["backbone"]
    assert backbone == ckpt_stage2["backbone"], (
        f"Backbone mismatch: stage1={backbone}, stage2={ckpt_stage2['backbone']}"
    )

    # Stage 1 saves the full 3-class test_idx; stage 2 saves ET+PV-only test_idx.
    # Verify that stage 2's test_idx is the ET+PV subset of stage 1's test_idx.
    test_idx_full = ckpt_stage1["test_idx"]
    test_idx_etpv = ckpt_stage2["test_idx"]
    assert set(test_idx_etpv).issubset(set(test_idx_full)), (
        "Stage 2 test_idx is not a subset of stage 1 test_idx — checkpoints may come from different splits."
    )

    cfg = BACKBONE_CONFIG[backbone]
    input_dim = cfg["dim"]
    features_dir = Path(args.data_root) / cfg["feature_dir"]

    # ── Sanity summary ────────────────────────────────────────────────
    log(f"{'=' * 60}", log_file)
    log(f"Hierarchical Probabilistic Fusion — Evaluation", log_file)
    log(f"{'=' * 60}", log_file)
    log(f"  Stage 1 ckpt : {args.ckpt_stage1}", log_file)
    log(f"  Stage 2 ckpt : {args.ckpt_stage2}", log_file)
    log(f"  Backbone     : {cfg['display_name']} ({backbone})", log_file)
    log(f"  Stage 1 model: {ckpt_stage1['model_type']}", log_file)
    log(f"  Stage 2 model: {ckpt_stage2['model_type']}", log_file)
    log(f"  Device       : {device}", log_file)
    log(f"  Output dir   : {run_dir}", log_file)
    log("", log_file)
    log(f"  Class semantics:", log_file)
    log(f"    Stage 1: [non-PMF, PMF]", log_file)
    log(f"    Stage 2: [ET, PV]", log_file)
    log(f"    Fused  : [ET, PV, PMF]", log_file)
    log("", log_file)
    log(f"  test_idx consistency: OK", log_file)
    log(f"    Stage 1 test_idx (full 3-class): {len(test_idx_full)} samples", log_file)
    log(f"    Stage 2 test_idx (ET+PV only) : {len(test_idx_etpv)} samples", log_file)
    log(f"{'=' * 60}", log_file)

    # ── Build full 3-class test set ───────────────────────────────────
    full_dataset = MPNBagDatasetFull(features_dir)
    log(f"\n  Total bags in dataset: {len(full_dataset)}", log_file)
    log(f"  Using full 3-class test set: {len(test_idx_full)} samples", log_file)

    test_loader = DataLoader(
        Subset(full_dataset, test_idx_full),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bags,
    )

    # ── Build models ──────────────────────────────────────────────────
    model_stage1 = build_model_from_ckpt(ckpt_stage1, input_dim, device)
    model_stage2 = build_model_from_ckpt(ckpt_stage2, input_dim, device)
    log(f"\n  Models loaded and set to eval mode.", log_file)

    # ── Run fusion evaluation ─────────────────────────────────────────
    log(f"\n  Running hierarchical fusion on test set...", log_file)
    (
        all_labels,
        all_preds,
        all_fused,
        all_slide_ids,
        all_s1_probs,
        all_s2_probs,
    ) = evaluate_hierarchical_fusion(
        model_stage1, model_stage2, test_loader,
        ckpt_stage1, ckpt_stage2, device,
    )

    # ── Report metrics ────────────────────────────────────────────────
    log(f"\n{'=' * 60}", log_file)
    log(f"Hierarchical Fusion — Test Results", log_file)
    log(f"{'=' * 60}", log_file)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cm_header = "         " + "  ".join(f"{name:>5}" for name in FUSED_CLASS_NAMES)
    log(f"\n  Confusion Matrix:", log_file)
    log(f"  {cm_header}", log_file)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:5d}" for v in row)
        log(f"  {FUSED_CLASS_NAMES[i]:>5}   {row_str}", log_file)

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=FUSED_CLASS_NAMES,
        digits=3,
        zero_division=0,
    )
    log(f"\n  Classification Report:", log_file)
    for line in report.splitlines():
        if line.rstrip():
            log(f"  {line}", log_file)

    # Summary metrics
    accuracy = 100.0 * sum(1 for l, p in zip(all_labels, all_preds) if l == p) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    log(f"\n  Test Accuracy  : {accuracy:.1f}%", log_file)
    log(f"  Macro F1       : {macro_f1:.3f}", log_file)
    log(f"  Macro Recall   : {macro_recall:.3f}", log_file)
    log(f"{'=' * 60}", log_file)

    # ── Optional CSV export ───────────────────────────────────────────
    if args.save_csv:
        csv_path = run_dir / "hierarchical_fusion_predictions.csv"

        label_inv = {0: "ET", 1: "PV", 2: "PMF"}
        df = pd.DataFrame(
            {
                "slide_id": all_slide_ids,
                "true_label": [label_inv[l] for l in all_labels],
                "pred_label": [label_inv[p] for p in all_preds],
                "p_et": all_fused[:, 0],
                "p_pv": all_fused[:, 1],
                "p_pmf": all_fused[:, 2],
                "p_nonpmf_stage1": all_s1_probs[:, 0],
                "p_pmf_stage1": all_s1_probs[:, 1],
                "p_et_stage2": all_s2_probs[:, 0],
                "p_pv_stage2": all_s2_probs[:, 1],
            }
        )
        df.to_csv(csv_path, index=False)
        log(f"\n  Predictions saved to: {csv_path}", log_file)


if __name__ == "__main__":
    main()
