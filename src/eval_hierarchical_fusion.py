"""
Evaluation script for 2-stage hierarchical probabilistic fusion.
Supports multi-backbone fusion (e.g., Stage 1 = TITAN, Stage 2 = UNI2).

Combines two frozen binary MIL models into a 3-class MPN classifier:
    Stage 1 (PMF vs non-PMF):  softmax -> [p_nonpmf, p_pmf]
    Stage 2 (ET vs PV):        softmax -> [p_et, p_pv]

Fusion:
    P(ET)  = p_nonpmf * p_et_given_nonpmf
    P(PV)  = p_nonpmf * p_pv_given_nonpmf
    P(PMF) = p_pmf
    pred   = argmax([P(ET), P(PV), P(PMF)])
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
from models.dual_stream_mil import DualStreamMIL
from models.mean_pool_mil import MeanPoolMIL
from models.multi_branch_mil import MultiBranchMIL

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

FUSED_CLASS_NAMES = ["ET", "PV", "PMF"]

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# ── helpers ──────────────────────────────────────────────────────────────
def log(msg: str, log_file: Optional[Path] = None) -> None:
    print(msg)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

# ── collate ──────────────────────────────────────────────────────────────
def collate_bags(batch):
    features_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    metrics_list = (
        [item[3] for item in batch] if len(batch[0]) > 3 else [{} for _ in batch]
    )
    return features_list, labels, slide_ids, metrics_list

# ── model builder ────────────────────────────────────────────────────────
def build_model_from_ckpt(ckpt: dict, input_dim: int, device: torch.device) -> nn.Module:
    model_type = ckpt["model_type"]
    ckpt_args = ckpt["args"]
    num_classes = 2  # both stages are binary

    if model_type == "simple":
        model = SimpleGatedMIL(input_dim=input_dim, num_classes=num_classes, topk=ckpt_args.get("topk", 0))
    elif model_type == "clam_sb":
        model = CLAM_SB(input_dim=input_dim, num_classes=num_classes)
    elif model_type == "explicit":
        model = ExplicitMetricsMIL(input_dim=input_dim, num_classes=num_classes)
    elif model_type == "hybrid":
        k = ckpt_args.get("topk", 0)
        model = HybridMIL(input_dim=input_dim, num_classes=num_classes, topk=k if k > 0 else 5)
    elif model_type == "residual_metric":
        model = ResidualMetricMIL(input_dim=input_dim, num_classes=num_classes)
    elif model_type == "dtfd":
        model = DTFDMIL(input_dim=input_dim, num_classes=num_classes, num_pseudo_bags=ckpt_args.get("num_pseudo_bags", 8))
    elif model_type == "dual_stream":
        k = ckpt_args.get("topk", 0)
        model = DualStreamMIL(input_dim=input_dim, num_classes=num_classes, topk=k if k > 0 else 5)
    elif model_type == "multi_branch":
        model = MultiBranchMIL(input_dim=input_dim, num_classes=num_classes, topk_focal=ckpt_args.get("topk", 5))
    elif model_type == "mean_pool":
        model = MeanPoolMIL(vision_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

# ── forward helper ───────────────────────────────────────────────────────
@torch.no_grad()
def forward_binary_probs(
    model: nn.Module, model_type: str, features: torch.Tensor, metrics: dict = None, attention_bias: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model_type == "dtfd":
        logits, _, _ = model(features)
    elif model_type in ("simple", "explicit", "residual_metric"):
        m = metrics if (attention_bias or model_type in ("explicit", "residual_metric")) else None
        logits, _, _ = model(features, metrics=m)
    elif model_type in ("dual_stream", "multi_branch"):
        logits, _, _ = model(features)
    else:
        logits, _, _ = model(features)

    probs = torch.softmax(logits, dim=-1)
    return logits, probs

# ── fusion evaluation ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_hierarchical_fusion(
    model_stage1: nn.Module, model_stage2: nn.Module,
    loader1: DataLoader, loader2: DataLoader,
    ckpt_stage1: dict, ckpt_stage2: dict,
    device: torch.device,
) -> Tuple[List[int], List[int], np.ndarray, List[str], np.ndarray, np.ndarray]:
    
    model_type1 = ckpt_stage1["model_type"]
    model_type2 = ckpt_stage2["model_type"]
    attn_bias1 = ckpt_stage1["args"].get("attention_bias", False)
    attn_bias2 = ckpt_stage2["args"].get("attention_bias", False)

    all_labels, all_preds, all_fused, all_slide_ids = [], [], [], []
    all_s1_probs, all_s2_probs = [], []

    for batch1, batch2 in tqdm(zip(loader1, loader2), desc="  Eval ", leave=False, total=len(loader1)):
        f_list1, labels1, s_ids1, m_list1 = batch1
        f_list2, labels2, s_ids2, m_list2 = batch2

        for i in range(len(f_list1)):
            assert s_ids1[i] == s_ids2[i], f"Mismatch Slide IDs! Stage1: {s_ids1[i]} vs Stage2: {s_ids2[i]}"
            
            features1 = f_list1[i].to(device)
            features2 = f_list2[i].to(device)
            label = labels1[i].item()
            metrics1 = m_list1[i]
            metrics2 = m_list2[i]

            # Stage 1: PMF vs non-PMF (Uses Backbone 1 Features)
            _, probs_s1 = forward_binary_probs(model_stage1, model_type1, features1, metrics=metrics1, attention_bias=attn_bias1)
            p_nonpmf, p_pmf = probs_s1[0].item(), probs_s1[1].item()

            # Stage 2: ET vs PV (Uses Backbone 2 Features)
            _, probs_s2 = forward_binary_probs(model_stage2, model_type2, features2, metrics=metrics2, attention_bias=attn_bias2)
            p_et_given_nonpmf, p_pv_given_nonpmf = probs_s2[0].item(), probs_s2[1].item()

            # Fuse
            p_et = p_nonpmf * p_et_given_nonpmf
            p_pv = p_nonpmf * p_pv_given_nonpmf
            fused = [p_et, p_pv, p_pmf]
            pred = int(np.argmax(fused))

            all_labels.append(label)
            all_preds.append(pred)
            all_fused.append(fused)
            all_slide_ids.append(s_ids1[i])
            all_s1_probs.append([p_nonpmf, p_pmf])
            all_s2_probs.append([p_et_given_nonpmf, p_pv_given_nonpmf])

    return all_labels, all_preds, np.array(all_fused), all_slide_ids, np.array(all_s1_probs), np.array(all_s2_probs)

# ── argument parsing ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2-stage hierarchical probabilistic fusion.")
    parser.add_argument("--data_root", default="data", help="Root data directory (default: data).")
    parser.add_argument("--ckpt_stage1", required=True, help="Path to stage 1 (PMF vs non-PMF) checkpoint.")
    parser.add_argument("--ckpt_stage2", required=True, help="Path to stage 2 (ET vs PV) checkpoint.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--save_csv", action="store_true", help="Save per-sample predictions to CSV.")
    parser.add_argument("--output_dir", default="results/fusion_eval", help="Base directory for outputs.")
    return parser.parse_args()

# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"fusion_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "fusion_eval.log"

    ckpt_stage1 = torch.load(args.ckpt_stage1, map_location=device, weights_only=False)
    ckpt_stage2 = torch.load(args.ckpt_stage2, map_location=device, weights_only=False)

    # Allow different backbones
    bb1, bb2 = ckpt_stage1["backbone"], ckpt_stage2["backbone"]
    cfg1, cfg2 = BACKBONE_CONFIG[bb1], BACKBONE_CONFIG[bb2]
    
    test_idx_full = ckpt_stage1["test_idx"]
    test_idx_etpv = ckpt_stage2["test_idx"]
    assert set(test_idx_etpv).issubset(set(test_idx_full)), "Stage 2 test_idx is not a subset of stage 1 test_idx."

    log(f"{'=' * 60}", log_file)
    log(f"Hierarchical Fusion — Multi-Backbone Evaluation", log_file)
    log(f"{'=' * 60}", log_file)
    log(f"  Stage 1 ckpt : {args.ckpt_stage1}", log_file)
    log(f"  Stage 2 ckpt : {args.ckpt_stage2}", log_file)
    log(f"  Stage 1 Model: {ckpt_stage1['model_type']} | Backbone: {cfg1['display_name']} ({bb1})", log_file)
    log(f"  Stage 2 Model: {ckpt_stage2['model_type']} | Backbone: {cfg2['display_name']} ({bb2})", log_file)
    log(f"{'=' * 60}", log_file)

    # Create two datasets if backbones differ, otherwise they point to the same folder
    feat_dir1 = Path(args.data_root) / cfg1["feature_dir"]
    feat_dir2 = Path(args.data_root) / cfg2["feature_dir"]
    dataset1 = MPNBagDatasetFull(feat_dir1)
    dataset2 = MPNBagDatasetFull(feat_dir2)

    loader1 = DataLoader(Subset(dataset1, test_idx_full), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_bags)
    loader2 = DataLoader(Subset(dataset2, test_idx_full), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_bags)

    model_stage1 = build_model_from_ckpt(ckpt_stage1, cfg1["dim"], device)
    model_stage2 = build_model_from_ckpt(ckpt_stage2, cfg2["dim"], device)

    (all_labels, all_preds, all_fused, all_slide_ids, all_s1_probs, all_s2_probs) = evaluate_hierarchical_fusion(
        model_stage1, model_stage2, loader1, loader2, ckpt_stage1, ckpt_stage2, device
    )

    log(f"\n{'=' * 60}", log_file)
    log(f"Hierarchical Fusion — Test Results", log_file)
    log(f"{'=' * 60}", log_file)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cm_header = "         " + "  ".join(f"{name:>5}" for name in FUSED_CLASS_NAMES)
    log(f"\n  Confusion Matrix:", log_file)
    log(f"  {cm_header}", log_file)
    for i, row in enumerate(cm):
        log(f"  {FUSED_CLASS_NAMES[i]:>5}   {'  '.join(f'{v:5d}' for v in row)}", log_file)

    report = classification_report(all_labels, all_preds, target_names=FUSED_CLASS_NAMES, digits=3, zero_division=0)
    log(f"\n  Classification Report:", log_file)
    for line in report.splitlines():
        if line.rstrip(): log(f"  {line}", log_file)

    accuracy = 100.0 * sum(1 for l, p in zip(all_labels, all_preds) if l == p) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    log(f"\n  Test Accuracy  : {accuracy:.1f}%", log_file)
    log(f"  Macro F1       : {macro_f1:.3f}", log_file)
    log(f"  Macro Recall   : {macro_recall:.3f}", log_file)
    log(f"{'=' * 60}", log_file)

    if args.save_csv:
        csv_path = run_dir / "hierarchical_fusion_predictions.csv"
        label_inv = {0: "ET", 1: "PV", 2: "PMF"}
        pd.DataFrame({
            "slide_id": all_slide_ids,
            "true_label": [label_inv[l] for l in all_labels],
            "pred_label": [label_inv[p] for p in all_preds],
            "p_et": all_fused[:, 0], "p_pv": all_fused[:, 1], "p_pmf": all_fused[:, 2],
            "p_nonpmf_stage1": all_s1_probs[:, 0], "p_pmf_stage1": all_s1_probs[:, 1],
            "p_et_stage2": all_s2_probs[:, 0], "p_pv_stage2": all_s2_probs[:, 1],
        }).to_csv(csv_path, index=False)
        log(f"\n  Predictions saved to: {csv_path}", log_file)

if __name__ == "__main__":
    main()
