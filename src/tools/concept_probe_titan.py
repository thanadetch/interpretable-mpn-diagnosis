"""
Zero-Shot Concept Probing Sanity Check — TITAN / CONCHv1.5.

Probes pre-extracted CONCHv1.5 patch embeddings against a set of
morphological text concepts (encoded via TITAN's text encoder) to
compute patch-level cosine similarities *without* training any
classifier.

Phases:
    1. Load TITAN text encoder + tokenizer; embed concept prompts.
    2. Define morphological concept prompts.
    3. For each ROI .pt file, compute patch–concept cosine similarities
       and aggregate into mean and top-20 % mean scores.
    4. Save ROI-level scores to a CSV for downstream group comparison.
    5. Copy the global top-20 patches per concept into a visual sanity-
       check directory.

Input:  data/features_titan/{Class}/{PatientID}/{ImgID}.pt
Output: data/roi_concept_scores.csv
        data/concept_sanity_check/{Concept_Name}/top_{rank}.png

Usage:
    python -m src.tools.concept_probe_titan

    python -m src.tools.concept_probe_titan \
        --features_dir data/features_titan \
        --output_csv data/roi_concept_scores.csv \
        --sanity_dir data/concept_sanity_check \
        --device cpu \
        --top_k 20
"""

import argparse
import heapq
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# Ensure src/ is on sys.path when invoked as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import hf_login

# =============================================================================
# Constants
# =============================================================================
DEFAULT_FEATURES_DIR = "data/features_titan"
DEFAULT_OUTPUT_CSV = "data/roi_concept_scores.csv"
DEFAULT_SANITY_DIR = "data/concept_sanity_check"
CLASSES = ["ET", "PV", "PMF"]
FEATURE_DIM = 768

# Phase 2 — Morphological concept prompts (no direct disease labels)
CONCEPT_PROMPTS: List[Dict[str, str]] = [
    # -----------------------------------------------------
    # Cellularity / marrow state
    # -----------------------------------------------------
    {"text": "hypercellular marrow region", "group": "cellularity"},
    {"text": "panmyelosis-like hypercellular marrow", "group": "cellularity"},
    {"text": "densely packed hematopoietic marrow", "group": "cellularity"},

    # -----------------------------------------------------
    # Open / adipose / low-cellularity context
    # -----------------------------------------------------
    {"text": "hypocellular or adipose-rich marrow region", "group": "open_adipose"},
    {"text": "open marrow space", "group": "open_adipose"},

    # -----------------------------------------------------
    # Fibrosis / stromal change
    # -----------------------------------------------------
    {"text": "fibrosis-like stromal region", "group": "fibrosis_stroma"},

    # -----------------------------------------------------
    # Bone / trabecular context
    # -----------------------------------------------------
    {"text": "trabecular bone-adjacent marrow", "group": "bone_trabecular"},
    {"text": "bone-rich marrow region", "group": "bone_trabecular"},

    # -----------------------------------------------------
    # Megakaryocyte morphology
    # -----------------------------------------------------
    {"text": "megakaryocyte-rich region", "group": "megakaryocyte_general"},
    {"text": "large hyperlobulated megakaryocyte-rich region", "group": "megakaryocyte_mature"},
    {"text": "atypical clustered megakaryocyte-rich region", "group": "megakaryocyte_atypical"},
]

PROMPT_TEXTS: List[str] = [c["text"] for c in CONCEPT_PROMPTS]


# =============================================================================
# Phase 1 — Text Encoder Setup
# =============================================================================

def sanitize_concept_name(concept: str) -> str:
    """Convert a concept prompt into a safe directory/file name."""
    return re.sub(r"[^a-z0-9]+", "_", concept.lower()).strip("_")


@torch.inference_mode()
def encode_concepts(
    concepts: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Load the TITAN model and tokenizer, encode concept prompts, and
    return L2-normalised text embeddings.

    Returns:
        text_features: Tensor of shape [num_concepts, 768], normalised.
    """
    from transformers import AutoModel, AutoTokenizer

    print("Loading TITAN model + tokenizer for text encoding...")
    hf_login()
    titan = AutoModel.from_pretrained(
        "MahmoodLab/TITAN", trust_remote_code=True
    )
    titan = titan.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "MahmoodLab/TITAN", trust_remote_code=True
    )

    # Tokenise all prompts
    tokens = tokenizer(
        concepts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)  # [num_concepts, L]

    # Encode with normalisation
    text_features = titan.encode_text(input_ids, normalize=True)  # [C, 768]
    text_features = text_features.float().cpu()

    print(f"  Encoded {len(concepts)} concepts → shape {tuple(text_features.shape)}")

    # Free GPU memory — we only need the text features from here on
    del titan, tokenizer, tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return text_features


# =============================================================================
# Phase 3 — Compute Concept Scores per ROI
# =============================================================================

def compute_roi_scores(
    feats: torch.Tensor,
    text_features: torch.Tensor,
    top_fraction: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute patch–concept cosine similarity and aggregate per ROI.

    Args:
        feats:         [N_patches, 768] image patch embeddings (raw).
        text_features: [num_concepts, 768] L2-normalised text embeddings.
        top_fraction:  Fraction of top patches used for top-k mean.

    Returns:
        mean_scores:      [num_concepts] mean cosine similarity.
        top20p_scores:    [num_concepts] mean of top-20 % patches.
    """
    # Normalise image features for cosine similarity
    feats_norm = torch.nn.functional.normalize(feats.float(), dim=-1)  # [N, 768]

    # Cosine similarity matrix: [N_patches, num_concepts]
    sim = feats_norm @ text_features.T  # text_features already normalised

    # Mean across all patches
    mean_scores = sim.mean(dim=0)  # [C]

    # Top-20 % mean per concept
    n_patches = sim.shape[0]
    k = max(1, math.ceil(n_patches * top_fraction))
    topk_vals, _ = sim.topk(k, dim=0)  # [k, C]
    top20p_scores = topk_vals.mean(dim=0)  # [C]

    return mean_scores, top20p_scores, sim


# =============================================================================
# Phase 5 — Global Top-K Tracker (min-heap per concept)
# =============================================================================

class TopKTracker:
    """
    Maintains a global top-k heap of (score, patch_path) per concept
    using a min-heap so the smallest score can be efficiently replaced.
    """

    def __init__(self, concepts: List[str], k: int = 20) -> None:
        self.k = k
        self.concepts = concepts
        # heaps[concept_idx] → list of (score, unique_counter, path)
        self._heaps: Dict[int, list] = {i: [] for i in range(len(concepts))}
        self._counter = 0  # tie-breaker for heapq

    def update(
        self,
        sim_matrix: torch.Tensor,
        patch_paths: List[str],
    ) -> None:
        """
        Update heaps with patch-level similarities from one ROI.

        Args:
            sim_matrix: [N_patches, num_concepts] cosine similarities.
            patch_paths: Corresponding patch file paths.
        """
        for concept_idx in range(sim_matrix.shape[1]):
            scores = sim_matrix[:, concept_idx]  # [N]
            heap = self._heaps[concept_idx]

            for patch_idx in range(scores.shape[0]):
                score = scores[patch_idx].item()
                self._counter += 1

                if len(heap) < self.k:
                    heapq.heappush(heap, (score, self._counter, patch_paths[patch_idx]))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, self._counter, patch_paths[patch_idx]))

    def get_top_patches(self, concept_idx: int) -> List[Tuple[float, str]]:
        """Return top patches sorted by score descending."""
        items = self._heaps[concept_idx]
        # Sort descending by score
        items.sort(key=lambda x: -x[0])
        return [(score, path) for score, _, path in items]


# =============================================================================
# Main Pipeline
# =============================================================================

def run(args: argparse.Namespace) -> None:
    features_dir = Path(args.features_dir)
    output_csv = Path(args.output_csv)
    sanity_dir = Path(args.sanity_dir)
    device = torch.device(args.device)
    top_k = args.top_k

    # ------------------------------------------------------------------
    # Phase 1 — Encode text concepts
    # ------------------------------------------------------------------
    text_features = encode_concepts(PROMPT_TEXTS, device)  # [C, 768]
    num_concepts = text_features.shape[0]

    # ------------------------------------------------------------------
    # Phase 2 — Print concept prompts for reference
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Concept Prompts")
    print(f"{'=' * 60}")
    for i, cp in enumerate(CONCEPT_PROMPTS):
        print(f"  [{i}] ({cp['group']:.<28s}) {cp['text']}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Phase 3 & 5 — Iterate over all .pt files
    # ------------------------------------------------------------------
    tracker = TopKTracker(PROMPT_TEXTS, k=top_k)
    records: List[Dict] = []

    # Discover all .pt files
    pt_files: List[Tuple[str, str, str, Path]] = []
    for class_name in CLASSES:
        class_dir = features_dir / class_name
        if not class_dir.exists():
            print(f"⚠ Class directory not found: {class_dir}")
            continue
        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            for pt_file in sorted(patient_dir.glob("*.pt")):
                img_id = pt_file.stem
                pt_files.append((class_name, patient_dir.name, img_id, pt_file))

    print(f"Found {len(pt_files)} ROI .pt files across {len(CLASSES)} classes.\n")

    for class_name, patient_id, img_id, pt_path in tqdm(
        pt_files, desc="Computing concept scores", unit="roi"
    ):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        feats = data["feats"]              # [N, 768]
        patch_paths = data["patch_paths"]  # list of str

        if feats.shape[0] == 0:
            continue

        # Phase 3 — Compute scores
        mean_scores, top20p_scores, sim = compute_roi_scores(
            feats, text_features, top_fraction=0.20
        )

        # Phase 5 — Update global top-K tracker
        tracker.update(sim, patch_paths)

        # Phase 4 — Collect records (one row per concept per ROI)
        for c_idx, cp in enumerate(CONCEPT_PROMPTS):
            records.append(
                {
                    "Class": class_name,
                    "PatientID": patient_id,
                    "ImgID": img_id,
                    "Concept": cp["text"],
                    "ConceptGroup": cp["group"],
                    "MeanScore": round(mean_scores[c_idx].item(), 6),
                    "Top20pMeanScore": round(top20p_scores[c_idx].item(), 6),
                }
            )

    # ------------------------------------------------------------------
    # Phase 4 — Save CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved ROI concept scores → {output_csv}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Phase 5 — Copy top patches for visual sanity check
    # ------------------------------------------------------------------
    print(f"\nCopying top-{top_k} patches per concept → {sanity_dir}/")
    copied = 0
    missing = 0

    for c_idx, cp in enumerate(CONCEPT_PROMPTS):
        concept_dir = sanity_dir / cp["group"] / sanitize_concept_name(cp["text"])
        concept_dir.mkdir(parents=True, exist_ok=True)

        top_patches = tracker.get_top_patches(c_idx)
        for rank, (score, src_path_str) in enumerate(top_patches, start=1):
            src = Path(src_path_str)
            if not src.exists():
                missing += 1
                continue
            dst = concept_dir / f"top_{rank}.png"
            shutil.copy2(src, dst)
            copied += 1

    print(f"  Copied : {copied} patch images")
    if missing > 0:
        print(f"  Missing: {missing} source patches (file not found)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Zero-Shot Concept Probing Complete")
    print(f"{'=' * 60}")
    print(f"  Concepts         : {num_concepts}")
    print(f"  ROIs processed   : {len(pt_files)}")
    print(f"  CSV rows         : {len(df)}")
    print(f"  Sanity patches   : {copied}")
    print(f"  Output CSV       : {output_csv}")
    print(f"  Sanity directory : {sanity_dir}")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-Shot Concept Probing: compute concept–patch cosine "
            "similarities from pre-extracted TITAN/CONCHv1.5 features."
        ),
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=DEFAULT_FEATURES_DIR,
        help="Root directory containing extracted .pt files "
             "(structure: {Class}/{PatientID}/{ImgID}.pt).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the output CSV with ROI-level concept scores.",
    )
    parser.add_argument(
        "--sanity_dir",
        type=str,
        default=DEFAULT_SANITY_DIR,
        help="Directory for top-K patch copies (visual sanity check).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for text encoding (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top patches to save per concept.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
