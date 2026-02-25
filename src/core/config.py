"""
Configuration constants for MPN Classification and Fibrosis Grading framework.
"""

from pathlib import Path
from typing import Dict

# ============================================================================
# Directory Paths
# ============================================================================
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.resolve()

# Data directories
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
PROCESSED_GRADING_DIR: Path = PROJECT_ROOT / "data" / "processed_grading"
PROCESSED_GRADING_CLEAN_DIR: Path = PROJECT_ROOT / "data" / "processed_grading_clean"
PROCESSED_SUBTYPE_DIR: Path = PROJECT_ROOT / "data" / "processed_subtype"
PROCESSED_SUBTYPE_CLEAN_DIR: Path = PROJECT_ROOT / "data" / "processed_subtype_clean"
# Other directories
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# ============================================================================
# Data Mode Configuration
# ============================================================================
# Maps data_mode argument to (data_dir, file_extension)
DATA_MODE_CONFIG: Dict[str, Dict[str, any]] = {
    "resize": {
        "data_dir": RAW_DATA_DIR,
        "extension": "tif",
        "description": "Load raw .tif images and resize to 224x224",
        "is_patch": False,
    },
    "patch": {
        "data_dir": PROCESSED_DATA_DIR,
        "extension": "png",
        "description": "Load preprocessed 512x512 .png patches",
        "is_patch": True,
    },
    "grading_patch": {
        "data_dir": PROCESSED_GRADING_DIR,
        "extension": "png",
        "description": "Fine-Grained 224x224 (Sharp Texture)",
        "is_patch": True,
    },
    "subtype_patch": {
        "data_dir": PROCESSED_SUBTYPE_DIR,
        "extension": "png",
        "description": "Coarse 512x512 (Cellular Context)",
        "is_patch": True,
    },
    "grading_patch_clean": {
        "data_dir": PROCESSED_GRADING_CLEAN_DIR,
        "extension": "png",
        "description": "Cleaned grading patches",
        "is_patch": True,
    },
    "subtype_patch_clean": {
        "data_dir": PROCESSED_SUBTYPE_CLEAN_DIR,
        "extension": "png",
        "description": "Cleaned subtype patches (H&E)",
        "is_patch": True,
    },
}

DEFAULT_DATA_MODE: str = "resize"

# ============================================================================
# Training Hyperparameters
# ============================================================================
SEED: int = 42
BATCH_SIZE: int = 32
LR: float = 1e-4
EPOCHS: int = 50
NUM_WORKERS: int = 4

# ============================================================================
# Data Split Ratios
# ============================================================================
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ============================================================================
# Class Mappings
# ============================================================================
# Classification Task: MPN Subtype (H&E Images)
CLASS_MAP: Dict[str, int] = {
    "ET": 0,  # Essential Thrombocythemia
    "PV": 1,  # Polycythemia Vera
    "PMF": 2,  # Primary Myelofibrosis
}

# Grading Task: Fibrosis Grade (Reticulin Images)
GRADE_MAP: Dict[str, int] = {
    "G0": 0,  # No fibrosis
    "G1": 1,  # Mild fibrosis
    "G2": 2,  # Moderate fibrosis
    "G3": 3,  # Severe fibrosis
}

# Inverse mappings for evaluation/visualization
CLASS_MAP_INV: Dict[int, str] = {v: k for k, v in CLASS_MAP.items()}
GRADE_MAP_INV: Dict[int, str] = {v: k for k, v in GRADE_MAP.items()}

# ============================================================================
# Image Configuration
# ============================================================================
IMAGE_SIZE: int = 224  # Default/Fallback size for pretrained models

# Task-specific resolutions
IMAGE_SIZE_GRADING: int = 224  # Standard size for Reticulin fibrosis grading
IMAGE_SIZE_SUBTYPE: int = 224

IMAGE_EXTENSIONS: tuple = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# ============================================================================
# Model Configuration
# ============================================================================
SUPPORTED_MODELS: tuple = ("resnet18", "densenet121")


# ============================================================================
# Hugging Face Authentication
# ============================================================================


def hf_login() -> None:
    """
    Log in to Hugging Face Hub without interactive token entry.

    Resolution order:
        1. HF_TOKEN environment variable
        2. Previously cached token (~/.cache/huggingface/token)

    Setup (one-time):
        - Local:  export HF_TOKEN='hf_...' in ~/.zshrc
        - Colab:  In a notebook cell BEFORE running scripts:
                    from google.colab import userdata
                    import os
                    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    """
    import os

    from huggingface_hub import login

    # 1. Environment variable (works everywhere including Colab)
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("  ✅ HF login via HF_TOKEN env var")
        return

    # 2. Cached token file
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            login(token=token, add_to_git_credential=False)
            print("  ✅ HF login via cached token")
            return

    raise RuntimeError(
        "No HF token found. Please set it up (one-time):\n"
        "  - Local: export HF_TOKEN='hf_...' in ~/.zshrc\n"
        "  - Colab: Run this in a cell BEFORE your script:\n"
        "      from google.colab import userdata; import os\n"
        '      os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")\n'
        "  - Or run: huggingface-cli login --token hf_..."
    )
