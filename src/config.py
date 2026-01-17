"""
Configuration constants for MPN Classification and Fibrosis Grading framework.
"""
from pathlib import Path
from typing import Dict

# ============================================================================
# Directory Paths
# ============================================================================
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

# Data directories
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
PROCESSED_GRADING_DIR: Path = PROJECT_ROOT / "data" / "processed_grading"
PROCESSED_GRADING_CLEAN_DIR: Path = PROJECT_ROOT / "data" / "processed_grading_clean"
PROCESSED_SUBTYPE_DIR: Path = PROJECT_ROOT / "data" / "processed_subtype"
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
IMAGE_SIZE: int = 224  # Standard size for pretrained models
IMAGE_EXTENSIONS: tuple = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# ============================================================================
# Model Configuration
# ============================================================================
SUPPORTED_MODELS: tuple = ("resnet18", "efficientnet_b0", "densenet121")
