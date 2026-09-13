"""Global configuration, clinical constants, and label definitions for ACDC pipeline."""

from pathlib import Path
from typing import Dict, List

# --- Cardiac Structure Labels ---
LABELS: Dict[str, int] = {
    "RV": 1,   # Right Ventricle Cavity
    "MYO": 2,  # Left Ventricle Myocardium
    "LV": 3,   # Left Ventricle Cavity
}

LABEL_NAMES: Dict[int, str] = {v: k for k, v in LABELS.items()}

# --- Visualization Palette (Medical-Standard Color Coding) ---
STRUCTURE_COLORS: Dict[str, str] = {
    "RV": "#e74c3c",   # Red
    "MYO": "#f1c40f",  # Yellow / Gold
    "LV": "#3498db",   # Blue
}

# --- Diagnostic Classes ---
DIAGNOSIS_CLASSES: List[str] = ["DCM", "HCM", "MINF", "NOR", "RV"]

DIAGNOSIS_DESCRIPTIONS: Dict[str, str] = {
    "NOR": "Normal cardiac function and morphology",
    "MINF": "Previous myocardial infarction (reduced ejection fraction, wall thinning)",
    "DCM": "Dilated cardiomyopathy (enlarged LV cavity, low ejection fraction)",
    "HCM": "Hypertrophic cardiomyopathy (thickened myocardium, hyperdynamic ejection fraction)",
    "RV": "Abnormal right ventricle (dilated or dysfunctional RV)",
}

# --- Clinical Constants ---
# Standard myocardial tissue density: ~1.05 g/cm^3 (g/mL)
MYOCARDIAL_DENSITY_G_PER_ML: float = 1.05

# --- nnU-Net v2 Dataset Defaults ---
DEFAULT_DATASET_ID: int = 4
DEFAULT_DATASET_NAME: str = f"Dataset{DEFAULT_DATASET_ID:03d}_ACDC"
DEFAULT_NNUNET_CONFIG: str = "3d_fullres"
DEFAULT_TRAINER: str = "nnUNetTrainer_250epochs"
DEFAULT_FOLD: int = 0

# --- Feature Columns for Diagnosis Classifier ---
BASE_CLINICAL_FEATURES: List[str] = [
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_ef_pct",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_ef_pct",
    "myo_mass_g",
]

ALL_CLINICAL_FEATURES: List[str] = BASE_CLINICAL_FEATURES + [
    "lv_edv_indexed",
    "myo_mass_indexed",
]

# --- Default Asset & Output Paths ---
DEFAULT_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_DIR: Path = DEFAULT_PROJECT_ROOT / "assets"
