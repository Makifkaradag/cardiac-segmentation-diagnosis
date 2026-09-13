"""Segmentation evaluation metrics, including 3D multi-structure Dice coefficient."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import LABELS
from src.data.io import load_nifti

logger = logging.getLogger(__name__)


def compute_dice_score(pred: np.ndarray, gt: np.ndarray, label_val: int) -> float:
    """Compute the Sorensen-Dice similarity coefficient for a specific label value.

    Dice = 2 * |A intersect B| / (|A| + |B|)

    Args:
        pred: Predicted integer segmentation array.
        gt: Ground-truth integer segmentation array.
        label_val: Integer class value to evaluate.

    Returns:
        Dice coefficient between 0.0 and 1.0 (returns 1.0 if both masks are empty).
    """
    pred_bin = (pred == label_val).astype(np.uint8)
    gt_bin = (gt == label_val).astype(np.uint8)

    denominator = int(pred_bin.sum() + gt_bin.sum())
    if denominator == 0:
        return 1.0

    intersection = int((pred_bin & gt_bin).sum())
    return (2.0 * intersection) / float(denominator)


def evaluate_segmentations(
    predictions_dir: Union[str, Path],
    ground_truth_dir: Union[str, Path],
    case_names: Optional[List[str]] = None,
    labels: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Evaluate predictions against ground-truth files for all specified cases.

    Args:
        predictions_dir: Directory containing predicted .nii.gz masks.
        ground_truth_dir: Directory containing ground-truth .nii.gz masks.
        case_names: Optional list of case names. If None, scans predictions_dir.
        labels: Label dictionary mapping structure name to label integer.

    Returns:
        DataFrame with case details and per-structure Dice coefficients.
    """
    pred_dir = Path(predictions_dir)
    gt_dir = Path(ground_truth_dir)

    if labels is None:
        labels = LABELS

    if case_names is None:
        case_names = sorted(
            p.name.replace(".nii.gz", "") for p in pred_dir.glob("*.nii.gz")
        )

    records: List[Dict] = []
    for case_name in case_names:
        pred_path = pred_dir / f"{case_name}.nii.gz"
        gt_path = gt_dir / f"{case_name}.nii.gz"

        if not pred_path.exists():
            logger.warning("Prediction not found for case: %s", case_name)
            continue
        if not gt_path.exists():
            logger.warning("Ground truth not found for case: %s", case_name)
            continue

        pred_arr, _ = load_nifti(pred_path)
        gt_arr, _ = load_nifti(gt_path)

        # Parse patient ID and phase from case name (e.g. ACDC_patient101_ED)
        parts = case_name.split("_")
        patient_id = parts[1] if len(parts) > 1 else case_name
        phase = parts[2].lower() if len(parts) > 2 else "unknown"

        row = {
            "case": case_name,
            "patient_id": patient_id,
            "phase": phase,
        }

        dice_vals = []
        for name, val in labels.items():
            d = compute_dice_score(pred_arr, gt_arr, val)
            row[f"dice_{name}"] = d
            dice_vals.append(d)

        row["mean_dice"] = float(np.mean(dice_vals)) if dice_vals else 0.0
        records.append(row)

    df = pd.DataFrame(records)
    logger.info("Evaluated %d cases. Mean Dice: %.3f", len(df), df["mean_dice"].mean() if not df.empty else 0.0)
    return df
