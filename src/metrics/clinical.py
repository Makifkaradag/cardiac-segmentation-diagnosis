"""Clinical indices derivation (EDV, ESV, Ejection Fraction, Myocardial Mass)."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import LABELS, MYOCARDIAL_DENSITY_G_PER_ML

logger = logging.getLogger(__name__)


def compute_volume_ml(
    mask_arr: np.ndarray,
    label_val: int,
    spacing: Tuple[float, float, float],
) -> float:
    """Compute anatomical structure volume in milliliters (mL) from voxel count and spacing.

    Volume (mL) = Voxel_Count * (dx * dy * dz) / 1000

    Args:
        mask_arr: 3D integer segmentation mask.
        label_val: Integer label value for structure of interest.
        spacing: 3-tuple representing voxel spacing (dx, dy, dz) in mm.

    Returns:
        Volume in milliliters (mL).
    """
    voxel_vol_mm3 = float(np.prod(spacing))
    voxel_count = int((mask_arr == label_val).sum())
    return voxel_count * voxel_vol_mm3 / 1000.0


def calculate_ejection_fraction(edv: float, esv: float) -> float:
    """Calculate ejection fraction (EF) as a percentage.

    EF (%) = (EDV - ESV) / EDV * 100

    Args:
        edv: End-Diastolic Volume in mL.
        esv: End-Systolic Volume in mL.

    Returns:
        Ejection fraction percentage (0-100%). Returns np.nan if EDV <= 0.
    """
    if edv <= 0 or np.isnan(edv):
        return np.nan
    return float((edv - esv) / edv * 100.0)


def extract_clinical_features_from_mask(
    ed_mask: np.ndarray,
    ed_spacing: Tuple[float, float, float],
    es_mask: np.ndarray,
    es_spacing: Tuple[float, float, float],
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
) -> Dict[str, float]:
    """Derive all standard clinical volumetric indices from ED and ES masks.

    Calculates:
      - Left Ventricular End-Diastolic Volume (LV EDV) in mL
      - Left Ventricular End-Systolic Volume (LV ESV) in mL
      - Left Ventricular Ejection Fraction (LV EF) in %
      - Right Ventricular End-Diastolic Volume (RV EDV) in mL
      - Right Ventricular End-Systolic Volume (RV ESV) in mL
      - Right Ventricular Ejection Fraction (RV EF) in %
      - Myocardial Mass in grams (ED volume * 1.05 g/mL)
      - Height-indexed metrics if height is provided.

    Args:
        ed_mask: 3D segmentation mask at end-diastole.
        ed_spacing: Voxel spacing at ED in mm.
        es_mask: 3D segmentation mask at end-systole.
        es_spacing: Voxel spacing at ES in mm.
        height_cm: Patient height in cm (optional).
        weight_kg: Patient weight in kg (optional).

    Returns:
        Dictionary of extracted clinical metrics.
    """
    lv_edv = compute_volume_ml(ed_mask, LABELS["LV"], ed_spacing)
    lv_esv = compute_volume_ml(es_mask, LABELS["LV"], es_spacing)
    rv_edv = compute_volume_ml(ed_mask, LABELS["RV"], ed_spacing)
    rv_esv = compute_volume_ml(es_mask, LABELS["RV"], es_spacing)
    myo_ed_vol = compute_volume_ml(ed_mask, LABELS["MYO"], ed_spacing)

    lv_ef = calculate_ejection_fraction(lv_edv, lv_esv)
    rv_ef = calculate_ejection_fraction(rv_edv, rv_esv)
    myo_mass = myo_ed_vol * MYOCARDIAL_DENSITY_G_PER_ML

    metrics = {
        "lv_edv_ml": lv_edv,
        "lv_esv_ml": lv_esv,
        "lv_ef_pct": lv_ef,
        "rv_edv_ml": rv_edv,
        "rv_esv_ml": rv_esv,
        "rv_ef_pct": rv_ef,
        "myo_mass_g": myo_mass,
    }

    if height_cm is not None and height_cm > 0:
        metrics["lv_edv_indexed"] = lv_edv / height_cm
        metrics["myo_mass_indexed"] = myo_mass / height_cm
    else:
        metrics["lv_edv_indexed"] = np.nan
        metrics["myo_mass_indexed"] = np.nan

    return metrics


def compute_clinical_agreement(
    comparison_df: pd.DataFrame,
    pairs: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compute MAE and Pearson correlation between predicted and ground-truth clinical indices.

    Args:
        comparison_df: DataFrame containing paired ground-truth and predicted columns.
        pairs: Dictionary mapping prediction column name to ground truth column name.

    Returns:
        Summary DataFrame with MAE and Pearson r per metric.
    """
    if pairs is None:
        pairs = {
            "lv_ef_pct_pred": "lv_ef_pct",
            "rv_ef_pct_pred": "rv_ef_pct",
            "myo_mass_g_pred": "myo_mass_g",
        }

    records = []
    for pred_col, gt_col in pairs.items():
        if pred_col not in comparison_df or gt_col not in comparison_df:
            continue
        valid_data = comparison_df[[pred_col, gt_col]].dropna()
        if len(valid_data) < 2:
            continue

        mae = float((valid_data[pred_col] - valid_data[gt_col]).abs().mean())
        rmse = float(np.sqrt(((valid_data[pred_col] - valid_data[gt_col]) ** 2).mean()))
        corr = float(valid_data[[pred_col, gt_col]].corr().iloc[0, 1])

        records.append({
            "metric": gt_col,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "pearson_r": round(corr, 3),
            "n_samples": len(valid_data),
        })

    return pd.DataFrame(records)
