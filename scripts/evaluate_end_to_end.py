#!/usr/bin/env python3
"""CLI script for end-to-end evaluation: raw predictions -> clinical indices -> diagnosis."""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.io import load_nifti
from src.data.parser import collect_dataset_metadata, parse_info_cfg
from src.metrics.clinical import compute_clinical_agreement, extract_clinical_features_from_mask
from src.models.classifier import CardiacDiagnosisClassifier
from src.visualization.plotting import plot_clinical_scatter, plot_confusion_matrix_heatmap

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_end_to_end")


def extract_features_from_predictions(
    predictions_dir: Path,
    patient_metadata: Dict[str, Dict],
    case_prefix: str = "ACDC",
) -> pd.DataFrame:
    """Extract clinical metrics from predicted NIfTI segmentations."""
    records: List[Dict] = []

    for patient_id, meta in patient_metadata.items():
        ed_file = predictions_dir / f"{case_prefix}_{patient_id}_ED.nii.gz"
        es_file = predictions_dir / f"{case_prefix}_{patient_id}_ES.nii.gz"

        if not (ed_file.exists() and es_file.exists()):
            continue

        ed_pred, ed_spacing = load_nifti(ed_file)
        es_pred, es_spacing = load_nifti(es_file)

        feats = extract_clinical_features_from_mask(
            ed_mask=ed_pred,
            ed_spacing=ed_spacing,
            es_mask=es_pred,
            es_spacing=es_spacing,
            height_cm=meta.get("height_cm"),
            weight_kg=meta.get("weight_kg"),
        )
        feats["patient_id"] = patient_id
        feats["group"] = meta.get("group")
        feats["height_cm"] = meta.get("height_cm")
        feats["weight_kg"] = meta.get("weight_kg")
        records.append(feats)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation: derive clinical features from predictions and evaluate classifier."
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Path to folder containing predicted .nii.gz masks.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to ACDC dataset root for patient metadata and ground-truth comparison.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/cardiac_rf_model.pkl",
        help="Path to saved CardiacDiagnosisClassifier.",
    )
    parser.add_argument(
        "--train-features-csv",
        type=str,
        default=None,
        help="Optional path to training features CSV to fit classifier if model-path is not found.",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="./assets",
        help="Output directory for agreement scatter and holdout confusion matrix.",
    )

    args = parser.parse_args()
    pred_dir = Path(args.predictions_dir)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        logger.error("Predictions directory does not exist: %s", pred_dir)
        sys.exit(1)

    # Load or extract patient metadata
    patient_meta: Dict[str, Dict] = {}
    if args.dataset_root:
        root = Path(args.dataset_root)
        df_meta = collect_dataset_metadata(root)
        for _, row in df_meta.iterrows():
            patient_meta[row["patient_id"]] = row.to_dict()
    else:
        # Infer patient IDs from prediction filenames
        pred_files = list(pred_dir.glob("*.nii.gz"))
        for pf in pred_files:
            parts = pf.name.replace(".nii.gz", "").split("_")
            if len(parts) >= 2:
                pid = parts[1]
                if pid not in patient_meta:
                    patient_meta[pid] = {"patient_id": pid}

    logger.info("Extracting clinical indices from predictions for %d patients ...", len(patient_meta))
    pred_features_df = extract_features_from_predictions(pred_dir, patient_meta)
    if pred_features_df.empty:
        logger.error("No clinical features could be extracted from predictions in %s", pred_dir)
        sys.exit(1)

    logger.info("Extracted features for %d patients from predictions.", len(pred_features_df))

    # Load Classifier
    model_path = Path(args.model_path)
    if model_path.exists():
        logger.info("Loading classifier from %s ...", model_path)
        with open(model_path, "rb") as f:
            classifier: CardiacDiagnosisClassifier = pickle.load(f)
    elif args.train_features_csv and Path(args.train_features_csv).exists():
        logger.info("Fitting classifier from training features %s ...", args.train_features_csv)
        train_df = pd.read_csv(args.train_features_csv)
        classifier = CardiacDiagnosisClassifier()
        classifier.fit(train_df)
    else:
        logger.error("Model file not found at %s and no training features CSV provided.", model_path)
        sys.exit(1)

    # If ground-truth labels are available on holdout, evaluate accuracy
    test_eval_df = pred_features_df[pred_features_df["group"].notna()].copy()
    if not test_eval_df.empty:
        y_true = test_eval_df["group"].values
        y_pred = classifier.predict(test_eval_df)
        acc = float(accuracy_score(y_true, y_pred))
        classes = sorted(list(set(y_true) | set(y_pred)))

        print("\n" + "=" * 55)
        print("  END-TO-END HOLDOUT CLASSIFICATION RESULTS")
        print("  (Raw MRI -> Predicted Mask -> Derived Metrics -> Diagnosis)")
        print("=" * 55)
        print(f"  Holdout Accuracy: {acc:.3f}\n")
        print(classification_report(y_true, y_pred, digits=2))
        print("=" * 55 + "\n")

        # Save holdout confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plot_confusion_matrix_heatmap(
            cm=cm,
            classes=classes,
            accuracy=acc,
            output_path=assets_dir / "stats_classification.png",
            title="End-to-End Holdout Confusion Matrix",
        )
    else:
        logger.info("No ground-truth diagnosis labels found in test set. Generating predictions only:")
        preds = classifier.predict(pred_features_df)
        pred_features_df["predicted_group"] = preds
        print(pred_features_df[["patient_id", "predicted_group"]].to_string(index=False))


if __name__ == "__main__":
    main()
