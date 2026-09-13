#!/usr/bin/env python3
"""CLI script to train Random Forest Diagnosis Classifier using clinical metrics derived from masks."""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.io import load_nifti
from src.data.parser import collect_dataset_metadata
from src.metrics.clinical import extract_clinical_features_from_mask
from src.models.classifier import CardiacDiagnosisClassifier
from src.visualization.plotting import plot_confusion_matrix_heatmap, plot_feature_importances

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_classifier")


def extract_features_from_dataset(patients_df: pd.DataFrame) -> pd.DataFrame:
    """Extract clinical features for all patients with available ground truth."""
    records = []
    for _, row in patients_df.iterrows():
        if row.get("ed_gt") is None or row.get("es_gt") is None:
            continue

        ed_gt, ed_spacing = load_nifti(row["ed_gt"])
        es_gt, es_spacing = load_nifti(row["es_gt"])

        feats = extract_clinical_features_from_mask(
            ed_mask=ed_gt,
            ed_spacing=ed_spacing,
            es_mask=es_gt,
            es_spacing=es_spacing,
            height_cm=row.get("height_cm"),
            weight_kg=row.get("weight_kg"),
        )
        feats["patient_id"] = row["patient_id"]
        feats["group"] = row.get("group")
        feats["split"] = row.get("split")
        feats["height_cm"] = row.get("height_cm")
        feats["weight_kg"] = row.get("weight_kg")
        records.append(feats)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clinical metrics, run 5-fold CV, and train Random Forest Classifier."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to ACDC dataset root containing training/ and testing/ folders.",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=None,
        help="Path to pre-computed clinical features CSV (optional, skips feature extraction).",
    )
    parser.add_argument(
        "--save-features-csv",
        type=str,
        default="./data/ground_truth_features.csv",
        help="Path to save extracted clinical features CSV.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="./models/cardiac_rf_model.pkl",
        help="Path to save trained classifier model.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="./assets",
        help="Directory to save output plots (confusion matrix, feature importances).",
    )

    args = parser.parse_args()
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    if args.features_csv and Path(args.features_csv).exists():
        logger.info("Loading pre-computed clinical features from %s ...", args.features_csv)
        features_df = pd.read_csv(args.features_csv)
    else:
        if not args.dataset_root:
            logger.error("Must provide either --dataset-root or an existing --features-csv.")
            sys.exit(1)
        root = Path(args.dataset_root)
        logger.info("Scanning dataset at %s ...", root)
        patients_df = collect_dataset_metadata(root)
        train_patients = patients_df[patients_df["split"] == "training"]
        logger.info("Extracting volumetric clinical features from %d training cases ...", len(train_patients))
        features_df = extract_features_from_dataset(train_patients)

        if args.save_features_csv:
            out_p = Path(args.save_features_csv)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(out_p, index=False)
            logger.info("Saved extracted features to %s", out_p)

    # Initialize Classifier
    classifier = CardiacDiagnosisClassifier()

    # 5-fold Cross-Validation
    logger.info("Running %d-fold Stratified Cross-Validation on %d patients ...", args.cv_folds, len(features_df))
    cv_results = classifier.evaluate_cv(features_df, n_splits=args.cv_folds)

    print("\n" + "=" * 55)
    print(f"  {args.cv_folds}-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
    print("=" * 55)
    print(f"  Overall CV Accuracy: {cv_results['accuracy']:.3f}\n")
    print(cv_results["report_str"])
    print("=" * 55 + "\n")

    # Plot Confusion Matrix
    cm_plot_path = assets_dir / "stats_classification_cv.png"
    plot_confusion_matrix_heatmap(
        cm=cv_results["confusion_matrix"],
        classes=cv_results["classes"],
        accuracy=cv_results["accuracy"],
        output_path=cm_plot_path,
        title=f"Confusion Matrix ({args.cv_folds}-Fold Cross-Validation)",
    )

    # Fit full model
    logger.info("Fitting final classifier on all training samples ...")
    classifier.fit(features_df)

    # Plot feature importances
    imp_df = classifier.get_feature_importances()
    print("\nFeature Importances:")
    print(imp_df.to_string(index=False))
    imp_plot_path = assets_dir / "feature_importances.png"
    plot_feature_importances(imp_df, output_path=imp_plot_path)

    # Save trained model
    if args.output_model:
        model_out = Path(args.output_model)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        with open(model_out, "wb") as f:
            pickle.dump(classifier, f)
        logger.info("Saved trained model to %s", model_out)


if __name__ == "__main__":
    main()
