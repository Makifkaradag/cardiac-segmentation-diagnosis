"""Random Forest Cardiac Diagnosis Classifier trained on segmentation-derived features."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import ALL_CLINICAL_FEATURES, BASE_CLINICAL_FEATURES, DIAGNOSIS_CLASSES

logger = logging.getLogger(__name__)


class CardiacDiagnosisClassifier:
    """Random Forest Classifier for predicting 5 cardiac pathology groups from volumetric indices.

    Classes:
      - NOR: Normal
      - MINF: Myocardial Infarction
      - DCM: Dilated Cardiomyopathy
      - HCM: Hypertrophic Cardiomyopathy
      - RV: Abnormal Right Ventricle
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        random_state: int = 42,
        class_weight: str = "balanced",
        feature_columns: Optional[List[str]] = None,
    ):
        """Initialize classifier with hyperparameters.

        Args:
            n_estimators: Number of decision trees.
            max_depth: Maximum tree depth.
            random_state: Reproducibility seed.
            class_weight: Class balancing strategy.
            feature_columns: Feature names used for training and prediction.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
        self.feature_columns = feature_columns or list(ALL_CLINICAL_FEATURES)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )
        self.is_fitted: bool = False
        self.classes_: Optional[List[str]] = None

    def prepare_feature_matrix(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """Clean dataframe, compute indexed features if missing, and extract X (and y).

        Args:
            df: Input DataFrame containing clinical metrics.
            is_training: If True, requires 'group' column and drops NaNs.

        Returns:
            Tuple of (feature matrix X, target array y, processed DataFrame).
        """
        data = df.copy()

        # Compute indexed metrics if height is available and indexed columns are absent
        if "height_cm" in data.columns:
            if "lv_edv_indexed" not in data.columns and "lv_edv_ml" in data.columns:
                data["lv_edv_indexed"] = data["lv_edv_ml"] / data["height_cm"]
            if "myo_mass_indexed" not in data.columns and "myo_mass_g" in data.columns:
                data["myo_mass_indexed"] = data["myo_mass_g"] / data["height_cm"]

        # If evaluating prediction features that have '_pred' suffix
        rename_map = {f"{c}_pred": c for c in self.feature_columns if f"{c}_pred" in data.columns}
        if rename_map:
            data = data.rename(columns=rename_map)

        if is_training:
            subset_cols = [c for c in self.feature_columns if c in data.columns] + ["group"]
            data = data.dropna(subset=subset_cols).copy()
            y = data["group"].values
        else:
            y = data["group"].values if "group" in data.columns else None

        # Fill any remaining NaNs in indexed columns with median if needed
        for col in self.feature_columns:
            if col in data.columns and data[col].isna().any():
                data[col] = data[col].fillna(data[col].median())

        missing_feats = [c for c in self.feature_columns if c not in data.columns]
        if missing_feats:
            raise ValueError(f"Missing required feature columns: {missing_feats}")

        X = data[self.feature_columns].values
        return X, y, data

    def evaluate_cv(
        self,
        features_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Perform Stratified K-Fold Cross Validation on training data.

        Args:
            features_df: DataFrame with ground truth clinical features and 'group'.
            n_splits: Number of cross-validation folds.

        Returns:
            Dictionary containing accuracy, classification report, and confusion matrix.
        """
        X, y, _ = self.prepare_feature_matrix(features_df, is_training=True)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )

        y_pred_cv = cross_val_predict(estimator, X, y, cv=cv)
        accuracy = float(accuracy_score(y, y_pred_cv))
        classes = sorted(list(set(y)))
        report = classification_report(y, y_pred_cv, digits=2, output_dict=True)
        report_str = classification_report(y, y_pred_cv, digits=2)
        cm = confusion_matrix(y, y_pred_cv, labels=classes)

        logger.info("5-Fold Cross-Validation Accuracy: %.3f", accuracy)
        return {
            "accuracy": accuracy,
            "y_true": y,
            "y_pred": y_pred_cv,
            "classes": classes,
            "report": report,
            "report_str": report_str,
            "confusion_matrix": cm,
        }

    def fit(self, features_df: pd.DataFrame) -> "CardiacDiagnosisClassifier":
        """Fit the full Random Forest classifier on all available training samples.

        Args:
            features_df: Training DataFrame containing features and diagnosis group.

        Returns:
            self
        """
        X, y, _ = self.prepare_feature_matrix(features_df, is_training=True)
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = list(self.model.classes_)
        logger.info("Fitted classifier on %d samples across %d classes.", len(X), len(self.classes_))
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict diagnosis classes for new cases.

        Args:
            test_df: DataFrame containing required clinical features.

        Returns:
            Numpy array of predicted diagnosis class strings.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted yet. Call .fit() first.")
        X, _, _ = self.prepare_feature_matrix(test_df, is_training=False)
        return self.model.predict(X)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict class probability distributions for new cases.

        Args:
            test_df: DataFrame containing required clinical features.

        Returns:
            Numpy array of shape (N, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted yet. Call .fit() first.")
        X, _, _ = self.prepare_feature_matrix(test_df, is_training=False)
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> pd.DataFrame:
        """Retrieve sorted feature importance scores.

        Returns:
            DataFrame with feature names and importance values.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        df_imp = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp
