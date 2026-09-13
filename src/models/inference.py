"""Inference engine supporting both CPU and GPU execution for 3D nnU-Net and diagnosis."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.config import ALL_CLINICAL_FEATURES, LABELS
from src.data.io import load_nifti, save_nifti
from src.metrics.clinical import extract_clinical_features_from_mask
from src.models.classifier import CardiacDiagnosisClassifier

logger = logging.getLogger(__name__)


def get_inference_device(preferred_device: Optional[str] = None) -> torch.device:
    """Determine the optimal compute device (CUDA or CPU).

    Args:
        preferred_device: 'cuda', 'cpu', or None for auto-detection.

    Returns:
        torch.device object.
    """
    if preferred_device:
        device_str = preferred_device.lower()
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_str)

    if torch.cuda.is_available():
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.info("No GPU detected. Running on CPU.")
    return torch.device("cpu")


class CardiacInferencePipeline:
    """End-to-end pipeline: raw patient MRI files -> 3D segmentation -> clinical metrics -> diagnosis."""

    def __init__(
        self,
        nnunet_model_dir: Union[str, Path],
        classifier_model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        checkpoint_name: str = "checkpoint_best.pth",
        fold: int = 0,
    ):
        """Initialize pipeline with model paths and compute device.

        Args:
            nnunet_model_dir: Path to nnU-Net trained model folder containing plans.json and fold_0.
            classifier_model_path: Path to serialized CardiacDiagnosisClassifier (.pkl).
            device: 'cuda', 'cpu', or None for auto-detect.
            checkpoint_name: Checkpoint filename, default 'checkpoint_best.pth'.
            fold: Training fold number, default 0.
        """
        self.model_dir = Path(nnunet_model_dir)
        self.device = get_inference_device(device)
        self.checkpoint_name = checkpoint_name
        self.fold = fold
        self.predictor = None
        self.classifier: Optional[CardiacDiagnosisClassifier] = None

        if classifier_model_path and Path(classifier_model_path).exists():
            import pickle

            with open(classifier_model_path, "rb") as f:
                self.classifier = pickle.load(f)
            logger.info("Loaded diagnosis classifier from %s", classifier_model_path)

    def _init_nnunet_predictor(self) -> None:
        """Lazily initialize nnUNetPredictor with chosen device (CPU or GPU)."""
        if self.predictor is not None:
            return

        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError:
            raise ImportError(
                "nnunetv2 is required for deep learning segmentation. "
                "Install via: pip install nnunetv2"
            )

        logger.info("Initializing nnU-Net predictor on %s ...", self.device)
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True if self.device.type == "cuda" else False,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        self.predictor.initialize_from_trained_model_folder(
            str(self.model_dir),
            use_folds=(self.fold,),
            checkpoint_name=self.checkpoint_name,
        )
        logger.info("Predictor successfully initialized.")

    def run_segmentation(
        self,
        image_path: Union[str, Path],
        output_mask_path: Union[str, Path],
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Run 3D segmentation on a single NIfTI volume.

        Args:
            image_path: Input NIfTI image (.nii or .nii.gz).
            output_mask_path: Target output path for predicted mask.

        Returns:
            Tuple of (predicted integer mask array, voxel spacing).
        """
        self._init_nnunet_predictor()
        img_p = Path(image_path)
        out_p = Path(output_mask_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
            tmp_in_p = Path(tmp_in)
            tmp_out_p = Path(tmp_out)

            # nnU-Net requires channel identifier _0000.nii.gz
            tmp_case = "input_case_0000.nii.gz"
            shutil.copy(img_p, tmp_in_p / tmp_case)

            self.predictor.predict_from_files(
                [[str(tmp_in_p / tmp_case)]],
                [str(tmp_out_p / "input_case.nii.gz")],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1 if self.device.type == "cpu" else 2,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0,
            )

            # Copy result to destination
            res_p = tmp_out_p / "input_case.nii.gz"
            shutil.copy(res_p, out_p)

        pred_mask, spacing = load_nifti(out_p)
        return pred_mask, spacing

    def predict_patient(
        self,
        ed_image_path: Union[str, Path],
        es_image_path: Union[str, Path],
        patient_id: str = "test_patient",
        height_cm: Optional[float] = None,
        weight_kg: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """Run end-to-end diagnosis on a patient's ED and ES cine MRI stacks.

        Args:
            ed_image_path: End-Diastole short-axis NIfTI.
            es_image_path: End-Systole short-axis NIfTI.
            patient_id: Identifier for patient.
            height_cm: Patient height in cm (optional).
            weight_kg: Patient weight in kg (optional).
            output_dir: Directory to store generated 3D masks.

        Returns:
            Dictionary with clinical volumetric metrics and predicted diagnosis.
        """
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            ed_mask_p = out_p / f"{patient_id}_ED_seg.nii.gz"
            es_mask_p = out_p / f"{patient_id}_ES_seg.nii.gz"
        else:
            tmp = tempfile.mkdtemp()
            ed_mask_p = Path(tmp) / "ed_seg.nii.gz"
            es_mask_p = Path(tmp) / "es_seg.nii.gz"

        logger.info("[%s] Segmenting ED volume ...", patient_id)
        ed_mask, ed_spacing = self.run_segmentation(ed_image_path, ed_mask_p)

        logger.info("[%s] Segmenting ES volume ...", patient_id)
        es_mask, es_spacing = self.run_segmentation(es_image_path, es_mask_p)

        logger.info("[%s] Deriving volumetric clinical metrics ...", patient_id)
        metrics = extract_clinical_features_from_mask(
            ed_mask=ed_mask,
            ed_spacing=ed_spacing,
            es_mask=es_mask,
            es_spacing=es_spacing,
            height_cm=height_cm,
            weight_kg=weight_kg,
        )

        result = {
            "patient_id": patient_id,
            "device": str(self.device),
            "ed_mask_path": str(ed_mask_p),
            "es_mask_path": str(es_mask_p),
            "metrics": metrics,
        }

        # Predict diagnosis if classifier is available
        if self.classifier is not None:
            df_sample = pd.DataFrame([metrics])
            pred_diag = self.classifier.predict(df_sample)[0]
            result["predicted_diagnosis"] = pred_diag

            if hasattr(self.classifier, "predict_proba"):
                try:
                    probs = self.classifier.predict_proba(df_sample)[0]
                    classes = self.classifier.classes_
                    result["diagnosis_probabilities"] = dict(zip(classes, [round(float(p), 4) for p in probs]))
                except Exception as exc:
                    logger.debug("Could not compute probabilities: %s", exc)

        return result
