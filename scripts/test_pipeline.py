#!/usr/bin/env python3
"""Universal test script: runs nnU-Net 3D segmentation and diagnosis on CPU or GPU.

Supports:
  - Any raw patient MRI (ED and ES phases)
  - Automatic data formatting
  - CPU and CUDA GPU auto-detection or manual selection
  - Clinical metric derivation (EF, volumes, myocardial mass)
  - Automated 5-class cardiac diagnosis prediction
  - Automatic weights download helper from Google Drive
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DIAGNOSIS_DESCRIPTIONS
from src.data.io import load_nifti
from src.data.parser import parse_info_cfg
from src.models.inference import CardiacInferencePipeline, get_inference_device
from src.visualization.plotting import plot_case_comparison

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")

DRIVE_WEIGHTS_URL = "https://drive.google.com/drive/folders/1NN6Qx6nsP55il4r6kfiiKTL7QwS26nWX?usp=sharing"
DRIVE_FOLDER_ID = "1NN6Qx6nsP55il4r6kfiiKTL7QwS26nWX"


def download_weights_from_drive(target_dir: Path) -> None:
    """Download pre-trained weights from Google Drive using gdown."""
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown

        logger.info("Downloading model weights from Google Drive folder (%s) ...", DRIVE_FOLDER_ID)
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=str(target_dir), quiet=False, use_cookies=False)
        logger.info("Weights downloaded successfully to %s", target_dir)
    except ImportError:
        print("\n" + "=" * 65)
        print("Model weights need to be downloaded from Google Drive:")
        print(f"URL: {DRIVE_WEIGHTS_URL}")
        print("\nYou can install 'gdown' to download automatically:")
        print("  pip install gdown")
        print(f"  gdown --folder {DRIVE_WEIGHTS_URL} -O {target_dir}")
        print("Or download manually in your browser and place files into:", target_dir)
        print("=" * 65 + "\n")


def resolve_patient_files(patient_dir: Path) -> dict:
    """Auto-detect ED and ES NIfTI files from a patient directory."""
    cfg_path = patient_dir / "Info.cfg"
    patient_id = patient_dir.name
    ed_img = None
    es_img = None
    height = None
    weight = None

    if cfg_path.exists():
        info = parse_info_cfg(cfg_path)
        ed_f = int(info.get("ED", 1))
        es_f = int(info.get("ES", 1))
        height = float(info.get("Height", 0)) or None
        weight = float(info.get("Weight", 0)) or None

        f_ed = patient_dir / f"{patient_id}_frame{ed_f:02d}.nii.gz"
        f_es = patient_dir / f"{patient_id}_frame{es_f:02d}.nii.gz"
        if f_ed.exists():
            ed_img = f_ed
        if f_es.exists():
            es_img = f_es

    # Fallback search by pattern
    if not ed_img:
        matches = list(patient_dir.glob("*ED*.nii.gz")) or list(patient_dir.glob("*ed*.nii.gz"))
        ed_img = matches[0] if matches else None
    if not es_img:
        matches = list(patient_dir.glob("*ES*.nii.gz")) or list(patient_dir.glob("*es*.nii.gz"))
        es_img = matches[0] if matches else None

    return {
        "patient_id": patient_id,
        "ed_img": ed_img,
        "es_img": es_img,
        "height_cm": height,
        "weight_kg": weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test ACDC segmentation and cardiac diagnosis models on CPU or GPU."
    )
    parser.add_argument(
        "--patient-dir",
        type=str,
        default=None,
        help="Path to patient folder containing NIfTI files (e.g. patient001/).",
    )
    parser.add_argument(
        "--ed-image",
        type=str,
        default=None,
        help="Direct path to End-Diastole NIfTI image (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--es-image",
        type=str,
        default=None,
        help="Direct path to End-Systole NIfTI image (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="test_case",
        help="Patient identifier name.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/nnunet_weights",
        help="Path to nnU-Net model folder (or directory containing checkpoint_best.pth).",
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        default="./models/cardiac_rf_model.pkl",
        help="Path to trained diagnosis classifier (.pkl).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device: 'cuda' for GPU, 'cpu' for CPU (default: auto-detect).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_outputs",
        help="Directory to save generated 3D segmentations and report.",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Automatically download model weights from Google Drive.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Patient height in cm (optional, for body-size indexing).",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="Patient weight in kg (optional).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visual comparison overlay image for the test case.",
    )

    args = parser.parse_args()

    # Step 1: Model weights check
    model_dir = Path(args.model_dir)
    if args.download_weights or not model_dir.exists():
        if not model_dir.exists() or not any(model_dir.iterdir()):
            download_weights_from_drive(model_dir)

    # Step 2: Resolve input files
    ed_path = None
    es_path = None
    patient_id = args.patient_id
    height = args.height
    weight = args.weight

    if args.patient_dir:
        p_dir = Path(args.patient_dir)
        if not p_dir.exists():
            logger.error("Patient directory not found: %s", p_dir)
            sys.exit(1)
        resolved = resolve_patient_files(p_dir)
        ed_path = resolved["ed_img"]
        es_path = resolved["es_img"]
        patient_id = resolved["patient_id"]
        height = height or resolved["height_cm"]
        weight = weight or resolved["weight_kg"]
    else:
        if args.ed_image:
            ed_path = Path(args.ed_image)
        if args.es_image:
            es_path = Path(args.es_image)

    if not ed_path or not Path(ed_path).exists():
        logger.error("ED image missing or not found. Specify --ed-image or a valid --patient-dir.")
        sys.exit(1)
    if not es_path or not Path(es_path).exists():
        logger.error("ES image missing or not found. Specify --es-image or a valid --patient-dir.")
        sys.exit(1)

    # Device announcement
    target_device = get_inference_device(args.device)
    print("\n" + "=" * 60)
    print("        ACDC CARDIAC MODEL INFERENCE & TEST PIPELINE")
    print("=" * 60)
    print(f"  Target Device : {target_device.type.upper()}")
    print(f"  Patient ID    : {patient_id}")
    print(f"  ED Image      : {ed_path}")
    print(f"  ES Image      : {es_path}")
    print(f"  Model Folder  : {model_dir}")
    print("=" * 60 + "\n")

    # Step 3: Run pipeline
    pipeline = CardiacInferencePipeline(
        nnunet_model_dir=model_dir,
        classifier_model_path=args.classifier_path if Path(args.classifier_path).exists() else None,
        device=str(target_device),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.predict_patient(
        ed_image_path=ed_path,
        es_image_path=es_path,
        patient_id=patient_id,
        height_cm=height,
        weight_kg=weight,
        output_dir=out_dir,
    )

    # Step 4: Display Clinical Findings
    m = result["metrics"]
    print("\n" + "-" * 55)
    print(f"  DERIVED CLINICAL METRICS FOR {patient_id.upper()}")
    print("-" * 55)
    print(f"  LV End-Diastolic Volume (EDV) : {m['lv_edv_ml']:.1f} mL")
    print(f"  LV End-Systolic Volume (ESV)  : {m['lv_esv_ml']:.1f} mL")
    print(f"  LV Ejection Fraction (LVEF)   : {m['lv_ef_pct']:.1f} %")
    print(f"  RV End-Diastolic Volume (EDV) : {m['rv_edv_ml']:.1f} mL")
    print(f"  RV End-Systolic Volume (ESV)  : {m['rv_esv_ml']:.1f} mL")
    print(f"  RV Ejection Fraction (RVEF)   : {m['rv_ef_pct']:.1f} %")
    print(f"  Myocardial Mass               : {m['myo_mass_g']:.1f} g")
    print("-" * 55)

    if "predicted_diagnosis" in result:
        diag = result["predicted_diagnosis"]
        desc = DIAGNOSIS_DESCRIPTIONS.get(diag, "")
        print(f"\n  PREDICTED DIAGNOSIS: {diag}")
        print(f"  Clinical Meaning   : {desc}")
        if "diagnosis_probabilities" in result:
            print("\n  Class Confidence Probabilities:")
            for cls_name, prob in result["diagnosis_probabilities"].items():
                bar = "#" * int(prob * 30)
                print(f"    {cls_name:5s} : {prob * 100:5.1f}%  [{bar:<30}]")
    print("=" * 55 + "\n")

    # Step 5: Optional visualization
    if args.visualize:
        vis_out = out_dir / f"{patient_id}_case_comparison.png"
        img_vol, _ = load_nifti(ed_path)
        pred_vol, _ = load_nifti(result["ed_mask_path"])
        plot_case_comparison(
            img_vol=img_vol,
            gt_vol=None,
            pred_vol=pred_vol,
            patient_id=patient_id,
            phase="ED",
            output_path=vis_out,
        )
        print(f"Saved visual overlay to {vis_out}")


if __name__ == "__main__":
    main()
