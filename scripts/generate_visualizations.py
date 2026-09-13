#!/usr/bin/env python3
"""CLI script to generate paper-ready visualizations, GIFs, and overlay figures."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.io import load_nifti
from src.metrics.segmentation import compute_dice_score
from src.visualization.plotting import (
    generate_slice_loop_gif,
    plot_case_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_visualizations")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate medical slice overlays and animated slice-by-slice GIFs."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing original images (e.g. ACDC_patient101_ED_0000.nii.gz).",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=str,
        default=None,
        help="Directory containing ground-truth masks.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory containing predicted masks.",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        required=True,
        help="Patient identifier (e.g., patient101 or patient117).",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="ed",
        choices=["ed", "es", "both"],
        help="Cardiac phase to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./assets",
        help="Destination directory for generated figures and GIFs.",
    )
    parser.add_argument(
        "--make-gif",
        action="store_true",
        help="Whether to render animated GIF looping through all slices.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Animation speed for GIFs in frames per second.",
    )

    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    gt_dir = Path(args.ground_truth_dir) if args.ground_truth_dir else None
    pred_dir = Path(args.predictions_dir) if args.predictions_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = ["ed", "es"] if args.phase == "both" else [args.phase.lower()]

    for ph in phases:
        case_name = f"ACDC_{args.patient_id}_{ph.upper()}"
        img_file = img_dir / f"{case_name}_0000.nii.gz"
        if not img_file.exists():
            img_file = img_dir / f"{case_name}.nii.gz"
        if not img_file.exists():
            logger.warning("Image file not found for %s in %s", case_name, img_dir)
            continue

        img_vol, _ = load_nifti(img_file)

        gt_vol = None
        if gt_dir:
            gt_file = gt_dir / f"{case_name}.nii.gz"
            if gt_file.exists():
                gt_vol, _ = load_nifti(gt_file)

        pred_vol = None
        if pred_dir:
            pred_file = pred_dir / f"{case_name}.nii.gz"
            if pred_file.exists():
                pred_vol, _ = load_nifti(pred_file)

        # Compute Dice scores if both are present
        dice_scores = {}
        if gt_vol is not None and pred_vol is not None:
            dice_scores = {
                "RV": compute_dice_score(pred_vol, gt_vol, 1),
                "MYO": compute_dice_score(pred_vol, gt_vol, 2),
                "LV": compute_dice_score(pred_vol, gt_vol, 3),
            }

        # 1. Static Case Comparison Photo
        photo_path = out_dir / f"case_photo_{args.patient_id}_{ph.upper()}.png"
        plot_case_comparison(
            img_vol=img_vol,
            gt_vol=gt_vol,
            pred_vol=pred_vol,
            patient_id=args.patient_id,
            phase=ph,
            dice_scores=dice_scores,
            output_path=photo_path,
        )

        # 2. Animated GIF if requested
        if args.make_gif:
            gif_path = out_dir / f"slice_grid_{args.patient_id}_{ph.upper()}.gif"
            generate_slice_loop_gif(
                img_vol=img_vol,
                gt_vol=gt_vol,
                pred_vol=pred_vol,
                patient_id=args.patient_id,
                phase=ph,
                output_path=gif_path,
                fps=args.fps,
                dice_scores=dice_scores,
            )


if __name__ == "__main__":
    main()
