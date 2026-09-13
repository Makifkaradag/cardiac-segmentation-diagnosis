#!/usr/bin/env python3
"""CLI script to evaluate 3D multi-structure segmentation predictions against ground truth."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LABELS
from src.metrics.segmentation import evaluate_segmentations
from src.visualization.plotting import plot_dice_summary_bar

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_segmentation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 3D multi-structure Dice scores for predicted holdout masks."
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Path to folder containing predicted .nii.gz segmentations.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=str,
        required=True,
        help="Path to folder containing ground-truth .nii.gz segmentations.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./assets/stats_dice_per_case.csv",
        help="Path to save per-case Dice metrics CSV.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="./assets/stats_dice_summary.png",
        help="Path to save Dice score bar plot.",
    )

    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    gt_dir = Path(args.ground_truth_dir)

    if not pred_dir.exists():
        logger.error("Predictions directory does not exist: %s", pred_dir)
        sys.exit(1)
    if not gt_dir.exists():
        logger.error("Ground truth directory does not exist: %s", gt_dir)
        sys.exit(1)

    logger.info("Computing Dice coefficients between %s and %s ...", pred_dir, gt_dir)
    results_df = evaluate_segmentations(
        predictions_dir=pred_dir,
        ground_truth_dir=gt_dir,
        labels=LABELS,
    )

    if results_df.empty:
        logger.error("No valid pairs found to evaluate.")
        sys.exit(1)

    # Print summary table
    print("\n" + "=" * 45)
    print("      ACDC SEGMENTATION EVALUATION SUMMARY")
    print("=" * 45)
    for name in LABELS:
        mean_d = results_df[f"dice_{name}"].mean()
        std_d = results_df[f"dice_{name}"].std()
        print(f"  Structure {name:4s} : Mean Dice = {mean_d:.3f} ± {std_d:.3f}")
    overall_mean = results_df["mean_dice"].mean()
    print("-" * 45)
    print(f"  Overall Mean Dice  : {overall_mean:.3f}")
    print("=" * 45 + "\n")

    # Save CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    logger.info("Saved per-case results to %s", out_csv)

    # Save summary plot
    if args.plot_output:
        plot_dice_summary_bar(results_df, output_path=args.plot_output)


if __name__ == "__main__":
    main()
