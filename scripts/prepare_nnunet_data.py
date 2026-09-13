#!/usr/bin/env python3
"""CLI script to parse ACDC dataset and format into nnU-Net v2 raw structure."""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME
from src.data.nnunet_converter import convert_acdc_to_nnunet
from src.data.parser import collect_dataset_metadata

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_nnunet_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and convert ACDC dataset into nnU-Net v2 raw structure."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to extracted ACDC directory containing training/ and testing/ folders.",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to ACDC database.zip file (will be extracted if provided).",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="./data/ACDC_database",
        help="Extraction directory if --zip-path is provided.",
    )
    parser.add_argument(
        "--nnunet-raw",
        type=str,
        default="./data/nnUNet_raw",
        help="Target nnUNet_raw folder.",
    )
    parser.add_argument(
        "--holdout-dir",
        type=str,
        default="./data/acdc_holdout",
        help="Target directory for holdout evaluation cases.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=DEFAULT_DATASET_ID,
        help=f"nnU-Net Dataset ID (default: {DEFAULT_DATASET_ID}).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f"nnU-Net Dataset folder name (default: {DEFAULT_DATASET_NAME}).",
    )

    args = parser.parse_args()

    # Extract zip if requested
    if args.zip_path:
        zip_p = Path(args.zip_path)
        if not zip_p.exists():
            logger.error("Specified zip file does not exist: %s", zip_p)
            sys.exit(1)
        ext_p = Path(args.extract_dir)
        ext_p.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s to %s ...", zip_p, ext_p)
        with zipfile.ZipFile(zip_p, "r") as zf:
            zf.extractall(ext_p)
        dataset_root = ext_p
    elif args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = Path("./data/ACDC_database")

    if not dataset_root.exists():
        logger.error(
            "Dataset root does not exist: %s. Please specify --dataset-root or --zip-path.",
            dataset_root,
        )
        sys.exit(1)

    logger.info("Scanning dataset metadata from %s ...", dataset_root)
    patients_df = collect_dataset_metadata(dataset_root)
    if patients_df.empty:
        logger.error("No patient records found in %s.", dataset_root)
        sys.exit(1)

    nnunet_raw_path = Path(args.nnunet_raw)
    holdout_path = Path(args.holdout_dir) if args.holdout_dir else None

    logger.info("Converting records to nnU-Net v2 raw structure at %s ...", nnunet_raw_path)
    train_cases, holdout_cases = convert_acdc_to_nnunet(
        patients_df=patients_df,
        nnunet_raw_dir=nnunet_raw_path,
        holdout_dir=holdout_path,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
    )

    logger.info("Successfully converted %d training cases and %d holdout cases.", len(train_cases), len(holdout_cases))


if __name__ == "__main__":
    main()
