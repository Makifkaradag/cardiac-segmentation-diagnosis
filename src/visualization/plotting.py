"""Comprehensive plotting and medical visualization utilities."""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import imageio
except ImportError:
    imageio = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.config import LABELS, STRUCTURE_COLORS

logger = logging.getLogger(__name__)


def get_best_slice(mask_arr: np.ndarray) -> int:
    """Find the slice index with maximum foreground structure pixels.

    Args:
        mask_arr: 3D integer array of shape (X, Y, Z).

    Returns:
        Index of the slice along Z with the largest segmented area.
    """
    scores = [(mask_arr[:, :, i] > 0).sum() for i in range(mask_arr.shape[2])]
    return int(np.argmax(scores)) if scores else 0


def overlay_structures(
    ax: plt.Axes,
    img_slice: np.ndarray,
    mask_slice: np.ndarray,
    title: str = "",
    alpha: float = 0.55,
    contour: bool = True,
    labels: Optional[Dict[str, int]] = None,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Overlay segmented cardiac structures on a grayscale 2D MRI slice.

    Args:
        ax: Matplotlib axes object.
        img_slice: 2D array of the MRI slice.
        mask_slice: 2D integer array of segmentation mask.
        title: Title for the subplot.
        alpha: Transparency of colored mask overlay.
        contour: Whether to draw structural boundary lines.
        labels: Mapping of structure names to label integers.
        colors: Mapping of structure names to hex colors.
    """
    if labels is None:
        labels = LABELS
    if colors is None:
        colors = STRUCTURE_COLORS

    ax.imshow(img_slice, cmap="gray")

    for name, label_val in labels.items():
        struct_mask = (mask_slice == label_val)
        if struct_mask.sum() > 0:
            colored = np.ma.masked_where(~struct_mask, struct_mask)
            cmap = ListedColormap([colors[name]])
            ax.imshow(colored, cmap=cmap, alpha=alpha)
            if contour:
                ax.contour(struct_mask, levels=[0.5], colors=[colors[name]], linewidths=1.2)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def plot_case_comparison(
    img_vol: np.ndarray,
    gt_vol: Optional[np.ndarray],
    pred_vol: Optional[np.ndarray],
    patient_id: str,
    phase: str,
    dice_scores: Optional[Dict[str, float]] = None,
    slice_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create a 3-panel comparison figure: [Raw MRI | Ground Truth | Prediction].

    Args:
        img_vol: 3D MRI volume.
        gt_vol: Optional 3D ground-truth mask.
        pred_vol: Optional 3D prediction mask.
        patient_id: Patient identifier.
        phase: Cardiac phase ('ED' or 'ES').
        dice_scores: Optional dictionary of Dice scores per structure.
        slice_idx: Slice index to plot (chosen automatically if None).
        output_path: If given, saves the figure to disk.

    Returns:
        Matplotlib Figure.
    """
    if slice_idx is None:
        ref_vol = pred_vol if pred_vol is not None else gt_vol
        slice_idx = get_best_slice(ref_vol) if ref_vol is not None else img_vol.shape[2] // 2

    n_panels = 1 + (1 if gt_vol is not None else 0) + (1 if pred_vol is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    # Panel 1: Raw MRI
    axes[ax_idx].imshow(img_vol[:, :, slice_idx], cmap="gray")
    axes[ax_idx].set_title(f"{patient_id} {phase.upper()} — Raw MRI", fontsize=11, fontweight="bold")
    axes[ax_idx].axis("off")
    ax_idx += 1

    # Panel 2: Ground Truth
    if gt_vol is not None:
        overlay_structures(
            axes[ax_idx],
            img_vol[:, :, slice_idx],
            gt_vol[:, :, slice_idx],
            title="Ground Truth",
        )
        ax_idx += 1

    # Panel 3: Prediction with Dice annotation
    if pred_vol is not None:
        pred_title = "Prediction"
        if dice_scores:
            dice_str = " | ".join(f"{k}:{v:.2f}" for k, v in dice_scores.items())
            pred_title = f"Prediction ({dice_str})"

        overlay_structures(
            axes[ax_idx],
            img_vol[:, :, slice_idx],
            pred_vol[:, :, slice_idx],
            title=pred_title,
        )

    # Global legend
    legend_elements = [
        Patch(facecolor=STRUCTURE_COLORS[n], label=n, alpha=0.7) for n in LABELS
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=len(LABELS),
        bbox_to_anchor=(0.5, 1.05),
        fontsize=11,
        frameon=False,
    )
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("Saved case comparison to %s", out_p)

    return fig


def generate_slice_loop_gif(
    img_vol: np.ndarray,
    gt_vol: Optional[np.ndarray],
    pred_vol: Optional[np.ndarray],
    patient_id: str,
    phase: str,
    output_path: Union[str, Path],
    fps: float = 3.0,
    dice_scores: Optional[Dict[str, float]] = None,
) -> None:
    """Generate an animated GIF traversing every slice of the 3D volume.

    Args:
        img_vol: 3D MRI volume.
        gt_vol: Optional 3D ground-truth mask.
        pred_vol: Optional 3D prediction mask.
        patient_id: Patient identifier.
        phase: Cardiac phase ('ED' or 'ES').
        output_path: Path to save the animated GIF.
        fps: Frames per second.
        dice_scores: Optional Dice score annotation.
    """
    if imageio is None:
        raise ImportError("imageio is required for generating GIFs. Install via: pip install imageio")
    frames = []
    n_slices = img_vol.shape[2]

    for s in range(n_slices):
        fig = plot_case_comparison(
            img_vol=img_vol,
            gt_vol=gt_vol,
            pred_vol=pred_vol,
            patient_id=patient_id,
            phase=phase,
            dice_scores=dice_scores,
            slice_idx=s,
        )
        # Capture figure buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        frame = imageio.v2.imread(buf)
        frames.append(frame)
        plt.close(fig)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 / fps)
    imageio.mimsave(str(out_p), frames, duration=duration, loop=0)
    logger.info("Generated animated slice GIF: %s (%d slices)", out_p, n_slices)


def plot_clinical_scatter(
    comparison_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create correlation scatter plots with y=x identity lines for clinical agreement."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [
        ("lv_ef_pct_pred", "lv_ef_pct", "LV Ejection Fraction (%)"),
        ("rv_ef_pct_pred", "rv_ef_pct", "RV Ejection Fraction (%)"),
        ("myo_mass_g_pred", "myo_mass_g", "Myocardial Mass (g)"),
    ]

    for ax, (pred_col, gt_col, title) in zip(axes, specs):
        if pred_col not in comparison_df or gt_col not in comparison_df:
            continue
        valid = comparison_df[[pred_col, gt_col]].dropna()
        if len(valid) == 0:
            continue

        ax.scatter(valid[gt_col], valid[pred_col], alpha=0.75, color="#2980b9", edgecolors="none", s=40)
        lims = [
            min(valid[gt_col].min(), valid[pred_col].min()),
            max(valid[gt_col].max(), valid[pred_col].max()),
        ]
        ax.plot(lims, lims, "--", color="#c0392b", linewidth=1.5, label="y = x (Perfect agreement)")
        ax.set_xlabel("Ground Truth (Expert)", fontsize=10)
        ax.set_ylabel("Predicted (Model Segmentation)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")

        mae = float((valid[pred_col] - valid[gt_col]).abs().mean())
        corr = float(valid[[pred_col, gt_col]].corr().iloc[0, 1])
        ax.text(
            0.05,
            0.95,
            f"MAE = {mae:.2f}\nPearson r = {corr:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#bdc3c7", alpha=0.85),
            fontsize=10,
        )

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("Saved clinical scatter to %s", out_p)
    return fig


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    classes: List[str],
    accuracy: float,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot confusion matrix as an annotated seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Diagnosis", fontsize=11, fontweight="bold")
    ax.set_title(f"{title}\nAccuracy: {accuracy:.3f}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("Saved confusion matrix heatmap to %s", out_p)
    return fig


def plot_feature_importances(
    importance_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    df = importance_df.sort_values("importance", ascending=True)
    ax.barh(df["feature"], df["importance"], color="#8e44ad", alpha=0.85)
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.set_title("Random Forest Clinical Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importances plot to %s", out_p)
    return fig


def plot_dice_summary_bar(
    dice_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot per-structure mean Dice score with standard deviation error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    names = list(LABELS.keys())
    means = [dice_df[f"dice_{name}"].mean() for name in names]
    stds = [dice_df[f"dice_{name}"].std() for name in names]
    colors = [STRUCTURE_COLORS[name] for name in names]

    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.85, capsize=6, edgecolor="#34495e", linewidth=1.2)
    ax.set_ylabel("Mean Sorensen-Dice Coefficient", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Holdout 3D Segmentation Performance (Mean ± Std)", fontsize=12, fontweight="bold")

    # Add numeric labels above bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m / 2, f"{m:.3f}", ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("Saved Dice summary bar to %s", out_p)
    return fig


def plot_diagnosis_class_gallery(
    gallery_cases: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a multi-panel gallery showcasing one representative case per diagnostic class."""
    n_cases = len(gallery_cases)
    fig, axes = plt.subplots(1, n_cases, figsize=(4.2 * n_cases, 4.8))
    if n_cases == 1:
        axes = [axes]

    for ax, item in zip(axes, gallery_cases):
        img_slice = item["img_slice"]
        mask_slice = item["mask_slice"]
        cls = item["group"]
        title = f"{cls}"

        overlay_structures(ax, img_slice, mask_slice, title=title)

        annotation = f"LVEF: {item.get('lv_ef', 0):.0f}%\nRVEF: {item.get('rv_ef', 0):.0f}%\nMyo: {item.get('myo_mass', 0):.0f}g"
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.65),
        )

    plt.suptitle("Representative Holdout Cases Across Diagnostic Classes", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("Saved class gallery to %s", out_p)
    return fig
