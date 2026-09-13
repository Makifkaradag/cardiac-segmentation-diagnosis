"""Visualization suite for medical imaging, segmentation overlays, and clinical plots."""

from src.visualization.plotting import (
    get_best_slice,
    overlay_structures,
    plot_case_comparison,
    generate_slice_loop_gif,
    plot_clinical_scatter,
    plot_confusion_matrix_heatmap,
    plot_feature_importances,
    plot_dice_summary_bar,
    plot_diagnosis_class_gallery,
)

__all__ = [
    "get_best_slice",
    "overlay_structures",
    "plot_case_comparison",
    "generate_slice_loop_gif",
    "plot_clinical_scatter",
    "plot_confusion_matrix_heatmap",
    "plot_feature_importances",
    "plot_dice_summary_bar",
    "plot_diagnosis_class_gallery",
]
