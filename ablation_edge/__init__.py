"""Automated annular ablation edge detection and ellipse fitting."""

from .data import AblationDataset, load_ground_truth, split_discs
from .detector import AblationEdgeDetector, EllipseResult
from .evaluate import evaluate_predictions, print_metrics
from .model import (
    AblationEllipseEstimator,
    ELLIPSE_COLUMNS,
    GrayImageLoader,
    make_detection_pipeline,
)

__all__ = [
    "AblationDataset",
    "AblationEdgeDetector",
    "AblationEllipseEstimator",
    "ELLIPSE_COLUMNS",
    "EllipseResult",
    "GrayImageLoader",
    "evaluate_predictions",
    "load_ground_truth",
    "make_detection_pipeline",
    "print_metrics",
    "split_discs",
]
