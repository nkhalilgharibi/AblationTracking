"""Automated annular ablation edge detection and ellipse fitting."""

from .data import AblationDataset, load_ground_truth, split_discs
from .detector import AblationEdgeDetector, EllipseResult
from .evaluate import evaluate_predictions, print_metrics

__all__ = [
    "AblationDataset",
    "AblationEdgeDetector",
    "EllipseResult",
    "evaluate_predictions",
    "load_ground_truth",
    "print_metrics",
    "split_discs",
]
