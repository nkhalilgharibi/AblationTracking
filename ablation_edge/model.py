"""sklearn-compatible wrappers for the OpenCV ablation-ring detector."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

from .data import GT_COLUMNS, Sample, load_gray_image
from .detector import AblationEdgeDetector, EllipseResult

# Full OpenCV / Fiji ellipse vector kept for visualization & evaluation.
# Order: BX, BY, Major, Minor, Angle (Angle = clockwise deg from +x).
# OpenCV cv2.fitEllipse angle is converted to this Fiji convention inside
# AblationEdgeDetector (_fit_ellipse_from_points), so predict() Angle is
# already comparable to ground-truth Fiji Angle — no extra conversion here.
ELLIPSE_COLUMNS = list(GT_COLUMNS)
MAJOR_IDX = ELLIPSE_COLUMNS.index("Major")
MINOR_IDX = ELLIPSE_COLUMNS.index("Minor")
ANGLE_IDX = ELLIPSE_COLUMNS.index("Angle")

# Hyperparameters exposed to GridSearch / RandomizedSearch.
# Defaults match AblationEdgeDetector.DEFAULT_PARAMS where applicable.
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "edge_blur": [7, 9, 11],
    "dark_threshold": [16, 18, 19, 20],
    "bright_threshold": [22],
    "use_adaptive_threshold": [False],
    "dark_run_mode": ["first"],
    "min_dark_run": [4, 5],
    "radius_outlier_sigma": [2.7],
    "center_max_area": [40000],
    "r_min": [70],
    "r_max": [200],
    "blur": [11],
    "threshold": [20],
    "morph_kernel": [2],
    "morph_iterations": [4],
    "temporal_alpha": [1.0],
}


class GrayImageLoader(BaseEstimator, TransformerMixin):
    """Load grayscale frames from paths, or pass through already-loaded arrays."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).ravel()
        images = np.empty(len(X), dtype=object)
        for i, item in enumerate(X):
            if isinstance(item, np.ndarray):
                images[i] = item
            else:
                images[i] = load_gray_image(item)
        return images


class AblationEllipseEstimator(BaseEstimator, RegressorMixin):
    """
    OpenCV ring/ellipse detector wrapped as a sklearn regressor.

    ``predict`` returns ``(n_samples, 5)`` columns
    ``[BX, BY, Major, Minor, Angle]`` so overlays / the disc viewer can use
    the full fit. Training score uses all five params (see scorer below).
    Failed detections are filled with ``0`` (misses are heavily penalized).
    """

    def __init__(
        self,
        edge_blur: int = 11,
        dark_threshold: float = 22,
        bright_threshold: float = 22,
        use_adaptive_threshold: bool = False,
        dark_run_mode: str = "first",
        min_dark_run: int = 4,
        radius_outlier_sigma: float = 2.5,
        center_max_area: int = 40000,
        r_min: int = 70,
        r_max: int = 200,
        blur: int = 11,
        threshold: int = 20,
        morph_kernel: int = 2,
        morph_iterations: int = 4,
        temporal_alpha: float = 1.0,
        refine_edges: bool = False,
    ) -> None:
        self.edge_blur = edge_blur
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.dark_run_mode = dark_run_mode
        self.min_dark_run = min_dark_run
        self.radius_outlier_sigma = radius_outlier_sigma
        self.center_max_area = center_max_area
        self.r_min = r_min
        self.r_max = r_max
        self.blur = blur
        self.threshold = threshold
        self.morph_kernel = morph_kernel
        self.morph_iterations = morph_iterations
        self.temporal_alpha = temporal_alpha
        self.refine_edges = refine_edges

    def _detector_params(self) -> dict[str, Any]:
        return {
            "edge_blur": self.edge_blur,
            "dark_threshold": self.dark_threshold,
            "bright_threshold": self.bright_threshold,
            "use_adaptive_threshold": self.use_adaptive_threshold,
            "dark_run_mode": self.dark_run_mode,
            "min_dark_run": self.min_dark_run,
            "radius_outlier_sigma": self.radius_outlier_sigma,
            "center_max_area": self.center_max_area,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "blur": self.blur,
            "threshold": self.threshold,
            "morph_kernel": self.morph_kernel,
            "morph_iterations": self.morph_iterations,
            "temporal_alpha": self.temporal_alpha,
        }

    def _make_detector(self) -> AblationEdgeDetector:
        return AblationEdgeDetector(refine_edges=self.refine_edges, **self._detector_params())

    def fit(self, X, y=None):
        """No learned weights; validates input and marks the estimator fitted."""
        X = np.asarray(X, dtype=object).ravel()
        if len(X) == 0:
            raise ValueError("AblationEllipseEstimator.fit requires at least one sample.")
        if y is not None:
            y = np.asarray(y, dtype=np.float64)
            if y.ndim != 2 or y.shape[1] != len(ELLIPSE_COLUMNS):
                raise ValueError(
                    f"y must have shape (n_samples, {len(ELLIPSE_COLUMNS)}) "
                    f"for {ELLIPSE_COLUMNS}."
                )
            if len(y) != len(X):
                raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")
        self.n_features_in_ = 1
        self.n_outputs_ = len(ELLIPSE_COLUMNS)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before predict.")
        images = np.asarray(X, dtype=object).ravel()
        out = np.zeros((len(images), len(ELLIPSE_COLUMNS)), dtype=np.float64)
        detector = self._make_detector()
        for i, image in enumerate(images):
            if not isinstance(image, np.ndarray):
                image = load_gray_image(image)
            result = detector.detect(image)
            if result is not None:
                out[i, 0] = float(result.BX)
                out[i, 1] = float(result.BY)
                out[i, 2] = float(result.Major)
                out[i, 3] = float(result.Minor)
                out[i, 4] = float(result.Angle)
        return out

    def predict_ellipse(self, image: np.ndarray) -> EllipseResult | None:
        """Full Fiji-format ellipse for a single image (inference helper)."""
        if not getattr(self, "is_fitted_", False):
            self.fit([image])
        return self._make_detector().detect(image)

    def to_detector(self) -> AblationEdgeDetector:
        """Materialize the underlying OpenCV detector with current params."""
        return self._make_detector()


def _major_minor_columns(y: np.ndarray) -> np.ndarray:
    """Extract Major/Minor from a full ``(n, 5)`` ellipse matrix (or pass through ``(n, 2)``)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {y.shape}")
    if y.shape[1] == len(ELLIPSE_COLUMNS):
        return y[:, [MAJOR_IDX, MINOR_IDX]]
    if y.shape[1] == 2:
        return y
    raise ValueError(
        f"y must have 2 or {len(ELLIPSE_COLUMNS)} columns, got {y.shape[1]}"
    )


# Geometry residuals are divided by this before squaring so they sit on a
# similar scale to the sin/cos(2θ) angle terms (all frames are 512×512).
IMAGE_SIZE_NORM = 512.0


# --- Previous cost (Major/Minor only). Keep for easy rollback. ---
# def major_minor_quadratic_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """
#     Mean quadratic loss on Major/Minor only:
#     ``mean( (dMajor)^2 + (dMinor)^2 )``.
#
#     Accepts full ``(n, 5)`` ellipse vectors; BX, BY, Angle are ignored for the loss.
#     """
#     major_minor_true = _major_minor_columns(y_true)
#     major_minor_pred = _major_minor_columns(y_pred)
#     if major_minor_true.shape != major_minor_pred.shape:
#         raise ValueError(
#             f"Shape mismatch after Major/Minor extract: "
#             f"{major_minor_true.shape} vs {major_minor_pred.shape}"
#         )
#     residuals = major_minor_pred - major_minor_true
#     return float(np.mean(np.sum(residuals**2, axis=1)))
#
#
# def major_minor_quadratic_scorer():
#     """sklearn scorer: greater is better → negative Major/Minor quadratic loss."""
#     return make_scorer(
#         lambda y_true, y_pred: -major_minor_quadratic_loss(y_true, y_pred),
#         greater_is_better=True,
#     )


def ellipse_quadratic_sincos_angle_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean quadratic loss on full ellipse ``[BX, BY, Major, Minor, Angle]``,
    with geometry normalized by ``IMAGE_SIZE_NORM`` (512) and Angle scored in
    the continuous 180°-period embedding (``sin(2θ)``, ``cos(2θ)``):

    ``mean( (dBX/S)² + (dBY/S)² + (dMajor/S)² + (dMinor/S)²
            + (sin2θ_pred − sin2θ_gt)² + (cos2θ_pred − cos2θ_gt)² )``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != len(ELLIPSE_COLUMNS):
        raise ValueError(
            f"Expected (n, {len(ELLIPSE_COLUMNS)}) ellipse vectors, got {y_true.shape}"
        )

    linear = (y_pred[:, :ANGLE_IDX] - y_true[:, :ANGLE_IDX]) / IMAGE_SIZE_NORM
    linear_sq = np.sum(linear**2, axis=1)

    # Degrees → radians for 2θ embedding (ellipse orientation period 180°).
    theta_pred = np.deg2rad(y_pred[:, ANGLE_IDX])
    theta_true = np.deg2rad(y_true[:, ANGLE_IDX])
    sin_err = np.sin(2.0 * theta_pred) - np.sin(2.0 * theta_true)
    cos_err = np.cos(2.0 * theta_pred) - np.cos(2.0 * theta_true)
    angle_sq = sin_err**2 + cos_err**2

    return float(np.mean(linear_sq + angle_sq))


def ellipse_quadratic_sincos_angle_scorer():
    """sklearn scorer: greater is better → negative sincos-angle quadratic loss."""
    return make_scorer(
        lambda y_true, y_pred: -ellipse_quadratic_sincos_angle_loss(y_true, y_pred),
        greater_is_better=True,
    )


def row_to_ellipse_dict(row: np.ndarray) -> dict[str, float]:
    """Map a ``(5,)`` predict row to Fiji-style ellipse params."""
    values = np.asarray(row, dtype=np.float64).ravel()
    if values.shape[0] != len(ELLIPSE_COLUMNS):
        raise ValueError(f"Expected {len(ELLIPSE_COLUMNS)} values, got {values.shape[0]}")
    return {name: float(values[i]) for i, name in enumerate(ELLIPSE_COLUMNS)}


def make_detection_pipeline(**estimator_params: Any) -> Pipeline:
    """Image-path → gray load → full ellipse estimator (BX, BY, Major, Minor, Angle)."""
    return Pipeline(
        steps=[
            ("load", GrayImageLoader()),
            ("detect", AblationEllipseEstimator(**estimator_params)),
        ]
    )


def param_grid_for_pipeline(
    search_space: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    """Prefix search keys for a ``load | detect`` pipeline."""
    space = search_space or DEFAULT_PARAM_GRID
    return {f"detect__{key}": values for key, values in space.items()}


def samples_to_xy(
    samples: list[Sample],
    *,
    preload: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sklearn ``(X, y, groups)`` from dataset samples.

    ``y`` is ``(n, 5)`` with columns ``[BX, BY, Major, Minor, Angle]``.
    Scoring uses all five (Angle already Fiji-convention from the detector).
    ``groups`` are disc IDs for GroupKFold.
    """
    paths_or_images: list[Any] = []
    targets: list[list[float]] = []
    groups: list[str] = []
    for sample in samples:
        if sample.ground_truth is None:
            continue
        item: Any = load_gray_image(sample.image_path) if preload else sample.image_path
        paths_or_images.append(item)
        gt = sample.ground_truth
        targets.append([float(gt[col]) for col in ELLIPSE_COLUMNS])
        groups.append(sample.disc_id)
    if not paths_or_images:
        raise ValueError("No labeled samples available for training.")
    X = np.empty(len(paths_or_images), dtype=object)
    X[:] = paths_or_images
    y = np.asarray(targets, dtype=np.float64)
    return X, y, np.asarray(groups, dtype=object)


def best_estimator_params(search) -> dict[str, Any]:
    """Pull ``detect__*`` params from a fitted Grid/RandomizedSearchCV."""
    best = search.best_params_
    return {
        key.removeprefix("detect__"): value
        for key, value in best.items()
        if key.startswith("detect__")
    }


SearchMode = Literal["grid", "random"]
