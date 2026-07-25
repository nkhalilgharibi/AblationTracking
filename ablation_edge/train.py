"""Training utilities: split creation and sklearn hyperparameter search."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from tqdm import tqdm

from .data import (
    DEFAULT_FRAME_MAX,
    DEFAULT_FRAME_MIN,
    AblationDataset,
    load_gray_image,
    load_split,
    save_split,
    split_discs,
    list_disc_ids,
)
from .detector import AblationEdgeDetector
from .evaluate import evaluate_predictions, print_metrics, summarize_metrics
from .model import (
    DEFAULT_PARAM_GRID,
    best_estimator_params,
    ellipse_quadratic_scorer,
    ellipse_quadratic_sincos_angle_scorer,
    make_detection_pipeline,
    param_grid_for_pipeline,
    samples_to_xy,
)

SearchMode = Literal["grid", "random"]
TuningLoss = Literal["circular", "sincos"]

TUNING_LOSS_NAMES: dict[TuningLoss, str] = {
    "circular": "mean_quadratic_ellipse_circular_angle_norm512",
    "sincos": "mean_quadratic_ellipse_sincos_angle_norm512",
}


def _format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``1h 02m 03.4s`` or ``12.3s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {sec:.1f}s"


def create_train_test_split(
    data_dir: Path | str,
    output_path: Path | str = "splits/train_test_split.json",
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    disc_ids = list_disc_ids(data_dir)
    train_ids, test_ids = split_discs(disc_ids, test_fraction=test_fraction, seed=seed)
    save_split(train_ids, test_ids, output_path)
    return train_ids, test_ids


def tune_detector(
    data_dir: Path | str,
    train_disc_ids: list[str],
    max_samples: int | None = None,
    frame_min: int | None = DEFAULT_FRAME_MIN,
    frame_max: int | None = DEFAULT_FRAME_MAX,
    *,
    frame_stride: int = 5,
    cv_folds: int = 3,
    search: SearchMode = "grid",
    n_iter: int = 24,
    search_space: dict[str, list[Any]] | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    verbose: int = 3,
    tuning_loss: TuningLoss = "circular",
) -> tuple[dict[str, Any], Any]:
    """
    Tune ring-fit hyperparameters with sklearn Grid/RandomizedSearch + GroupKFold.

    Predictions and labels store the full ellipse ``[BX, BY, Major, Minor, Angle]``
    (Angle = Fiji clockwise degrees from +x; OpenCV angle is converted upstream).
    Scoring uses all five params. Angle term is either circular ``dAngle²``
    (``tuning_loss='circular'``) or the hybrid-style quadratic embedding
    ``(sin2θ_pred−sin2θ_gt)²+(cos2θ_pred−cos2θ_gt)²`` (``tuning_loss='sincos'``).

    By default only every ``frame_stride``-th frame in ``[frame_min, frame_max]``
    is used (e.g. 6, 11, 16, …, 41 when min=6, max=45, stride=5).
    """
    dataset = AblationDataset(data_dir, train_disc_ids, frame_min=frame_min, frame_max=frame_max)
    samples = list(dataset)
    print(f"Labeled frames in window {frame_min}–{frame_max}: {len(samples)}")

    if frame_stride is not None and frame_stride > 1:
        base = int(frame_min) if frame_min is not None else 0
        before = len(samples)
        samples = [
            sample
            for sample in samples
            if (int(sample.frame) - base) % int(frame_stride) == 0
        ]
        print(
            f"Keeping every {frame_stride}-th frame "
            f"(from {frame_min}): {before} → {len(samples)} samples"
        )

    print(f"max_samples cap: {max_samples}")
    if max_samples is not None and len(samples) > max_samples:
        rng = np.random.default_rng(random_state)
        rng.shuffle(samples)
        samples = samples[:max_samples]
        print(f"After random cap: {len(samples)} samples")

    X, y, groups = samples_to_xy(samples, preload=True)
    print(f"Tuning training samples (labeled frames): {len(X)}")
    n_groups = len(np.unique(groups))
    print(f"Tuning discs (GroupKFold groups): {n_groups}")
    n_splits = max(2, min(cv_folds, n_groups))
    cv = GroupKFold(n_splits=n_splits)

    pipeline = make_detection_pipeline()
    param_grid = param_grid_for_pipeline(search_space or DEFAULT_PARAM_GRID)
    if tuning_loss == "sincos":
        scorer = ellipse_quadratic_sincos_angle_scorer()
        angle_term = "sincos(2θ) embedding residual"
    else:
        scorer = ellipse_quadratic_scorer()
        angle_term = "circular dAngle²"

    print(
        f"Tuning with {search} search, GroupKFold(k={n_splits}), "
        f"loss={tuning_loss} ({angle_term}), "
        f"{len(X)} samples across {n_groups} discs..."
    )

    if search == "random":
        search_cv = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        search_cv = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    search_cv.fit(X, y, groups=groups)

    best_params = {
        **AblationEdgeDetector.DEFAULT_PARAMS,
        **best_estimator_params(search_cv),
    }
    best_params["tuning_score"] = float(-search_cv.best_score_)  # positive quadratic loss
    best_params["cv_folds"] = n_splits
    best_params["search"] = search
    best_params["tuning_loss"] = TUNING_LOSS_NAMES[tuning_loss]

    print(f"Best CV quadratic loss (BX, BY, Major, Minor, Angle): {best_params['tuning_score']:.4f}")
    print(f"Best params: {best_estimator_params(search_cv)}")
    return best_params, search_cv


def run_evaluation(
    data_dir: Path | str,
    disc_ids: list[str],
    detector: AblationEdgeDetector,
    label: str,
    max_samples: int | None = None,
    frame_min: int | None = DEFAULT_FRAME_MIN,
    frame_max: int | None = DEFAULT_FRAME_MAX,
) -> tuple[Any, Any]:
    rows = []
    dataset = AblationDataset(data_dir, disc_ids, frame_min=frame_min, frame_max=frame_max)
    samples = list(dataset)
    if max_samples is not None and len(samples) > max_samples:
        rng = np.random.default_rng(1)
        rng.shuffle(samples)
        samples = samples[:max_samples]
    for sample in tqdm(samples, desc=f"Evaluating {label}"):
        image = load_gray_image(sample.image_path)
        prediction = detector.detect(image)
        rows.append(
            {
                "disc_id": sample.disc_id,
                "frame": sample.frame,
                "ground_truth": sample.ground_truth,
                "prediction": prediction,
            }
        )
    eval_df = evaluate_predictions(rows)
    print_metrics(eval_df, title=label)
    return eval_df, summarize_metrics(eval_df)


def train_pipeline(
    data_dir: Path | str = "Data",
    split_path: Path | str = "splits/train_test_split.json",
    tune_sample_limit: int | None = None,
    frame_min: int | None = DEFAULT_FRAME_MIN,
    frame_max: int | None = DEFAULT_FRAME_MAX,
    *,
    tune_frame_stride: int = 5,
    cv_folds: int = 3,
    search: SearchMode = "grid",
    n_iter: int = 24,
    n_jobs: int = 1,
    verbose: int = 3,
    tuning_loss: TuningLoss = "circular",
    model_out: Path | str | None = None,
) -> tuple[AblationEdgeDetector, dict[str, Any]]:
    """
    Split discs → tune OpenCV detector with sklearn search + GroupKFold → evaluate → save.

    Residual correction is intentionally omitted for now.
    Hyperparameter search defaults to every ``tune_frame_stride``-th frame
    in ``[frame_min, frame_max]`` (e.g. 6, 11, …, 41).

    ``tuning_loss`` selects the GridSearch angle term (``circular`` or ``sincos``).
    ``model_out`` defaults to ``splits/detector_model.json`` (or
    ``detector_model_sincos.json`` when ``tuning_loss='sincos'``).
    """
    pipeline_t0 = time.perf_counter()
    data_dir = Path(data_dir)
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        train_ids, test_ids = load_split(split_path)
    else:
        train_ids, test_ids = create_train_test_split(data_dir, split_path)

    print(f"Train discs: {len(train_ids)}, test discs: {len(test_ids)}")
    print(
        f"Using frames {frame_min}–{frame_max} "
        f"(tune stride={tune_frame_stride}, tuning_loss={tuning_loss})"
    )

    tune_t0 = time.perf_counter()
    tuned_params, _search_cv = tune_detector(
        data_dir,
        train_ids,
        max_samples=tune_sample_limit,
        frame_min=frame_min,
        frame_max=frame_max,
        frame_stride=tune_frame_stride,
        cv_folds=cv_folds,
        search=search,
        n_iter=n_iter,
        n_jobs=n_jobs,
        verbose=verbose,
        tuning_loss=tuning_loss,
    )
    tune_elapsed = time.perf_counter() - tune_t0
    print(f"Hyperparameter search finished in {_format_duration(tune_elapsed)}")

    detector_kwargs = {
        key: value
        for key, value in tuned_params.items()
        if key in AblationEdgeDetector.DEFAULT_PARAMS
    }
    detector = AblationEdgeDetector(**detector_kwargs)

    run_evaluation(
        data_dir,
        train_ids,
        detector,
        label="Train set",
        max_samples=None, #500,
        frame_min=frame_min,
        frame_max=frame_max,
    )
    run_evaluation(
        data_dir,
        test_ids,
        detector,
        label="Test set",
        max_samples=None, #500,
        frame_min=frame_min,
        frame_max=frame_max,
    )

    total_elapsed = time.perf_counter() - pipeline_t0
    loss_name = TUNING_LOSS_NAMES[tuning_loss]
    artifact = {
        "params": detector.params,
        "train_discs": train_ids,
        "test_discs": test_ids,
        "frame_min": frame_min,
        "frame_max": frame_max,
        "tune_frame_stride": tune_frame_stride,
        "coordinate_system": "ellipse_center",
        "target": "ablation_ring_outer",
        "ellipse_columns": ["BX", "BY", "Major", "Minor", "Angle"],
        "tuning_targets": ["BX", "BY", "Major", "Minor", "Angle"],
        "tuning_loss": loss_name,
        "image_size_norm": 512.0,
        "tuning_score": tuned_params.get("tuning_score"),
        "cv_folds": tuned_params.get("cv_folds"),
        "search": tuned_params.get("search"),
        "correction_enabled": False,
        "timing_seconds": {
            "hyperparameter_search": round(tune_elapsed, 3),
            "total": round(total_elapsed, 3),
        },
    }
    if model_out is None:
        filename = (
            "detector_model_sincos.json"
            if tuning_loss == "sincos"
            else "detector_model.json"
        )
        artifact_path = split_path.parent / filename
    else:
        artifact_path = Path(model_out)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nSaved model artifact to {artifact_path}")
    print(f"Total training time: {_format_duration(total_elapsed)}")
    return detector, artifact


def load_trained_detector(artifact_path: Path | str) -> AblationEdgeDetector:
    """Load tuned detector params from ``detector_model.json`` (no residual correction)."""
    payload = json.loads(Path(artifact_path).read_text())
    merged = {**AblationEdgeDetector.DEFAULT_PARAMS, **payload.get("params", {})}
    # Older artifacts often stored adaptive dark cuts that fail on dark discs.
    if "use_adaptive_threshold" not in payload.get("params", {}):
        merged["use_adaptive_threshold"] = False
    if "dark_threshold" in payload.get("params", {}) and payload["params"]["dark_threshold"] >= 24:
        merged["dark_threshold"] = min(int(payload["params"]["dark_threshold"]), 22)
    return AblationEdgeDetector(**merged)
