"""Training utilities: split creation, tuning, and correction-model fitting."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .data import (
    DEFAULT_FRAME_MAX,
    DEFAULT_FRAME_MIN,
    AblationDataset,
    load_gray_image,
    save_split,
    split_discs,
    list_disc_ids,
)
from .detector import AblationEdgeDetector, compare_to_ground_truth
from .evaluate import evaluate_predictions, print_metrics, summarize_metrics


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


def _load_split(split_path: Path | str) -> tuple[list[str], list[str]]:
    payload = json.loads(Path(split_path).read_text())
    return payload["train"], payload["test"]


def tune_detector(
    data_dir: Path | str,
    train_disc_ids: list[str],
    max_samples: int | None = None,
    frame_min: int | None = DEFAULT_FRAME_MIN,
    frame_max: int | None = DEFAULT_FRAME_MAX,
) -> dict[str, Any]:
    """Grid-search ring-fit detector settings on the training discs."""
    dataset = AblationDataset(data_dir, train_disc_ids, frame_min=frame_min, frame_max=frame_max)
    samples = list(dataset)
    if max_samples is not None:
        rng = np.random.default_rng(0)
        rng.shuffle(samples)
        samples = samples[:max_samples]

    search_space = {
        "edge_blur": [9, 11],
        "dark_threshold": [20, 22, 24],
        "bright_threshold": [22],
        "use_adaptive_threshold": [False],
        "dark_run_mode": ["first"],
        "min_dark_run": [4, 6],
        "radius_outlier_sigma": [2.5, 3.0],
        "center_max_area": [40000],
        "r_min": [70],
        "r_max": [200],
        "blur": [11],
        "threshold": [20],
        "morph_kernel": [2],
        "morph_iterations": [4],
        "temporal_alpha": [1.0],
    }

    keys = list(search_space)
    best_score = float("inf")
    best_params = AblationEdgeDetector.DEFAULT_PARAMS.copy()

    for values in product(*(search_space[key] for key in keys)):
        params = dict(zip(keys, values))
        detector = AblationEdgeDetector(**params, refine_edges=False)
        errors: list[float] = []
        for sample in samples:
            image = load_gray_image(sample.image_path)
            pred = detector.detect(image)
            if pred is None or sample.ground_truth is None:
                continue
            err = compare_to_ground_truth(pred.as_dict(), sample.ground_truth)
            errors.append(
                err["BX"]
                + err["BY"]
                + err["Major"] / 3.0
                + err["Minor"] / 3.0
                + err["Angle"] / 5.0
            )
        if errors:
            score = float(np.mean(errors))
            if score < best_score:
                best_score = score
                best_params = {**AblationEdgeDetector.DEFAULT_PARAMS, **params}

    best_params["tuning_score"] = best_score
    return best_params


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
    tune_sample_limit: int = 400,
    correction_sample_limit: int | None = 800,
    frame_min: int | None = DEFAULT_FRAME_MIN,
    frame_max: int | None = DEFAULT_FRAME_MAX,
) -> tuple[AblationEdgeDetector, dict[str, Any]]:
    data_dir = Path(data_dir)
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        train_ids, test_ids = _load_split(split_path)
    else:
        train_ids, test_ids = create_train_test_split(data_dir, split_path)

    print(f"Train discs: {len(train_ids)}, test discs: {len(test_ids)}")
    print(f"Using frames {frame_min}–{frame_max}")
    print("Tuning ablation-ring detector hyperparameters on training set...")
    tuned_params = tune_detector(
        data_dir, train_ids, max_samples=tune_sample_limit, frame_min=frame_min, frame_max=frame_max
    )
    detector = AblationEdgeDetector(**{k: v for k, v in tuned_params.items() if k != "tuning_score"})

    print("\nCollecting training detections for residual correction...")
    train_images: list[Any] = []
    train_gts: list[dict[str, float]] = []
    train_samples = list(
        AblationDataset(data_dir, train_ids, frame_min=frame_min, frame_max=frame_max)
    )
    if correction_sample_limit is not None and len(train_samples) > correction_sample_limit:
        rng = np.random.default_rng(0)
        rng.shuffle(train_samples)
        train_samples = train_samples[:correction_sample_limit]
    for sample in tqdm(train_samples, desc="Train correction"):
        image = load_gray_image(sample.image_path)
        raw = detector._best_raw_detection(image)
        if raw is None or sample.ground_truth is None:
            continue
        train_images.append(image)
        train_gts.append(sample.ground_truth)

    detector.fit_correction(train_images, train_gts)

    run_evaluation(
        data_dir,
        train_ids,
        detector,
        label="Train set (with correction)",
        max_samples=500,
        frame_min=frame_min,
        frame_max=frame_max,
    )
    run_evaluation(
        data_dir,
        test_ids,
        detector,
        label="Test set (with correction)",
        max_samples=500,
        frame_min=frame_min,
        frame_max=frame_max,
    )

    assert detector.correction_model is not None
    ridge = detector.correction_model.named_steps["ridge"]
    scaler = detector.correction_model.named_steps["scaler"]
    artifact = {
        "params": detector.params,
        "train_discs": train_ids,
        "test_discs": test_ids,
        "frame_min": frame_min,
        "frame_max": frame_max,
        "coordinate_system": "ellipse_center",
        "correction_mode": "residual",
        "target": "ablation_ring_outer",
        "correction_features": detector.FEATURE_NAMES,
        "correction_targets": ["dBX", "dBY", "dMajor", "dMinor", "dAngle"],
        "correction_coef": ridge.coef_.tolist(),
        "correction_intercept": ridge.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "tuning_score": tuned_params.get("tuning_score"),
    }
    artifact_path = split_path.parent / "detector_model.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nSaved model artifact to {artifact_path}")
    return detector, artifact


def load_trained_detector(artifact_path: Path | str) -> AblationEdgeDetector:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    payload = json.loads(Path(artifact_path).read_text())
    # Merge saved knobs over current defaults so newly added keys (MAD reject,
    # morph_*, dark_run_mode) are present even for older artifacts.
    merged = {**AblationEdgeDetector.DEFAULT_PARAMS, **payload.get("params", {})}
    # Older artifacts often stored adaptive dark cuts that fail on dark discs.
    if "use_adaptive_threshold" not in payload.get("params", {}):
        merged["use_adaptive_threshold"] = False
    if "dark_threshold" in payload.get("params", {}) and payload["params"]["dark_threshold"] >= 24:
        # Prefer the notebook-tuned cut unless the artifact is freshly retuned.
        merged["dark_threshold"] = min(int(payload["params"]["dark_threshold"]), 22)
    detector = AblationEdgeDetector(**merged)

    # Stale residual models were fit against older, often-wrong raw ellipses and
    # can make a good ring fit much worse. Only load correction when the artifact
    # explicitly opts in with correction_enabled=true (set by a fresh retrain).
    if not payload.get("correction_enabled", False) or "correction_coef" not in payload:
        return detector

    ridge = Ridge(alpha=5.0)
    ridge.coef_ = np.asarray(payload["correction_coef"], dtype=np.float64)
    ridge.intercept_ = np.asarray(payload["correction_intercept"], dtype=np.float64)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    detector.correction_model = Pipeline([("scaler", scaler), ("ridge", ridge)])
    return detector
