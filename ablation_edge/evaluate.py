"""Evaluation utilities for ellipse detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GT_COLUMNS
from .detector import EllipseResult, _signed_angle_delta, compare_to_ground_truth


def evaluate_predictions(
    rows: list[dict[str, float | int | str | None]],
) -> pd.DataFrame:
    """
    Build a per-frame evaluation table.

    Each row should contain ground-truth columns and either:
    - pred_* columns, or
    - an EllipseResult under the key "prediction".
    """
    normalized: list[dict[str, float | int | str]] = []
    for row in rows:
        entry = {
            "disc_id": row["disc_id"],
            "frame": row["frame"],
        }
        for col in GT_COLUMNS:
            entry[f"gt_{col}"] = float(row["ground_truth"][col])

        prediction = row.get("prediction")
        if isinstance(prediction, EllipseResult):
            pred_dict = prediction.as_dict()
        elif isinstance(prediction, dict) and "BX" in prediction:
            pred_dict = prediction
        else:
            pred_dict = {col: row.get(f"pred_{col}") for col in GT_COLUMNS}

        if pred_dict.get("BX") is None:
            entry["detected"] = False
            normalized.append(entry)
            continue

        entry["detected"] = True
        for col in GT_COLUMNS:
            entry[f"pred_{col}"] = float(pred_dict[col])
            entry[f"err_{col}"] = compare_to_ground_truth(
                {k: float(pred_dict[k]) for k in GT_COLUMNS},
                row["ground_truth"],
            )[col]
        normalized.append(entry)

    return pd.DataFrame(normalized)


def _signed_residuals(detected: pd.DataFrame, col: str) -> np.ndarray:
    """pred − gt; Angle uses shortest signed difference on a 180° period."""
    pred = detected[f"pred_{col}"].to_numpy(dtype=np.float64)
    gt = detected[f"gt_{col}"].to_numpy(dtype=np.float64)
    if col == "Angle":
        return np.asarray(
            [_signed_angle_delta(p, g) for p, g in zip(pred, gt)],
            dtype=np.float64,
        )
    return pred - gt


def summarize_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    detected = eval_df[eval_df["detected"]]
    summary = {
        "frames_total": len(eval_df),
        "frames_detected": len(detected),
        "detection_rate": len(detected) / max(len(eval_df), 1),
    }
    for col in GT_COLUMNS:
        if len(detected) == 0:
            summary[f"mae_{col}"] = np.nan
            summary[f"rmse_{col}"] = np.nan
            summary[f"mean_{col}"] = np.nan
            summary[f"median_{col}"] = np.nan
            continue
        abs_err = detected[f"err_{col}"].to_numpy(dtype=np.float64)
        resid = _signed_residuals(detected, col)
        summary[f"mae_{col}"] = float(np.mean(abs_err))
        summary[f"rmse_{col}"] = float(np.sqrt(np.mean(resid**2)))
        summary[f"mean_{col}"] = float(np.mean(resid))
        summary[f"median_{col}"] = float(np.median(abs_err))
    return pd.DataFrame([summary])


def print_metrics(eval_df: pd.DataFrame, title: str = "Evaluation") -> None:
    summary = summarize_metrics(eval_df).iloc[0]
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"Detected {int(summary['frames_detected'])} / {int(summary['frames_total'])} "
        f"({summary['detection_rate']:.1%})"
    )
    for col in GT_COLUMNS:
        print(
            f"  {col:>5}  MAE={summary[f'mae_{col}']:8.2f}  "
            f"RMSE={summary[f'rmse_{col}']:8.2f}  "
            f"mean={summary[f'mean_{col}']:8.2f}  "
            f"median={summary[f'median_{col}']:8.2f}"
        )


def metrics_by_disc(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disc_id, group in eval_df.groupby("disc_id"):
        summary = summarize_metrics(group).iloc[0].to_dict()
        summary["disc_id"] = disc_id
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("disc_id")
