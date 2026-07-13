"""BX/BY center stability analysis across frames within each disc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import GT_COLUMNS, fiji_csv_to_ellipse_params, load_ground_truth, list_disc_ids


def _step_stats(values: pd.Series) -> dict[str, float]:
    steps = values.diff().abs().dropna()
    if steps.empty:
        return {
            "max_step": 0.0,
            "mean_step": 0.0,
            "median_step": 0.0,
            "total_range": 0.0,
            "n_frames": len(values),
        }
    return {
        "max_step": float(steps.max()),
        "mean_step": float(steps.mean()),
        "median_step": float(steps.median()),
        "total_range": float(values.max() - values.min()),
        "n_frames": len(values),
    }


def center_stability_from_series(
    frames: pd.Series,
    bx: pd.Series,
    by: pd.Series,
    disc_id: str,
    source: str,
) -> list[dict[str, float | str | int]]:
    order = frames.argsort()
    bx = bx.iloc[order].reset_index(drop=True)
    by = by.iloc[order].reset_index(drop=True)
    bx_stats = _step_stats(bx)
    by_stats = _step_stats(by)
    return [
        {"disc_id": disc_id, "source": source, "param": "BX", **bx_stats},
        {"disc_id": disc_id, "source": source, "param": "BY", **by_stats},
    ]


def center_stability_from_csv(csv_path: Path | str, disc_id: str | None = None) -> pd.DataFrame:
    csv_path = Path(csv_path)
    disc_id = disc_id or csv_path.stem
    df = load_ground_truth(csv_path).sort_values("frame")
    centers = df.apply(fiji_csv_to_ellipse_params, axis=1)
    bx = centers.apply(lambda p: p["BX"])
    by = centers.apply(lambda p: p["BY"])
    rows = center_stability_from_series(df["frame"], bx, by, disc_id, "ground_truth")
    return pd.DataFrame(rows)


def center_stability_from_eval(
    eval_df: pd.DataFrame,
    source: str = "prediction",
) -> pd.DataFrame:
    prefix = "gt_" if source == "ground_truth" else "pred_"
    rows: list[dict[str, float | str | int]] = []
    for disc_id, group in eval_df.groupby("disc_id"):
        group = group.sort_values("frame")
        if source == "prediction":
            group = group[group["detected"]]
            if group.empty:
                continue
        rows.extend(
            center_stability_from_series(
                group["frame"],
                group[f"{prefix}BX"],
                group[f"{prefix}BY"],
                disc_id,
                source,
            )
        )
    return pd.DataFrame(rows)


def summarize_all_discs(
    data_dir: Path | str,
    eval_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ground-truth stability from Fiji CSVs, plus prediction stability when eval_df is given."""
    rows: list[dict[str, float | str | int]] = []
    for disc_id in list_disc_ids(data_dir):
        rows.extend(center_stability_from_csv(Path(data_dir) / f"{disc_id}.csv", disc_id).to_dict("records"))
    if eval_df is not None:
        rows.extend(center_stability_from_eval(eval_df, "prediction").to_dict("records"))
    return pd.DataFrame(rows)


def flag_unstable_centers(
    stability_df: pd.DataFrame,
    max_step_threshold: float = 5.0,
    range_threshold: float = 10.0,
) -> pd.DataFrame:
    """Return rows where frame-to-frame step or total range exceeds thresholds."""
    unstable = stability_df[
        (stability_df["max_step"] > max_step_threshold)
        | (stability_df["total_range"] > range_threshold)
    ].copy()
    return unstable.sort_values(["source", "max_step"], ascending=[True, False])


def print_stability_report(
    stability_df: pd.DataFrame,
    max_step_threshold: float = 5.0,
    range_threshold: float = 10.0,
) -> None:
    print("\nBX/BY center stability (pixels)")
    print("=" * 40)
    for source in stability_df["source"].unique():
        subset = stability_df[stability_df["source"] == source]
        print(f"\n{source}:")
        for param in ("BX", "BY"):
            param_df = subset[subset["param"] == param]
            print(
                f"  {param}  max_step: median={param_df['max_step'].median():.1f}  "
                f"max={param_df['max_step'].max():.1f}  "
                f"total_range: median={param_df['total_range'].median():.1f}  "
                f"max={param_df['total_range'].max():.1f}"
            )

    flagged = flag_unstable_centers(stability_df, max_step_threshold, range_threshold)
    print(
        f"\nFlagged (max_step > {max_step_threshold} px or "
        f"total_range > {range_threshold} px): {len(flagged)} disc/param pairs"
    )
    if not flagged.empty:
        cols = ["disc_id", "source", "param", "max_step", "mean_step", "total_range", "n_frames"]
        print(flagged[cols].to_string(index=False))


def pivot_stability_wide(stability_df: pd.DataFrame) -> pd.DataFrame:
    """One row per disc with BX/BY max_step and total_range for each source."""
    rows = []
    for (disc_id, source), group in stability_df.groupby(["disc_id", "source"]):
        entry: dict[str, float | str | int] = {"disc_id": disc_id, "source": source}
        for _, row in group.iterrows():
            param = row["param"]
            entry[f"{param}_max_step"] = row["max_step"]
            entry[f"{param}_mean_step"] = row["mean_step"]
            entry[f"{param}_total_range"] = row["total_range"]
        entry["n_frames"] = int(group["n_frames"].iloc[0])
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(["disc_id", "source"])
