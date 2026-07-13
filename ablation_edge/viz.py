"""Visualization helpers for Fiji ellipse fits vs predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .data import (
    GT_COLUMNS,
    fiji_csv_to_ellipse_params,
    fiji_ellipse_outline,
    load_gray_image,
)
from .detector import EllipseResult


def _center_params(params: dict[str, float] | EllipseResult) -> dict[str, float]:
    """Ellipse center + Major/Minor/Angle; convert Fiji CSV rows if needed."""
    values = params.as_dict() if isinstance(params, EllipseResult) else dict(params)
    if "Width" in values or "Height" in values:
        return fiji_csv_to_ellipse_params(values)
    return {key: float(values[key]) for key in GT_COLUMNS}


def fiji_bbox_patch(
    row: dict[str, float] | pd.Series,
    *,
    edgecolor: str = "yellow",
    linewidth: float = 1.0,
    linestyle: str = "--",
) -> Rectangle:
    """Axis-aligned Fiji CSV bounding box (BX/BY = upper-left corner)."""
    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return Rectangle(
        (float(values["BX"]), float(values["BY"])),
        float(values["Width"]),
        float(values["Height"]),
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def ellipse_outline_patch(
    params: dict[str, float] | EllipseResult,
    *,
    edgecolor: str,
    linewidth: float = 1.5,
) -> Line2D:
    """Closed curve matching Fiji ``EllipseFitter.drawEllipse``."""
    draw = _center_params(params)
    outline = fiji_ellipse_outline(
        draw["BX"],
        draw["BY"],
        draw["Major"],
        draw["Minor"],
        draw["Angle"],
    )
    if len(outline) == 0:
        return Line2D([], [], color=edgecolor, linewidth=linewidth)
    closed = np.vstack([outline, outline[:1]])
    return Line2D(
        closed[:, 0],
        closed[:, 1],
        color=edgecolor,
        linewidth=linewidth,
    )


def update_ellipse_outline_patch(
    line: Line2D,
    params: dict[str, float] | EllipseResult | None,
) -> None:
    if params is None:
        line.set_visible(False)
        return
    draw = _center_params(params)
    outline = fiji_ellipse_outline(
        draw["BX"],
        draw["BY"],
        draw["Major"],
        draw["Minor"],
        draw["Angle"],
    )
    if len(outline) == 0:
        line.set_data([], [])
        line.set_visible(False)
        return
    closed = np.vstack([outline, outline[:1]])
    line.set_data(closed[:, 0], closed[:, 1])
    line.set_visible(True)


def draw_ellipse_overlay(
    image: np.ndarray,
    gt: dict[str, float] | None,
    pred: EllipseResult | dict[str, float] | None,
    gt_color: tuple[int, int, int] = (0, 255, 0),
    pred_color: tuple[int, int, int] = (0, 0, 255),
    line_width: int = 1,
) -> np.ndarray:
    """Draw ground-truth (green) and predicted (red) Fiji ellipses on a grayscale image."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    def _draw(params: dict[str, float] | EllipseResult, color: tuple[int, int, int]) -> None:
        draw = _center_params(params)
        outline = fiji_ellipse_outline(
            draw["BX"],
            draw["BY"],
            draw["Major"],
            draw["Minor"],
            draw["Angle"],
        )
        cv2.polylines(
            canvas,
            [outline.astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

    if gt is not None:
        _draw(gt, gt_color)
    if pred is not None:
        pred_dict = pred.as_dict() if isinstance(pred, EllipseResult) else pred
        _draw(pred_dict, pred_color)
    return canvas


def plot_disc_parameters(
    eval_df: pd.DataFrame,
    disc_id: str,
    output_path: Path | str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Time-series of GT vs predicted ellipse parameters for one disc."""
    disc_df = eval_df[eval_df["disc_id"] == disc_id].sort_values("frame")
    frames = disc_df["frame"].to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()
    fig.suptitle(f"{disc_id}: fitted vs ground truth", fontsize=13)

    for ax, col in zip(axes[:5], GT_COLUMNS):
        ax.plot(frames, disc_df[f"gt_{col}"], "g-o", markersize=3, label="Fiji GT", linewidth=1)
        detected = disc_df["detected"]
        ax.plot(
            frames[detected],
            disc_df.loc[detected, f"pred_{col}"],
            "r-o",
            markersize=3,
            label="Detected",
            linewidth=1,
        )
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    ax = axes[5]
    if "err_BX" in disc_df.columns:
        ax.plot(frames[detected], disc_df.loc[detected, "err_BX"], label="|err BX|", linewidth=1)
        ax.plot(frames[detected], disc_df.loc[detected, "err_BY"], label="|err BY|", linewidth=1)
        ax.set_ylabel("|error| px")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_xlabel("Frame")

    axes[4].set_xlabel("Frame")
    axes[3].set_xlabel("Frame")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_center_drift(
    eval_df: pd.DataFrame,
    disc_id: str,
    output_path: Path | str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Frame-to-frame |ΔBX| and |ΔBY| for GT and predictions."""
    disc_df = eval_df[eval_df["disc_id"] == disc_id].sort_values("frame").copy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{disc_id}: center frame-to-frame change (pixels)", fontsize=13)

    for ax, col in zip(axes, ("BX", "BY")):
        gt = disc_df[f"gt_{col}"].diff().abs()
        pred = disc_df.loc[disc_df["detected"], f"pred_{col}"].diff().abs()
        pred_frames = disc_df.loc[disc_df["detected"], "frame"]
        ax.plot(disc_df["frame"].iloc[1:], gt.iloc[1:], "g-o", markersize=3, label="Fiji GT", linewidth=1)
        ax.plot(pred_frames.iloc[1:], pred.iloc[1:], "r-o", markersize=3, label="Detected", linewidth=1)
        ax.axhline(5.0, color="k", linestyle="--", alpha=0.4, label="5 px")
        ax.set_ylabel(f"|Δ{col}|")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def save_overlay_frames(
    eval_df: pd.DataFrame,
    data_dir: Path | str,
    disc_id: str,
    output_dir: Path | str,
    frame_indices: Iterable[int] | None = None,
    max_frames: int = 6,
) -> list[Path]:
    """Save overlay images for selected frames under output_dir."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    disc_df = eval_df[eval_df["disc_id"] == disc_id].sort_values("frame")
    if frame_indices is None:
        n = len(disc_df)
        if n <= max_frames:
            picks = list(range(n))
        else:
            picks = np.linspace(0, n - 1, max_frames, dtype=int).tolist()
        rows = disc_df.iloc[picks]
    else:
        rows = disc_df[disc_df["frame"].isin(frame_indices)]

    saved: list[Path] = []
    for _, row in rows.iterrows():
        image = load_gray_image(data_dir / f"{disc_id}_frame{int(row['frame']):04d}.tif")
        gt = {col: float(row[f"gt_{col}"]) for col in GT_COLUMNS}
        pred = None
        if row["detected"]:
            pred = {col: float(row[f"pred_{col}"]) for col in GT_COLUMNS}
        overlay = draw_ellipse_overlay(image, gt, pred)
        out_path = output_dir / f"frame_{int(row['frame']):04d}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(out_path)
    return saved
