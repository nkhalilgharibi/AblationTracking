"""Interactive disc/frame viewer for comparing Fiji GT and detected ellipses."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider, TextBox
from tqdm import tqdm

from .data import (
    DEFAULT_FRAME_MAX,
    DEFAULT_FRAME_MIN,
    GT_COLUMNS,
    fiji_csv_to_ellipse_params,
    image_frame_to_csv_frame,
    load_gray_image,
    load_ground_truth,
    list_disc_ids,
)
from .detector import AblationEdgeDetector, EllipseResult, compare_to_ground_truth
from .train import load_trained_detector
from .viz import fiji_bbox_patch, update_ellipse_outline_patch

DEFAULT_HYBRID_MODEL = Path("splits/hybrid_model.pt")


def resolve_disc_id(text: str, available: list[str]) -> str | None:
    """Match user input to a disc ID (full ID, numeric suffix, or partial match)."""
    text = text.strip()
    if not text:
        return None
    if text in available:
        return text
    candidate = text if text.startswith("disc") else f"disc{text}"
    if candidate in available:
        return candidate
    suffix_matches = [disc_id for disc_id in available if disc_id.endswith(text)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        return None
    partial = [disc_id for disc_id in available if text in disc_id]
    if len(partial) == 1:
        return partial[0]
    return None


def format_params(params: dict[str, float] | EllipseResult | None, prefix: str) -> str:
    if params is None:
        return f"{prefix}: (none)"
    values = params.as_dict() if isinstance(params, EllipseResult) else params
    parts = [f"{key}={values[key]:.1f}" for key in GT_COLUMNS]
    return f"{prefix}:  " + "  ".join(parts)


class DiscViewer:
    """Matplotlib UI to scroll through frames and compare GT vs predicted ellipses."""

    def __init__(
        self,
        data_dir: Path | str = "Data",
        model_path: Path | str = "splits/detector_model.json",
        hybrid_model_path: Path | str | None = DEFAULT_HYBRID_MODEL,
        cache_path: Path | str | None = None,
        frame_min: int = DEFAULT_FRAME_MIN,
        frame_max: int = DEFAULT_FRAME_MAX,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        if hybrid_model_path is not None:
            hybrid_model_path = Path(hybrid_model_path)
            if not hybrid_model_path.exists():
                hybrid_model_path = None
        self.hybrid_model_path = hybrid_model_path
        self.cache_path = Path(cache_path) if cache_path else None
        self.frame_min = frame_min
        self.frame_max = frame_max
        self.available_discs = list_disc_ids(self.data_dir)
        self.detector: AblationEdgeDetector | None = None
        self.hybrid_predictor = None

        self.disc_id: str | None = None
        self.frames: list[int] = []
        self.gt_by_frame: dict[int, dict[str, float]] = {}
        self.gt_csv_by_frame: dict[int, dict[str, float]] = {}
        self.pred_by_frame: dict[int, EllipseResult | None] = {}
        self.hybrid_by_frame: dict[int, EllipseResult | None] = {}
        self.frame_idx = 0

        self._build_ui()

    def _build_ui(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        plt.subplots_adjust(bottom=0.28, top=0.9)
        self.fig.canvas.manager.set_window_title("Ablation ellipse viewer")  # type: ignore[union-attr]

        self.image_artist = self.ax.imshow(np.zeros((512, 512), dtype=np.uint8), cmap="gray", vmin=0, vmax=80)
        self.gt_line = Line2D([], [], color="lime", linewidth=1.5, visible=False)
        self.pred_line = Line2D([], [], color="red", linewidth=1.5, visible=False)
        self.hybrid_line = Line2D([], [], color="cyan", linewidth=1.5, visible=False)
        self.ax.add_line(self.gt_line)
        self.ax.add_line(self.pred_line)
        self.ax.add_line(self.hybrid_line)
        self.bbox_patch = fiji_bbox_patch(
            {"BX": 0.0, "BY": 0.0, "Width": 1.0, "Height": 1.0},
            edgecolor="yellow",
            linewidth=1.0,
        )
        self.bbox_patch.set_visible(False)
        self.ax.add_patch(self.bbox_patch)
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        self.title = self.fig.suptitle("Enter a disc ID and click Load", fontsize=12)
        self.info_text = self.fig.text(
            0.5,
            0.17,
            "Green = Fiji GT    Red = classical    Cyan = hybrid (classical+AI)    Yellow = Fiji bbox\n"
            "Use slider or ← → keys to change frame",
            ha="center",
            fontsize=10,
            family="monospace",
        )

        ax_disc = self.fig.add_axes([0.08, 0.08, 0.42, 0.05])
        ax_load = self.fig.add_axes([0.52, 0.08, 0.12, 0.05])
        ax_slider = self.fig.add_axes([0.08, 0.02, 0.84, 0.03])

        self.disc_box = TextBox(ax_disc, "Disc", initial="")
        self.load_btn = Button(ax_load, "Load")
        self.frame_slider = Slider(ax_slider, "Frame", 0, 1, valinit=0, valstep=1, initcolor="none")
        self.frame_slider.set_active(False)

        self.load_btn.on_clicked(self._on_load_clicked)
        self.disc_box.on_submit(self._on_disc_submit)
        self.frame_slider.on_changed(self._on_frame_changed)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _ensure_detector(self) -> AblationEdgeDetector:
        if self.detector is None:
            self.detector = load_trained_detector(self.model_path)
        return self.detector

    def _ensure_hybrid_predictor(self):
        if self.hybrid_model_path is None:
            return None
        if self.hybrid_predictor is None:
            from .hybrid_refine import HybridEllipsePredictor

            self.hybrid_predictor = HybridEllipsePredictor(
                self.hybrid_model_path,
                detector=self._ensure_detector(),
                detector_model_path=self.model_path,
            )
        return self.hybrid_predictor

    def _on_disc_submit(self, _text: str) -> None:
        self._load_disc(self.disc_box.text)

    def _on_load_clicked(self, _event) -> None:
        self._load_disc(self.disc_box.text)

    def _on_frame_changed(self, value: float) -> None:
        if not self.frames:
            return
        self.frame_idx = int(value)
        self._show_current_frame()

    def _on_key(self, event) -> None:
        if not self.frames:
            return
        if event.key in ("right", "up"):
            self._step_frame(1)
        elif event.key in ("left", "down"):
            self._step_frame(-1)
        elif event.key == "home":
            self._set_frame_index(0)
        elif event.key == "end":
            self._set_frame_index(len(self.frames) - 1)

    def _step_frame(self, delta: int) -> None:
        self._set_frame_index(self.frame_idx + delta)

    def _set_frame_index(self, idx: int) -> None:
        if not self.frames:
            return
        idx = max(0, min(len(self.frames) - 1, idx))
        if idx == self.frame_idx:
            return
        self.frame_idx = idx
        self.frame_slider.set_val(idx)
        self._show_current_frame()

    def _load_disc(self, disc_text: str) -> None:
        disc_id = resolve_disc_id(disc_text, self.available_discs)
        if disc_id is None:
            self.title.set_text(f"Could not find disc: {disc_text!r}")
            self.fig.canvas.draw_idle()
            return

        self.title.set_text(f"Loading {disc_id}...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        gt_df = load_ground_truth(self.data_dir / f"{disc_id}.csv").set_index("frame")
        frame_paths = sorted(self.data_dir.glob(f"{disc_id}_frame*.tif"))
        frame_re = re.compile(rf"^{re.escape(disc_id)}_frame(\d+)\.tif$")
        frames: list[int] = []
        for path in frame_paths:
            match = frame_re.match(path.name)
            if match is None:
                continue
            image_frame = int(match.group(1))
            csv_frame = image_frame_to_csv_frame(image_frame)
            if (
                csv_frame in gt_df.index
                and self.frame_min <= image_frame <= self.frame_max
            ):
                frames.append(image_frame)

        if not frames:
            self.title.set_text(f"No labeled frames found for {disc_id}")
            self.fig.canvas.draw_idle()
            return

        gt_by_frame = {
            image_frame: fiji_csv_to_ellipse_params(
                gt_df.loc[image_frame_to_csv_frame(image_frame)]
            )
            for image_frame in frames
        }
        gt_csv_by_frame = {
            image_frame: {
                key: float(gt_df.loc[image_frame_to_csv_frame(image_frame), key])
                for key in ("BX", "BY", "Width", "Height")
            }
            for image_frame in frames
        }
        pred_by_frame = self._load_predictions(disc_id, frames)
        hybrid_by_frame = self._load_hybrid_predictions(disc_id, frames)

        self.disc_id = disc_id
        self.frames = frames
        self.gt_by_frame = gt_by_frame
        self.gt_csv_by_frame = gt_csv_by_frame
        self.pred_by_frame = pred_by_frame
        self.hybrid_by_frame = hybrid_by_frame
        self.frame_idx = 0

        self.frame_slider.eventson = False
        self.frame_slider.valmax = len(frames) - 1
        self.frame_slider.ax.set_xlim(0, max(len(frames) - 1, 1))
        self.frame_slider.set_val(0)
        self.frame_slider.set_active(True)
        self.frame_slider.eventson = True

        self._show_current_frame()

    def _load_predictions(
        self,
        disc_id: str,
        frames: list[int],
    ) -> dict[int, EllipseResult | None]:
        cached = self._predictions_from_cache(disc_id, frames)
        if cached is not None:
            return cached

        detector = self._ensure_detector()
        detector.reset_temporal_state()
        pred_by_frame: dict[int, EllipseResult | None] = {}
        for frame in tqdm(frames, desc=f"Detecting {disc_id}", leave=False):
            image_path = self.data_dir / f"{disc_id}_frame{frame:04d}.tif"
            image = load_gray_image(image_path)
            pred_by_frame[frame] = detector.detect(image, temporal_smooth=True)
        return pred_by_frame

    def _load_hybrid_predictions(
        self,
        disc_id: str,
        frames: list[int],
    ) -> dict[int, EllipseResult | None]:
        predictor = self._ensure_hybrid_predictor()
        if predictor is None:
            return {}
        predictor.reset_temporal_state()
        pred_by_frame: dict[int, EllipseResult | None] = {}
        for frame in tqdm(frames, desc=f"Hybrid {disc_id}", leave=False):
            image = load_gray_image(self.data_dir / f"{disc_id}_frame{frame:04d}.tif")
            pred_by_frame[frame] = predictor.detect(image, temporal_smooth=True)
        return pred_by_frame

    def _predictions_from_cache(
        self,
        disc_id: str,
        frames: list[int],
    ) -> dict[int, EllipseResult | None] | None:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        import pandas as pd

        df = pd.read_csv(self.cache_path) if self.cache_path.suffix == ".csv" else None
        if df is None:
            try:
                df = pd.read_parquet(self.cache_path)
            except ImportError:
                return None
        disc_df = df[df["disc_id"] == disc_id]
        if disc_df.empty:
            return None
        pred_by_frame: dict[int, EllipseResult | None] = {}
        for frame in frames:
            rows = disc_df[disc_df["frame"] == frame]
            if rows.empty or not bool(rows.iloc[0]["detected"]):
                pred_by_frame[frame] = None
                continue
            row = rows.iloc[0]
            pred_by_frame[frame] = EllipseResult(
                BX=float(row["pred_BX"]),
                BY=float(row["pred_BY"]),
                Major=float(row["pred_Major"]),
                Minor=float(row["pred_Minor"]),
                Angle=float(row["pred_Angle"]),
                method="cached",
            )
        return pred_by_frame

    def _show_current_frame(self) -> None:
        if not self.frames or self.disc_id is None:
            return

        frame = self.frames[self.frame_idx]
        image = load_gray_image(self.data_dir / f"{self.disc_id}_frame{frame:04d}.tif")
        gt = self.gt_by_frame[frame]
        pred = self.pred_by_frame.get(frame)
        hybrid = self.hybrid_by_frame.get(frame)

        self.image_artist.set_data(image)
        update_ellipse_outline_patch(self.gt_line, gt)
        update_ellipse_outline_patch(self.pred_line, pred)
        update_ellipse_outline_patch(self.hybrid_line, hybrid)
        csv_row = self.gt_csv_by_frame.get(frame)
        if csv_row is not None:
            display_bbox = fiji_bbox_patch(csv_row)
            self.bbox_patch.set_bounds(
                display_bbox.get_x(),
                display_bbox.get_y(),
                display_bbox.get_width(),
                display_bbox.get_height(),
            )
            self.bbox_patch.set_visible(True)
        else:
            self.bbox_patch.set_visible(False)

        gt_line = format_params(gt, "Fiji GT")
        pred_line = format_params(pred, "Classical")
        hybrid_line = format_params(hybrid, "Hybrid")
        err_line = ""
        if pred is not None:
            errors = compare_to_ground_truth(pred.as_dict(), gt)
            err_bits = [f"{key}={errors[key]:.1f}" for key in GT_COLUMNS]
            err_line = "Classical |pred−GT|:  " + "  ".join(err_bits)
        hybrid_err_line = ""
        if hybrid is not None:
            herr = compare_to_ground_truth(hybrid.as_dict(), gt)
            herr_bits = [f"{key}={herr[key]:.1f}" for key in GT_COLUMNS]
            hybrid_err_line = "Hybrid |pred−GT|:  " + "  ".join(herr_bits)

        self.title.set_text(
            f"{self.disc_id}   frame {frame}   ({self.frame_idx + 1}/{len(self.frames)})"
            f"   [frames {self.frame_min}–{self.frame_max}]"
        )
        self.info_text.set_text(
            f"{gt_line}\n{pred_line}\n{hybrid_line}\n{err_line}\n{hybrid_err_line}\n"
            "Green = Fiji GT    Red = classical    Cyan = hybrid    Yellow = bbox    ← → step"
        )
        self.fig.canvas.draw_idle()

    def show(self, initial_disc: str | None = None) -> None:
        if initial_disc:
            self.disc_box.set_val(initial_disc)
            self._load_disc(initial_disc)
        plt.show()


def launch_viewer(
    data_dir: Path | str = "Data",
    model_path: Path | str = "splits/detector_model.json",
    hybrid_model_path: Path | str | None = DEFAULT_HYBRID_MODEL,
    cache_path: Path | str | None = None,
    disc: str | None = None,
    frame_min: int = DEFAULT_FRAME_MIN,
    frame_max: int = DEFAULT_FRAME_MAX,
) -> None:
    viewer = DiscViewer(
        data_dir=data_dir,
        model_path=model_path,
        hybrid_model_path=hybrid_model_path,
        cache_path=cache_path,
        frame_min=frame_min,
        frame_max=frame_max,
    )
    viewer.show(initial_disc=disc)
