"""Data loading and train/test splitting for annular ablation experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

FRAME_PATTERN = re.compile(r"^(disc\d+)_frame(\d+)\.tif$")
GT_COLUMNS = ["BX", "BY", "Major", "Minor", "Angle"]
FIJI_BBOX_COLUMNS = ["Width", "Height"]
DEFAULT_FRAME_MIN = 6
DEFAULT_FRAME_MAX = 45


def fiji_major_axis_vector(angle_deg: float) -> tuple[float, float]:
    """
    Unit vector along the major axis in image pixels (+x right, +y down).

    ImageJ ``EllipseFitter`` defines ``theta`` as the major-axis angle measured
    **clockwise** from +x (see ``drawEllipse`` in ``ij.process.EllipseFitter``).
    """
    rad = np.deg2rad(float(angle_deg))
    return float(np.cos(rad)), float(np.sin(rad))


def fiji_angle_from_axis_vector(vx: float, vy: float) -> float:
    """Clockwise major-axis angle in degrees from +x, in [0, 180)."""
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)


def fiji_ellipse_outline(
    cx: float,
    cy: float,
    major: float,
    minor: float,
    angle_deg: float,
    *,
    round_center: bool = True,
) -> np.ndarray:
    """
    Closed (N, 2) polygon tracing the same curve Fiji draws with ``drawEllipse``.

    Port of ``ij.process.EllipseFitter.drawEllipse`` (ImageJ 1.x).
    """
    if major <= 0.0 or minor <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    theta = np.deg2rad(float(angle_deg))
    sint = float(np.sin(theta))
    cost = float(np.cos(theta))
    half_major = float(major) / 2.0
    half_minor = float(minor) / 2.0
    rmajor2 = 1.0 / (half_major * half_major)
    rminor2 = 1.0 / (half_minor * half_minor)
    g11 = rmajor2 * cost * cost + rminor2 * sint * sint
    g12 = (rmajor2 - rminor2) * sint * cost
    g22 = rmajor2 * sint * sint + rminor2 * cost * cost
    k1 = -g12 / g11
    k2 = (g12 * g12 - g11 * g22) / (g11 * g11)
    k3 = 1.0 / g11
    if abs(k2) < 1e-15:
        ymax = max(1, int(round(half_minor)))
    else:
        ymax = int(np.floor(np.sqrt(np.abs(k3 / k2))))
    if ymax < 1:
        ymax = 1

    xc = int(round(cx)) if round_center else int(cx)
    yc = int(round(cy)) if round_center else int(cy)
    ymin = -ymax
    txmin: dict[int, int] = {}
    txmax: dict[int, int] = {}
    for y in range(0, ymax + 1):
        discriminant = k2 * y * y + k3
        if discriminant < 0.0:
            txmin[y] = 0
            txmax[y] = 0
            continue
        j2 = float(np.sqrt(discriminant))
        j1 = k1 * y
        txmin[y] = int(round(j1 - j2))
        txmax[y] = int(round(j1 + j2))

    points: list[tuple[float, float]] = []
    for y in range(ymin, ymax):
        x = txmax[-y] if y < 0 else -txmin[y]
        points.append((xc + x, yc + y))
    for y in range(ymax, ymin, -1):
        x = txmin[-y] if y < 0 else -txmax[y]
        points.append((xc + x, yc + y))
    return np.asarray(points, dtype=np.float64)


def fiji_bbox_size(major: float, minor: float, angle_deg: float) -> tuple[float, float]:
    """Axis-aligned bounding-box size of a rotated ellipse (Fiji Width/Height)."""
    a = float(major) / 2.0
    b = float(minor) / 2.0
    rad = np.deg2rad(float(angle_deg))
    cos_a = abs(np.cos(rad))
    sin_a = abs(np.sin(rad))
    width = 2.0 * np.sqrt((a * cos_a) ** 2 + (b * sin_a) ** 2)
    height = 2.0 * np.sqrt((a * sin_a) ** 2 + (b * cos_a) ** 2)
    return width, height


def canonicalize_fiji_ellipse_params(params: dict[str, float]) -> dict[str, float]:
    """
    Normalize ellipse parameters: Major >= Minor, Angle in [0, 180).

    Do not reparameterize near-circles toward horizontal — that caused large
    frame-to-frame angle jumps when Major ≈ Minor. Temporal EMA unwraps angle.
    """
    major = float(params["Major"])
    minor = float(params["Minor"])
    angle = float(params["Angle"]) % 180.0

    if major < minor:
        major, minor = minor, major
        angle = (angle + 90.0) % 180.0

    return {
        **params,
        "BX": float(params["BX"]),
        "BY": float(params["BY"]),
        "Major": major,
        "Minor": minor,
        "Angle": angle,
    }


def fiji_csv_to_ellipse_params(row: pd.Series | dict[str, float]) -> dict[str, float]:
    """
    Convert Fiji Results columns to ellipse parameters used by training and viz.

    Fiji BX/BY are the upper-left corner of the axis-aligned bounding rectangle.
    Returned BX/BY are the geometric ellipse center in image pixels.

    Major and Minor are full axis lengths. Angle follows ImageJ Fit Ellipse: the
    major-axis orientation measured **clockwise** from +x (0–180°), as in
    ``ij.process.EllipseFitter``.
    """
    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    major = float(values["Major"])
    minor = float(values["Minor"])
    angle = float(values["Angle"])
    if "Width" in values and "Height" in values:
        width = float(values["Width"])
        height = float(values["Height"])
    else:
        width, height = fiji_bbox_size(major, minor, angle)
    return {
        "BX": float(values["BX"]) + width / 2.0,
        "BY": float(values["BY"]) + height / 2.0,
        "Major": major,
        "Minor": minor,
        "Angle": angle,
    }


@dataclass(frozen=True)
class Sample:
    disc_id: str
    frame: int
    image_path: Path
    ground_truth: dict[str, float] | None = None


def load_gray_image(path: Path | str) -> np.ndarray:
    """Load a grayscale image, scaling 16-bit TIFFs to 8-bit."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    return image


def load_ground_truth(csv_path: Path | str) -> pd.DataFrame:
    """Load Fiji ellipse measurements for one disc."""
    df = pd.read_csv(csv_path)
    frame_col = df.columns[0]
    df = df.rename(columns={frame_col: "frame"})
    df["frame"] = df["frame"].astype(int)
    missing = [col for col in GT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return df


def list_disc_ids(data_dir: Path | str) -> list[str]:
    data_dir = Path(data_dir)
    return sorted(path.stem for path in data_dir.glob("disc*.csv"))


def split_discs(
    disc_ids: list[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split whole discs into train and test sets (no frame leakage)."""
    rng = np.random.default_rng(seed)
    ids = list(disc_ids)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    test_ids = sorted(ids[:n_test])
    train_ids = sorted(ids[n_test:])
    return train_ids, test_ids


def save_split(
    train_ids: list[str],
    test_ids: list[str],
    output_path: Path | str,
) -> None:
    payload = {"train": train_ids, "test": test_ids}
    Path(output_path).write_text(json.dumps(payload, indent=2))


def load_split(split_path: Path | str) -> tuple[list[str], list[str]]:
    payload = json.loads(Path(split_path).read_text())
    return payload["train"], payload["test"]


def assert_disjoint_disc_splits(train_ids: list[str], test_ids: list[str]) -> None:
    """Fail if any disc appears in both train and test (prevents frame leakage)."""
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Train/test disc overlap: {sorted(overlap)}")
    if not train_ids:
        raise ValueError("Train disc list is empty.")
    if not test_ids:
        raise ValueError("Test disc list is empty.")


def crop_box(
    center_x: float,
    center_y: float,
    crop_size: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) clamped crop box around a center."""
    height, width = image_shape
    half = crop_size // 2
    x0 = int(round(center_x)) - half
    y0 = int(round(center_y)) - half
    x0 = max(0, min(x0, width - crop_size))
    y0 = max(0, min(y0, height - crop_size))
    return x0, y0, x0 + crop_size, y0 + crop_size


def crop_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


class AblationDataset:
    """Iterate over disc/frame pairs with optional Fiji ground truth."""

    def __init__(
        self,
        data_dir: Path | str,
        disc_ids: list[str] | None = None,
        require_ground_truth: bool = True,
        frame_min: int | None = DEFAULT_FRAME_MIN,
        frame_max: int | None = DEFAULT_FRAME_MAX,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.disc_ids = disc_ids or list_disc_ids(self.data_dir)
        self.require_ground_truth = require_ground_truth
        self.frame_min = frame_min
        self.frame_max = frame_max

    def _frame_in_range(self, frame: int) -> bool:
        if self.frame_min is not None and frame < self.frame_min:
            return False
        if self.frame_max is not None and frame > self.frame_max:
            return False
        return True

    def __len__(self) -> int:
        return sum(len(list(self._frames_for_disc(d))) for d in self.disc_ids)

    def __iter__(self) -> Iterator[Sample]:
        for disc_id in self.disc_ids:
            gt_df = None
            csv_path = self.data_dir / f"{disc_id}.csv"
            if csv_path.exists():
                gt_df = load_ground_truth(csv_path).set_index("frame")
            elif self.require_ground_truth:
                continue

            for frame, image_path in self._frames_for_disc(disc_id):
                if not self._frame_in_range(frame):
                    continue
                gt = None
                if gt_df is not None and frame in gt_df.index:
                    gt = fiji_csv_to_ellipse_params(gt_df.loc[frame])
                elif self.require_ground_truth:
                    continue
                yield Sample(disc_id, frame, image_path, gt)

    def _frames_for_disc(self, disc_id: str) -> Iterator[tuple[int, Path]]:
        for path in sorted(self.data_dir.glob(f"{disc_id}_frame*.tif")):
            match = FRAME_PATTERN.match(path.name)
            if match is None:
                continue
            frame = int(match.group(2))
            if not self._frame_in_range(frame):
                continue
            yield frame, path
