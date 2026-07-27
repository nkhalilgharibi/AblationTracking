"""Edge detection and ellipse fitting for annular ablation cuts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .data import GT_COLUMNS, canonicalize_fiji_ellipse_params, fiji_angle_from_axis_vector


@dataclass
class EllipseResult:
    """Ellipse fit; BX/BY are the geometric center in image pixels."""

    BX: float
    BY: float
    Major: float
    Minor: float
    Angle: float
    method: str = "combined"
    confidence: float = 1.0

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _fit_ellipse_from_points(points: np.ndarray) -> dict[str, float] | None:
    """Fit points with OpenCV and return Fiji-format params (major-axis angle from +x)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 5:
        return None
    (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(points)
    major, minor = max(axis1, axis2), min(axis1, axis2)
    rad = np.deg2rad(float(angle))
    if axis1 >= axis2:
        vx, vy = float(np.cos(rad)), float(np.sin(rad))
    else:
        vx, vy = float(-np.sin(rad)), float(np.cos(rad))
    fiji_angle = fiji_angle_from_axis_vector(vx, vy)
    return canonicalize_fiji_ellipse_params(
        {
            "BX": float(cx),
            "BY": float(cy),
            "Major": float(major),
            "Minor": float(minor),
            "Angle": fiji_angle,
        }
    )


def _sector_extremity_points(
    points: np.ndarray,
    radii: np.ndarray,
    thetas: np.ndarray,
    n_sectors: int,
) -> np.ndarray:
    """Outermost detected edge point in each angular sector."""
    if len(points) == 0:
        return points
    selected: list[np.ndarray] = []
    for sector in range(n_sectors):
        t0 = sector * 2.0 * np.pi / n_sectors
        t1 = (sector + 1) * 2.0 * np.pi / n_sectors
        mask = (thetas >= t0) & (thetas < t1)
        if not np.any(mask):
            continue
        idx = int(np.argmax(radii[mask]))
        selected.append(points[mask][idx])
    if not selected:
        return points
    return np.asarray(selected, dtype=np.float32)


def _arc_tip_points(
    points: np.ndarray,
    radii: np.ndarray,
    thetas: np.ndarray,
) -> np.ndarray:
    """Tips of the visible ablation arc (largest angular gap between detections)."""
    if len(points) < 12:
        return np.empty((0, 2), dtype=np.float32)
    order = np.argsort(thetas)
    sorted_t = thetas[order]
    gaps = np.diff(np.concatenate([sorted_t, sorted_t[:1] + 2.0 * np.pi]))
    tip_a = order[int(np.argmax(gaps))]
    tip_b = order[(int(np.argmax(gaps)) + 1) % len(thetas)]
    return np.asarray([points[tip_a], points[tip_b]], dtype=np.float32)


def _visible_arc_fraction(thetas: np.ndarray) -> float:
    """Fraction of 360° covered by detected ablation edge rays."""
    if len(thetas) < 10:
        return 1.0
    sorted_t = np.sort(thetas)
    gaps = np.diff(np.concatenate([sorted_t, sorted_t[:1] + 2.0 * np.pi]))
    visible = 2.0 * np.pi - float(np.max(gaps))
    return float(np.clip(visible / (2.0 * np.pi), 0.0, 1.0))


def _partial_arc_visible(
    thetas: np.ndarray,
    radii: np.ndarray,
    confident: np.ndarray,
    arc_pct: float,
) -> float:
    if confident.size and int(confident.sum()) >= 10:
        arc_t = thetas[confident]
        arc_r = radii[confident]
    else:
        arc_t, arc_r = thetas, radii
    return _outer_envelope_arc_fraction(arc_t, arc_r, arc_pct)


def _outer_envelope_points(
    points: np.ndarray,
    radii: np.ndarray,
    percentile: float,
) -> np.ndarray:
    if len(points) == 0:
        return points
    thr = float(np.percentile(radii, percentile))
    return np.asarray(points[radii >= thr], dtype=np.float32)


def _outer_envelope_arc_fraction(
    thetas: np.ndarray,
    radii: np.ndarray,
    percentile: float = 80.0,
    min_rays: int = 10,
) -> float:
    """
    Visible arc fraction using only outer-envelope rays.

    Partial Fiji cuts leave angular gaps where rays hit shorter-radius tissue;
    counting all rays overestimates coverage (~98%). Outer-percentile rays
    better match the annotated cut span.
    """
    if len(thetas) < min_rays:
        return 1.0
    thr = float(np.percentile(radii, percentile))
    mask = radii >= thr
    if int(mask.sum()) < min_rays:
        return _visible_arc_fraction(thetas)
    return _visible_arc_fraction(thetas[mask])


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _signed_angle_delta(angle: float, reference: float) -> float:
    """Signed difference angle − reference in (−90°, 90°] (180° period)."""
    return float((float(angle) - float(reference) + 90.0) % 180.0 - 90.0)


def _unwrap_angle(angle: float, reference: float) -> float:
    """Map angle to the [0, 180) representative closest to reference."""
    return (float(reference) + _signed_angle_delta(angle, reference)) % 180.0


def _ema_ellipse(
    current: dict[str, float],
    previous: dict[str, float] | None,
    alpha: float,
) -> dict[str, float]:
    """Exponentially smooth ellipse params; keep angle continuous."""
    if previous is None or alpha >= 1.0:
        return canonicalize_fiji_ellipse_params(current)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    delta = _signed_angle_delta(current["Angle"], previous["Angle"])
    # Near-circular fits have noisy orientation; hold previous angle.
    circular = current["Minor"] / max(current["Major"], 1.0) > 0.92
    if circular:
        delta = 0.0
        alpha_angle = 0.0
    else:
        alpha_angle = alpha
    blended = {
        "BX": alpha * current["BX"] + (1.0 - alpha) * previous["BX"],
        "BY": alpha * current["BY"] + (1.0 - alpha) * previous["BY"],
        "Major": alpha * current["Major"] + (1.0 - alpha) * previous["Major"],
        "Minor": alpha * current["Minor"] + (1.0 - alpha) * previous["Minor"],
        "Angle": (previous["Angle"] + alpha_angle * delta) % 180.0,
    }
    return canonicalize_fiji_ellipse_params(blended)


class AblationEdgeDetector:
    """
    Fit an ellipse to the *outer* boundary of the dark annular ablation ring.

    Along each ray from the disc center the intensity profile is:
        bright (inner tissue) → dark (ablation ring) → bright (outer tissue).

    Fiji ellipses annotate that outer dark→bright transition. Detection:
    1. Estimate disc center from the bright central tissue blob
    2. Sample the outer ablation edge on rays around 360°
    3. Fit an ellipse to those points (two-pass center refine)

    Training tunes classical hyperparameters via sklearn search (see ``ablation_edge.model``).
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        # Blob fallback — matches test.ipynb OpenCV knobs.
        "blur": 11,
        "threshold": 20,
        "morph_kernel": 2,
        "morph_iterations": 4,
        "min_area": 5000,
        # Ring-fit path.
        "edge_blur": 11,
        "dark_threshold": 22,
        "bright_threshold": 22,
        # Fixed dark cut is more stable than annulus-percentile adaptive on dark discs
        # (annulus median can sit ~12 and falsely clip to a low floor).
        "use_adaptive_threshold": False,
        "dark_percentile": 65.0,
        "gradient_percentile": 55,
        "r_min": 70,
        "r_max": 200,
        "n_angles": 180,
        "min_dark_run": 4,
        # Fiji annotators emphasize cut extremities; use mild point biasing, not global
        # ellipse inflation (enclosing_percentile=0).
        "dark_run_mode": "first",
        "extremity_sectors": 36,
        "use_arc_tips": True,
        "extremity_arc_full": 0.86,  # plain fit when outer arc >= 86%
        "extremity_arc_severe": 0.65,  # sector+outer when outer arc < 65%
        "extremity_arc_max": 0.80,  # legacy: mild extremity weighting below this
        "extremity_arc_percentile": 80.0,
        "enclosing_percentile": 0.0,
        "radius_filter_mode": "extremity",  # extremity | symmetric | none
        "radius_outlier_sigma": 2.5,
        "center_max_area": 40000,
        "major_min": 150.0,
        "major_max": 450.0,
        "temporal_alpha": 0.45,
    }

    def __init__(self, refine_edges: bool = False, **params: Any) -> None:
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.refine_edges = refine_edges
        self._prev_ellipse: dict[str, float] | None = None
        self._prev_dark_threshold: float | None = None
        self._smooth_thresholds: bool = False

    def reset_temporal_state(self) -> None:
        """Clear EMA state before processing a new disc sequence."""
        self._prev_ellipse = None
        self._prev_dark_threshold = None

    def detect(self, image: np.ndarray, *, temporal_smooth: bool = False) -> EllipseResult | None:
        self._smooth_thresholds = temporal_smooth
        fitted = self._detect_ablation_ring(image)
        confidence = 1.0
        if fitted is None:
            fitted = self._detect_from_blob(image)
            confidence = 0.5
        if fitted is None:
            return None

        if temporal_smooth:
            alpha = float(self.params.get("temporal_alpha", 0.35))
            fitted = _ema_ellipse(fitted, self._prev_ellipse, alpha)
            self._prev_ellipse = {key: fitted[key] for key in GT_COLUMNS}

        method = "ring"
        if temporal_smooth and float(self.params.get("temporal_alpha", 0.35)) < 1.0:
            method = f"{method}_stable"
        return EllipseResult(**fitted, method=method, confidence=float(confidence))

    def detect_sequence(self, images: list[np.ndarray]) -> list[EllipseResult | None]:
        """Detect a time series with temporal EMA smoothing (stable within a disc)."""
        self.reset_temporal_state()
        return [self.detect(image, temporal_smooth=True) for image in images]

    def _best_raw_detection(self, image: np.ndarray) -> dict[str, float] | None:
        return self._detect_ablation_ring(image) or self._detect_from_blob(image)

    def _estimate_thresholds(self, blurred: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
        """
        Derive dark/bright cuts from intensities in the ablation annulus.

        Fixed cv2.threshold / dark_threshold values are brittle across discs.
        Global Otsu (~39 on these images) sits above the ring and is too high.
        Percentile in the radial band matches the dark ring better.
        """
        if not bool(self.params.get("use_adaptive_threshold", True)):
            return float(self.params["dark_threshold"]), float(self.params["bright_threshold"])

        height, width = blurred.shape
        r_min = int(self.params["r_min"])
        r_max = int(self.params["r_max"])
        yy, xx = np.ogrid[:height, :width]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        band = blurred[(rr >= r_min) & (rr <= r_max)]
        if band.size < 500:
            return float(self.params["dark_threshold"]), float(self.params["bright_threshold"])

        dark_pct = float(self.params.get("dark_percentile", 65.0))
        dark = float(np.percentile(band, dark_pct))
        # Keep dark cut near the notebook / tuned fixed range (~20–25).
        dark = float(np.clip(dark, 18.0, 28.0))
        if self._smooth_thresholds and self._prev_dark_threshold is not None:
            dark = 0.3 * dark + 0.7 * self._prev_dark_threshold
        if self._smooth_thresholds:
            self._prev_dark_threshold = dark
        bright = float(max(dark + 2.0, self.params["bright_threshold"]))
        return dark, bright

    def _estimate_disc_center(self, blurred: np.ndarray) -> tuple[float, float]:
        height, width = blurred.shape
        cx0, cy0 = width * 0.44, height * 0.58
        bright_threshold = float(self.params["bright_threshold"])
        bright = (blurred >= bright_threshold).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
        best_idx = None
        best_score = -1.0
        # Central tissue blob is typically ~8k–25k px; huge components are merged
        # background bright regions that pull the center far off (bad discs).
        max_area = int(self.params.get("center_max_area", 40000))
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 5000 or area > max_area:
                continue
            bx, by = float(centroids[idx][0]), float(centroids[idx][1])
            # Strong prior toward the usual disc location.
            dist = np.hypot(bx - cx0, by - cy0)
            score = area / (1.0 + (dist / 25.0) ** 2)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            # Relax area cap once if nothing matched.
            for idx in range(1, count):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < 5000 or area > 120000:
                    continue
                bx, by = float(centroids[idx][0]), float(centroids[idx][1])
                dist = np.hypot(bx - cx0, by - cy0)
                score = area / (1.0 + (dist / 25.0) ** 2)
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_idx is None:
            return cx0, cy0
        return float(centroids[best_idx][0]), float(centroids[best_idx][1])

    def _outer_ablation_points(
        self,
        blurred: np.ndarray,
        magnitude: np.ndarray,
        cx: float,
        cy: float,
        dark_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Collect outer-ablation-ring edge points around a center."""
        if dark_threshold is None:
            dark_threshold, _ = self._estimate_thresholds(blurred, cx, cy)
        dark_threshold = float(dark_threshold)
        r_min = int(self.params["r_min"])
        r_max = int(self.params["r_max"])
        n_angles = int(self.params["n_angles"])
        min_run = int(self.params["min_dark_run"])
        run_mode = str(self.params.get("dark_run_mode", "longest"))
        outlier_sigma = float(self.params.get("radius_outlier_sigma", 2.5))
        height, width = blurred.shape
        points: list[list[float]] = []
        edge_radii: list[float] = []
        confident: list[bool] = []

        for theta in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            radii = list(range(r_min, r_max))
            dark_flags: list[bool] = []
            intensities: list[float] = []
            grads: list[float] = []
            for radius in radii:
                x = int(round(cx + radius * cos_t))
                y = int(round(cy + radius * sin_t))
                if 0 <= x < width and 0 <= y < height:
                    value = float(blurred[y, x])
                    dark_flags.append(value < dark_threshold)
                    intensities.append(value)
                    grads.append(float(magnitude[y, x]))
                else:
                    dark_flags.append(False)
                    intensities.append(0.0)
                    grads.append(0.0)

            runs: list[tuple[int, int]] = []
            idx = 0
            while idx < len(dark_flags):
                if not dark_flags[idx]:
                    idx += 1
                    continue
                start = idx
                while idx < len(dark_flags) and dark_flags[idx]:
                    idx += 1
                end = idx - 1
                if end - start + 1 >= min_run:
                    runs.append((start, end))

            edge_radius: int | None = None
            from_dark_run = False
            if runs:
                if run_mode == "last":
                    start, end = runs[-1]
                elif run_mode == "first":
                    start, end = runs[0]
                elif run_mode == "transition":
                    # Outer ablation boundary: just outside the dark ring.
                    start, end = runs[0]
                    edge_radius = radii[min(end + 1, len(radii) - 1)]
                else:  # longest
                    start, end = max(runs, key=lambda t: t[1] - t[0])
                if edge_radius is None:
                    edge_radius = radii[end]
                from_dark_run = True
            else:
                # Fallback: strongest rising intensity within a darkish neighborhood.
                best_radius = None
                best_grad = -1.0
                for j in range(1, len(intensities) - 1):
                    if intensities[j] >= dark_threshold + 8.0:
                        continue
                    rise = intensities[j + 1] - intensities[j - 1]
                    if rise > best_grad:
                        best_grad = rise
                        best_radius = radii[j]
                edge_radius = best_radius

            if edge_radius is not None:
                points.append([cx + edge_radius * cos_t, cy + edge_radius * sin_t])
                edge_radii.append(float(edge_radius))
                confident.append(from_dark_run)

        if len(points) < 25:
            return None

        pts = np.asarray(points, dtype=np.float32)
        radii_arr = np.asarray(edge_radii, dtype=np.float64)
        conf_arr = np.asarray(confident, dtype=bool)
        thetas_arr = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)[: len(radii_arr)]

        filter_mode = str(self.params.get("radius_filter_mode", "extremity"))
        if filter_mode != "none":
            med = float(np.median(radii_arr))
            mad = float(np.median(np.abs(radii_arr - med))) + 1e-6
            if filter_mode == "extremity":
                # Drop inner speckle hits; keep outer points (Fiji extremity semantics).
                keep = radii_arr >= med - outlier_sigma * 1.4826 * mad
            else:
                keep = np.abs(radii_arr - med) < outlier_sigma * 1.4826 * mad
            if int(keep.sum()) >= 25:
                pts = pts[keep]
                radii_arr = radii_arr[keep]
                thetas_arr = thetas_arr[keep]
                conf_arr = conf_arr[keep]

        return pts, radii_arr, thetas_arr, conf_arr

    def _fiji_fit_points(
        self,
        points: np.ndarray,
        radii: np.ndarray,
        thetas: np.ndarray,
        confident: np.ndarray,
    ) -> np.ndarray:
        """Build point set with mild extremity emphasis on partial arcs."""
        arc_max = float(self.params.get("extremity_arc_max", 0.80))
        arc_pct = float(self.params.get("extremity_arc_percentile", 80.0))
        visible = _partial_arc_visible(thetas, radii, confident, arc_pct)
        if visible >= arc_max:
            return np.asarray(points, dtype=np.float32)

        n_sectors = int(self.params.get("extremity_sectors", 36))
        sector_dup = 2 if visible < arc_max else 1
        parts: list[np.ndarray] = [points]
        if n_sectors > 0:
            sector_pts = _sector_extremity_points(points, radii, thetas, n_sectors)
            if len(sector_pts):
                for _ in range(sector_dup):
                    parts.append(sector_pts)
        if bool(self.params.get("use_arc_tips", True)):
            tip_pts = points[confident] if confident.any() else points
            tip_radii = radii[confident] if confident.any() else radii
            tip_thetas = thetas[confident] if confident.any() else thetas
            tips = _arc_tip_points(tip_pts, tip_radii, tip_thetas)
            if len(tips):
                parts.extend([tips, tips])
        return np.asarray(np.vstack(parts), dtype=np.float32)

    def _fit_from_rays(
        self,
        points: np.ndarray,
        radii: np.ndarray,
        thetas: np.ndarray,
        confident: np.ndarray,
    ) -> dict[str, float] | None:
        """
        Adaptive Fiji-style fit: plain / weighted / outer / sector+outer by arc span.
        """
        arc_pct = float(self.params.get("extremity_arc_percentile", 80.0))
        arc_full = float(self.params.get("extremity_arc_full", 0.86))
        arc_severe = float(
            self.params.get(
                "extremity_arc_severe",
                self.params.get("extremity_arc_max", 0.65),
            )
        )
        n_sectors = int(self.params.get("extremity_sectors", 36))

        visible = _partial_arc_visible(thetas, radii, confident, arc_pct)
        plain_pts = self._fiji_fit_points(points, radii, thetas, confident)
        plain = _fit_ellipse_from_points(plain_pts)
        if plain is None:
            plain = _fit_ellipse_from_points(points)
        if plain is None:
            return None

        if visible >= arc_full:
            return plain

        outer_pts = _outer_envelope_points(points, radii, arc_pct)
        sector = _sector_extremity_points(points, radii, thetas, n_sectors)

        if visible < arc_severe:
            if len(sector) >= 5 and len(outer_pts) >= 5:
                severe = _fit_ellipse_from_points(np.vstack([sector, outer_pts]))
                if severe is not None:
                    return severe
        return plain

    def _detect_ablation_ring(self, image: np.ndarray) -> dict[str, float] | None:
        blur = int(self.params["edge_blur"])
        if blur % 2 == 0:
            blur += 1
        major_min = float(self.params["major_min"])
        major_max = float(self.params["major_max"])

        blurred = cv2.GaussianBlur(image.astype(np.float32), (blur, blur), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        cx, cy = self._estimate_disc_center(blurred)
        dark_threshold, _bright = self._estimate_thresholds(blurred, cx, cy)
        collected = self._outer_ablation_points(blurred, magnitude, cx, cy, dark_threshold)
        if collected is None:
            return None
        points, radii, thetas, confident = collected
        fitted = self._fit_from_rays(points, radii, thetas, confident)
        if fitted is None:
            return None

        # Second pass from the fitted center (improves axes when center was off).
        collected2 = self._outer_ablation_points(
            blurred, magnitude, fitted["BX"], fitted["BY"], dark_threshold
        )
        if collected2 is not None:
            points2, radii2, thetas2, confident2 = collected2
            fitted2 = self._fit_from_rays(points2, radii2, thetas2, confident2)
            if fitted2 is not None and major_min <= fitted2["Major"] <= major_max:
                maj_delta = abs(fitted2["Major"] - fitted["Major"]) / max(
                    fitted["Major"], 1.0
                )
                center_delta = float(
                    np.hypot(
                        fitted2["BX"] - fitted["BX"],
                        fitted2["BY"] - fitted["BY"],
                    )
                )
                angle_delta = _angle_diff(fitted2["Angle"], fitted["Angle"])
                # Partial-arc fits can blow up on re-centering; keep stable pass-1.
                if (
                    maj_delta < 0.08
                    and center_delta < 15.0
                    and angle_delta < 20.0
                ):
                    fitted = fitted2

        if not (major_min <= fitted["Major"] <= major_max):
            return None
        return fitted

    def _detect_from_blob(self, image: np.ndarray) -> dict[str, float] | None:
        """Fallback: notebook-style dark-blob contour ellipse (test.ipynb knobs)."""
        blur = int(self.params.get("blur", 11))
        if blur % 2 == 0:
            blur += 1
        threshold = int(self.params.get("threshold", 20))
        min_area = int(self.params.get("min_area", 5000))
        morph_k = int(self.params.get("morph_kernel", 2))
        morph_iter = int(self.params.get("morph_iterations", 4))
        major_min = float(self.params["major_min"])
        major_max = float(self.params["major_max"])

        blurred = cv2.GaussianBlur(image, (blur, blur), 0)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        height, width = image.shape
        best_score = -1.0
        best: dict[str, float] | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or len(contour) < 5:
                continue
            fitted = _fit_ellipse_from_points(contour)
            if fitted is None:
                continue
            if not (major_min <= fitted["Major"] <= major_max):
                continue
            # Skip flooded whole-image contours (common on dark discs at thr~20).
            if area > 0.55 * height * width:
                continue
            cx, cy = fitted["BX"], fitted["BY"]
            if not (width * 0.25 < cx < width * 0.7 and height * 0.35 < cy < height * 0.8):
                continue
            score = area / ((1.0 + cx / 100.0) * (1.0 + cy / 100.0))
            if score > best_score:
                best_score = score
                best = fitted
        return best

def compare_to_ground_truth(
    prediction: dict[str, float],
    ground_truth: dict[str, float],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for key in GT_COLUMNS:
        diff = abs(prediction[key] - ground_truth[key])
        if key == "Angle":
            diff = _angle_diff(prediction[key], ground_truth[key])
        errors[key] = diff
    return errors
