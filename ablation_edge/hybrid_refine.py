"""Classical ring fit + CNN residual refinement (hybrid AI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import GT_COLUMNS, canonicalize_fiji_ellipse_params, crop_box, crop_image
from .detector import AblationEdgeDetector, EllipseResult, _angle_residual


def build_hybrid_refiner(*, crop_size: int = 256, feature_dim: int = 128) -> Any:
    """CNN that maps (crop, raw ellipse features) → residual corrections."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch required: pip install torch") from exc

    class HybridResidualNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, feature_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            # raw: BX,BY,Major,Minor,sin2A,cos2A (normalized)
            self.head = nn.Sequential(
                nn.Linear(feature_dim + 6, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 5),  # dBX, dBY, dMajor, dMinor, dAngle (degrees, signed)
            )
            self.crop_size = crop_size

        def forward(self, crop: Any, raw_feat: Any) -> Any:
            feats = self.encoder(crop).flatten(1)
            return self.head(torch.cat([feats, raw_feat], dim=1))

    return HybridResidualNet()


def encode_raw_features(raw: dict[str, float], image_shape: tuple[int, int], crop_size: int) -> np.ndarray:
    """Normalize raw classical ellipse for the residual head."""
    height, width = image_shape
    angle_rad = np.deg2rad(float(raw["Angle"]))
    return np.asarray(
        [
            float(raw["BX"]) / width,
            float(raw["BY"]) / height,
            float(raw["Major"]) / crop_size,
            float(raw["Minor"]) / crop_size,
            float(np.sin(2.0 * angle_rad)),
            float(np.cos(2.0 * angle_rad)),
        ],
        dtype=np.float32,
    )


def residual_targets(raw: dict[str, float], gt: dict[str, float], crop_size: int) -> np.ndarray:
    """Fiji − classical residuals (angles as signed degrees in (−90, 90])."""
    return np.asarray(
        [
            (gt["BX"] - raw["BX"]) / crop_size,
            (gt["BY"] - raw["BY"]) / crop_size,
            (gt["Major"] - raw["Major"]) / crop_size,
            (gt["Minor"] - raw["Minor"]) / crop_size,
            _angle_residual(gt["Angle"], raw["Angle"]) / 90.0,
        ],
        dtype=np.float32,
    )


def apply_residual(
    raw: dict[str, float],
    delta: np.ndarray,
    crop_size: int,
) -> dict[str, float]:
    """Apply normalized residual to classical ellipse → Fiji params."""
    return canonicalize_fiji_ellipse_params(
        {
            "BX": float(raw["BX"] + float(delta[0]) * crop_size),
            "BY": float(raw["BY"] + float(delta[1]) * crop_size),
            "Major": float(max(raw["Major"] + float(delta[2]) * crop_size, 1.0)),
            "Minor": float(max(raw["Minor"] + float(delta[3]) * crop_size, 1.0)),
            "Angle": float((raw["Angle"] + float(delta[4]) * 90.0) % 180.0),
        }
    )


def hybrid_residual_loss(pred: Any, target: Any) -> Any:
    import torch.nn.functional as F

    # Weight center more — that was the pure-NN failure mode; classical is already close.
    weights = pred.new_tensor([2.0, 2.0, 1.0, 1.0, 1.5])
    err = (pred - target).abs() * weights
    loss = err.mean()
    return loss, {"mae": float(err.detach().mean().cpu())}


class HybridEllipsePredictor:
    """
    Classical radial / blob ellipse fit, then CNN residual refinement.

    Keeps the same geometric fitting as ``AblationEdgeDetector``; AI only
    predicts Fiji − raw corrections from an image crop around the classical fit.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        detector: AblationEdgeDetector | None = None,
        detector_model_path: Path | str = "splits/detector_model.json",
        *,
        device: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch required: pip install torch") from exc

        from .train import load_trained_detector

        self._torch = torch
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        self.crop_size = int(payload.get("crop_size", 256))
        self.frame_min = int(payload.get("frame_min", 6))
        self.frame_max = int(payload.get("frame_max", 45))

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = build_hybrid_refiner(crop_size=self.crop_size)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.detector = detector or load_trained_detector(detector_model_path)

    def reset_temporal_state(self) -> None:
        self.detector.reset_temporal_state()

    def refine(self, image: np.ndarray, raw: dict[str, float]) -> dict[str, float]:
        torch = self._torch
        box = crop_box(raw["BX"], raw["BY"], self.crop_size, image.shape)
        patch = crop_image(image, box).astype(np.float32) / 255.0
        crop_t = torch.from_numpy(patch)[None, None, ...].to(self.device)
        raw_t = torch.from_numpy(encode_raw_features(raw, image.shape, self.crop_size))[None, :].to(
            self.device
        )
        with torch.no_grad():
            delta = self.model(crop_t, raw_t).cpu().numpy()[0]
        return apply_residual(raw, delta, self.crop_size)

    def detect(self, image: np.ndarray, *, temporal_smooth: bool = False) -> EllipseResult | None:
        raw_result = self.detector.detect(image, temporal_smooth=temporal_smooth)
        if raw_result is None:
            return None
        raw = {key: float(raw_result.as_dict()[key]) for key in GT_COLUMNS}
        refined = self.refine(image, raw)
        return EllipseResult(**refined, method="hybrid_cnn", confidence=float(raw_result.confidence))

    def detect_sequence(self, images: list[np.ndarray]) -> list[EllipseResult | None]:
        self.reset_temporal_state()
        return [self.detect(image, temporal_smooth=True) for image in images]
