"""Classical ring fit + CNN residual refinement (hybrid AI).

Ports only the CNN refiner from the withNN branch. Classical detection uses
this branch's ``AblationEdgeDetector`` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import GT_COLUMNS, canonicalize_fiji_ellipse_params, crop_box, crop_image
from .detector import AblationEdgeDetector, EllipseResult, _angle_residual

DEFAULT_ENCODER_CHANNELS: tuple[int, ...] = (32, 64, 128)
DEFAULT_MLP_HIDDEN: tuple[int, ...] = (128, 64)
DEFAULT_DROPOUT = 0.2


def build_hybrid_refiner(
    *,
    crop_size: int = 256,
    encoder_channels: Sequence[int] = DEFAULT_ENCODER_CHANNELS,
    mlp_hidden: Sequence[int] = DEFAULT_MLP_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
) -> Any:
    """CNN that maps (crop, raw ellipse features) → residual corrections."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch required: pip install torch") from exc

    channels = tuple(int(c) for c in encoder_channels)
    if len(channels) < 1:
        raise ValueError("encoder_channels must contain at least one channel width.")
    hidden = tuple(int(h) for h in mlp_hidden)
    feature_dim = channels[-1]

    class HybridResidualNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            enc_layers: list[nn.Module] = []
            in_ch = 1
            for i, out_ch in enumerate(channels):
                enc_layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
                enc_layers.append(nn.ReLU(inplace=True))
                if i < len(channels) - 1:
                    enc_layers.append(nn.MaxPool2d(2))
                else:
                    enc_layers.append(nn.AdaptiveAvgPool2d(1))
                in_ch = out_ch
            self.encoder = nn.Sequential(*enc_layers)

            head_layers: list[nn.Module] = []
            in_dim = feature_dim + 6  # raw: BX,BY,Major,Minor,sin2A,cos2A
            if hidden:
                head_layers.append(nn.Linear(in_dim, hidden[0]))
                head_layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    head_layers.append(nn.Dropout(dropout))
                for h_in, h_out in zip(hidden, hidden[1:]):
                    head_layers.append(nn.Linear(h_in, h_out))
                    head_layers.append(nn.ReLU(inplace=True))
                head_layers.append(nn.Linear(hidden[-1], 5))
            else:
                head_layers.append(nn.Linear(in_dim, 5))
            self.head = nn.Sequential(*head_layers)

            self.crop_size = crop_size
            self.encoder_channels = channels
            self.mlp_hidden = hidden
            self.dropout = float(dropout)

        def forward(self, crop: Any, raw_feat: Any) -> Any:
            feats = self.encoder(crop).flatten(1)
            return self.head(torch.cat([feats, raw_feat], dim=1))

    return HybridResidualNet()


def encode_raw_features(
    raw: dict[str, float],
    image_shape: tuple[int, int],
    crop_size: int,
) -> np.ndarray:
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


def residual_targets(
    raw: dict[str, float],
    gt: dict[str, float],
    crop_size: int,
) -> np.ndarray:
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


def ellipse_params_vector(params: dict[str, float]) -> np.ndarray:
    """Pack BX, BY, Major, Minor, Angle as float32."""
    return np.asarray(
        [float(params[c]) for c in GT_COLUMNS],
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


def hybrid_quadratic_sincos_loss(
    delta: Any,
    raw_params: Any,
    gt_params: Any,
    crop_size: int,
) -> tuple[Any, dict[str, float]]:
    """
    Equal-weight quadratic loss on refined ellipse vs GT.

    Geometry terms are normalized by ``crop_size``; angle uses the continuous
    ``sin(2θ), cos(2θ)`` embedding (same idea as classical sincos cost).
    All six terms have weight 1.0.
    """
    import torch

    s = float(crop_size)
    bx = raw_params[..., 0] + delta[..., 0] * s
    by = raw_params[..., 1] + delta[..., 1] * s
    major = torch.clamp(raw_params[..., 2] + delta[..., 2] * s, min=1.0)
    minor = torch.clamp(raw_params[..., 3] + delta[..., 3] * s, min=1.0)
    angle = raw_params[..., 4] + delta[..., 4] * 90.0

    gt_bx = gt_params[..., 0]
    gt_by = gt_params[..., 1]
    gt_major = gt_params[..., 2]
    gt_minor = gt_params[..., 3]
    gt_angle = gt_params[..., 4]

    ang_rad = torch.deg2rad(angle)
    gt_ang_rad = torch.deg2rad(gt_angle)

    terms = torch.stack(
        [
            ((bx - gt_bx) / s) ** 2,
            ((by - gt_by) / s) ** 2,
            ((major - gt_major) / s) ** 2,
            ((minor - gt_minor) / s) ** 2,
            (torch.sin(2.0 * ang_rad) - torch.sin(2.0 * gt_ang_rad)) ** 2,
            (torch.cos(2.0 * ang_rad) - torch.cos(2.0 * gt_ang_rad)) ** 2,
        ],
        dim=-1,
    )
    per_sample = terms.mean(dim=-1)
    loss = per_sample.mean()
    return loss, {"mse": float(loss.detach().cpu())}


def arch_from_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture kwargs stored with a hybrid checkpoint."""
    return {
        "crop_size": int(payload.get("crop_size", 256)),
        "encoder_channels": tuple(
            payload.get("encoder_channels", DEFAULT_ENCODER_CHANNELS)
        ),
        "mlp_hidden": tuple(payload.get("mlp_hidden", DEFAULT_MLP_HIDDEN)),
        "dropout": float(payload.get("dropout", DEFAULT_DROPOUT)),
    }


class HybridEllipsePredictor:
    """
    Classical radial / blob ellipse fit, then CNN residual refinement.

    Keeps this branch's ``AblationEdgeDetector`` unchanged; AI only predicts
    Fiji − raw corrections from an image crop around the classical fit.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        detector: AblationEdgeDetector | None = None,
        detector_model_path: Path | str = "splits/detector_model_sincos.json",
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
        arch = arch_from_checkpoint(payload)
        self.crop_size = arch["crop_size"]
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

        self.model = build_hybrid_refiner(**arch)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        det_path = payload.get("detector_model", detector_model_path)
        self.detector = detector or load_trained_detector(det_path)

    def reset_temporal_state(self) -> None:
        self.detector.reset_temporal_state()

    def refine(self, image: np.ndarray, raw: dict[str, float]) -> dict[str, float]:
        torch = self._torch
        box = crop_box(raw["BX"], raw["BY"], self.crop_size, image.shape)
        patch = crop_image(image, box).astype(np.float32) / 255.0
        crop_t = torch.from_numpy(patch)[None, None, ...].to(self.device)
        raw_t = torch.from_numpy(
            encode_raw_features(raw, image.shape, self.crop_size)
        )[None, :].to(self.device)
        with torch.no_grad():
            delta = self.model(crop_t, raw_t).cpu().numpy()[0]
        return apply_residual(raw, delta, self.crop_size)

    def detect(self, image: np.ndarray, *, temporal_smooth: bool = False) -> EllipseResult | None:
        raw_result = self.detector.detect(image, temporal_smooth=temporal_smooth)
        if raw_result is None:
            return None
        raw = {key: float(raw_result.as_dict()[key]) for key in GT_COLUMNS}
        refined = self.refine(image, raw)
        return EllipseResult(
            **refined,
            method="hybrid_cnn",
            confidence=float(raw_result.confidence),
        )

    def detect_sequence(self, images: list[np.ndarray]) -> list[EllipseResult | None]:
        self.reset_temporal_state()
        return [self.detect(image, temporal_smooth=True) for image in images]
