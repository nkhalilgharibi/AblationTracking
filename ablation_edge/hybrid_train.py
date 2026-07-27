"""Train CNN residual refiner on top of this branch's AblationEdgeDetector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from .data import (
    DEFAULT_FRAME_MAX,
    DEFAULT_FRAME_MIN,
    GT_COLUMNS,
    AblationDataset,
    assert_disjoint_disc_splits,
    crop_box,
    crop_image,
    load_gray_image,
    load_split,
)
from .evaluate import evaluate_predictions, print_metrics
from .hybrid_refine import (
    DEFAULT_DROPOUT,
    DEFAULT_ENCODER_CHANNELS,
    DEFAULT_MLP_HIDDEN,
    apply_residual,
    arch_from_checkpoint,
    build_hybrid_refiner,
    encode_raw_features,
    ellipse_params_vector,
    hybrid_quadratic_sincos_loss,
)
from .train import load_trained_detector


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch required: pip install torch") from exc
    return torch


def _collect_pairs(
    data_dir: Path,
    disc_ids: list[str],
    detector,
    *,
    frame_min: int,
    frame_max: int,
    crop_size: int,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Run classical detector once; store (crop, raw features, raw/gt params)."""
    dataset = AblationDataset(data_dir, disc_ids, frame_min=frame_min, frame_max=frame_max)
    pairs: list[dict[str, Any]] = []
    for sample in tqdm(list(dataset), desc="Collecting classical detections"):
        if sample.ground_truth is None:
            continue
        image = load_gray_image(sample.image_path)
        raw_result = detector.detect(image, temporal_smooth=False)
        if raw_result is None:
            continue
        raw = {key: float(raw_result.as_dict()[key]) for key in GT_COLUMNS}
        box = crop_box(raw["BX"], raw["BY"], crop_size, image.shape)
        patch = crop_image(image, box).astype(np.float32) / 255.0
        pairs.append(
            {
                "crop": patch,
                "raw_feat": encode_raw_features(raw, image.shape, crop_size),
                "raw_params": ellipse_params_vector(raw),
                "gt_params": ellipse_params_vector(sample.ground_truth),
                "raw": raw,
                "gt": sample.ground_truth,
                "disc_id": sample.disc_id,
                "frame": sample.frame,
            }
        )
        if max_samples is not None and len(pairs) >= max_samples:
            break
    return pairs


try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    TorchDataset = object  # type: ignore[misc, assignment]


def _augment_crop(
    crop: np.ndarray,
    *,
    max_shift: int = 8,
    brightness: float = 0.15,
    contrast: float = 0.15,
    noise_std: float = 0.02,
    rng: np.random.Generator,
) -> np.ndarray:
    """Photometric + small translation jitter on a float crop in [0, 1]."""
    out = crop.astype(np.float32, copy=True)
    h, w = out.shape

    if max_shift > 0:
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        if dy != 0 or dx != 0:
            padded = np.pad(out, max_shift, mode="edge")
            y0 = max_shift + dy
            x0 = max_shift + dx
            out = padded[y0 : y0 + h, x0 : x0 + w]

    if brightness > 0:
        scale = float(rng.uniform(1.0 - brightness, 1.0 + brightness))
        out = out * scale

    if contrast > 0:
        factor = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
        mean = float(out.mean())
        out = (out - mean) * factor + mean

    if noise_std > 0:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


class _HybridPairDataset(TorchDataset):
    def __init__(
        self,
        pairs: list[dict[str, Any]],
        torch_mod,
        *,
        augment: bool = False,
        aug_max_shift: int = 8,
        aug_brightness: float = 0.15,
        aug_contrast: float = 0.15,
        aug_noise_std: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.pairs = pairs
        self._torch = torch_mod
        self.augment = augment
        self.aug_max_shift = aug_max_shift
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.aug_noise_std = aug_noise_std
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = self._torch
        item = self.pairs[index]
        crop = item["crop"]
        if self.augment:
            crop = _augment_crop(
                crop,
                max_shift=self.aug_max_shift,
                brightness=self.aug_brightness,
                contrast=self.aug_contrast,
                noise_std=self.aug_noise_std,
                rng=self._rng,
            )
        return {
            "crop": torch.from_numpy(crop).unsqueeze(0),
            "raw_feat": torch.from_numpy(item["raw_feat"]),
            "raw_params": torch.from_numpy(item["raw_params"]),
            "gt_params": torch.from_numpy(item["gt_params"]),
        }


def train_hybrid_refiner(
    data_dir: Path | str = "Data",
    split_path: Path | str = "splits/train_test_split.json",
    detector_model_path: Path | str = "splits/detector_model_sincos.json",
    output_dir: Path | str = "splits",
    *,
    crop_size: int = 256,
    encoder_channels: Sequence[int] = DEFAULT_ENCODER_CHANNELS,
    mlp_hidden: Sequence[int] = DEFAULT_MLP_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    batch_size: int = 16,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    frame_min: int = DEFAULT_FRAME_MIN,
    frame_max: int = DEFAULT_FRAME_MAX,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    device: str | None = None,
    val_fraction: float = 0.15,
    model_basename: str = "hybrid_model",
    augment: bool = False,
    aug_max_shift: int = 8,
    aug_brightness: float = 0.15,
    aug_contrast: float = 0.15,
    aug_noise_std: float = 0.02,
) -> dict[str, Any]:
    torch = _require_torch()
    from torch.utils.data import DataLoader, Subset

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_channels = tuple(int(c) for c in encoder_channels)
    mlp_hidden = tuple(int(h) for h in mlp_hidden)

    train_ids, test_ids = load_split(split_path)
    assert_disjoint_disc_splits(train_ids, test_ids)
    print(f"Train discs: {len(train_ids)} | Test discs: {len(test_ids)} (no leakage)")
    print(f"Classical detector: {detector_model_path}")
    print(
        f"Arch: encoder_channels={encoder_channels} "
        f"mlp_hidden={mlp_hidden} dropout={dropout} augment={augment}"
    )

    detector = load_trained_detector(detector_model_path)
    train_pairs = _collect_pairs(
        data_dir,
        train_ids,
        detector,
        frame_min=frame_min,
        frame_max=frame_max,
        crop_size=crop_size,
        max_samples=max_train_samples,
    )
    test_pairs = _collect_pairs(
        data_dir,
        test_ids,
        detector,
        frame_min=frame_min,
        frame_max=frame_max,
        crop_size=crop_size,
        max_samples=max_test_samples,
    )
    print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")
    if len(train_pairs) < 20:
        raise ValueError("Too few classical detections to train hybrid refiner.")

    # Train-only aug: split indices first, then wrap train/val with separate datasets
    # so val stays unaugmented.
    n_total = len(train_pairs)
    n_val = max(1, int(round(n_total * val_fraction)))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    aug_kwargs = dict(
        aug_max_shift=aug_max_shift,
        aug_brightness=aug_brightness,
        aug_contrast=aug_contrast,
        aug_noise_std=aug_noise_std,
    )
    train_ds = Subset(
        _HybridPairDataset(
            train_pairs, torch, augment=augment, seed=0, **aug_kwargs
        ),
        train_idx,
    )
    val_ds = Subset(
        _HybridPairDataset(
            train_pairs, torch, augment=False, seed=1, **aug_kwargs
        ),
        val_idx,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device_t = torch.device(device)
    print(f"Device: {device_t}")

    model = build_hybrid_refiner(
        crop_size=crop_size,
        encoder_channels=encoder_channels,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    ).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val = float("inf")
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / f"{model_basename}.pt"
    config_path = output_dir / f"{model_basename}.json"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            crop = batch["crop"].to(device_t)
            raw_feat = batch["raw_feat"].to(device_t)
            raw_params = batch["raw_params"].to(device_t)
            gt_params = batch["gt_params"].to(device_t)
            optimizer.zero_grad(set_to_none=True)
            pred = model(crop, raw_feat)
            loss, _ = hybrid_quadratic_sincos_loss(
                pred, raw_params, gt_params, crop_size
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = _loader_loss(model, val_loader, device_t, crop_size)
        scheduler.step()
        mean_train = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": mean_train, "val_loss": val_loss})
        print(f"Epoch {epoch:03d}  train={mean_train:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "crop_size": crop_size,
                    "encoder_channels": list(encoder_channels),
                    "mlp_hidden": list(mlp_hidden),
                    "dropout": float(dropout),
                    "frame_min": frame_min,
                    "frame_max": frame_max,
                    "train_discs": train_ids,
                    "test_discs": test_ids,
                    "best_val_loss": best_val,
                    "detector_model": str(detector_model_path),
                    "tuning_loss": "mean_quadratic_ellipse_sincos_angle_equal_weight",
                },
                checkpoint_path,
            )

    config = {
        "model_type": "hybrid_classical_cnn_residual",
        "crop_size": crop_size,
        "encoder_channels": list(encoder_channels),
        "mlp_hidden": list(mlp_hidden),
        "dropout": float(dropout),
        "frame_min": frame_min,
        "frame_max": frame_max,
        "train_discs": train_ids,
        "test_discs": test_ids,
        "checkpoint": str(checkpoint_path),
        "best_val_loss": best_val,
        "history": history,
        "detector_model": str(detector_model_path),
        "tuning_loss": "mean_quadratic_ellipse_sincos_angle_equal_weight",
        "augment": {
            "enabled": bool(augment),
            "max_shift": int(aug_max_shift),
            "brightness": float(aug_brightness),
            "contrast": float(aug_contrast),
            "noise_std": float(aug_noise_std),
        },
        "split_policy": "disc_level_no_frame_leakage",
        "strategy": "this-branch classical ring fit + modular CNN residual",
    }
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved {checkpoint_path}")

    metrics = evaluate_hybrid_on_pairs(
        test_pairs,
        checkpoint_path,
        device_t,
        train_pairs=train_pairs,
    )
    config["test_metrics"] = metrics.get("test")
    config["train_metrics"] = metrics.get("train")
    config_path.write_text(json.dumps(config, indent=2))
    return config


def _loader_loss(model, loader, device_t, crop_size: int) -> float:
    torch = _require_torch()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["crop"].to(device_t), batch["raw_feat"].to(device_t))
            loss, _ = hybrid_quadratic_sincos_loss(
                pred,
                batch["raw_params"].to(device_t),
                batch["gt_params"].to(device_t),
                crop_size,
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def evaluate_hybrid_on_pairs(
    test_pairs: list[dict[str, Any]],
    checkpoint_path: Path | str,
    device_t,
    *,
    train_pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    arch = arch_from_checkpoint(payload)
    crop_size = arch["crop_size"]
    model = build_hybrid_refiner(**arch)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device_t)
    model.eval()

    def _eval_split(pairs: list[dict[str, Any]], label: str) -> dict[str, Any]:
        rows_raw = []
        rows_hyb = []
        with torch.no_grad():
            for item in pairs:
                crop = torch.from_numpy(item["crop"])[None, None, ...].to(device_t)
                raw_feat = torch.from_numpy(item["raw_feat"])[None, :].to(device_t)
                delta = model(crop, raw_feat).cpu().numpy()[0]
                hybrid = apply_residual(item["raw"], delta, crop_size)
                rows_raw.append(
                    {
                        "disc_id": item["disc_id"],
                        "frame": item["frame"],
                        "ground_truth": item["gt"],
                        "prediction": item["raw"],
                    }
                )
                rows_hyb.append(
                    {
                        "disc_id": item["disc_id"],
                        "frame": item["frame"],
                        "ground_truth": item["gt"],
                        "prediction": hybrid,
                    }
                )

        raw_df = evaluate_predictions(rows_raw)
        hyb_df = evaluate_predictions(rows_hyb)
        print_metrics(raw_df, title=f"Classical only ({label})")
        print_metrics(hyb_df, title=f"Hybrid classical+CNN ({label})")

        from .evaluate import summarize_metrics

        def _summary(df):
            sm = summarize_metrics(df).iloc[0]
            out = {
                "frames_total": int(sm["frames_total"]),
                "frames_detected": int(sm["frames_detected"]),
                "detection_rate": float(sm["detection_rate"]),
            }
            for col in GT_COLUMNS:
                out[f"mae_{col}"] = float(sm[f"mae_{col}"])
                out[f"rmse_{col}"] = float(sm[f"rmse_{col}"])
            return out

        return {"classical": _summary(raw_df), "hybrid": _summary(hyb_df)}

    result: dict[str, Any] = {"test": _eval_split(test_pairs, "held-out test")}
    if train_pairs is not None:
        result["train"] = _eval_split(train_pairs, "train")
    return result
