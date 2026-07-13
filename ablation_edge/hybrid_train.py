"""Train CNN residual refiner on top of classical AblationEdgeDetector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    apply_residual,
    build_hybrid_refiner,
    encode_raw_features,
    hybrid_residual_loss,
    residual_targets,
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
    """Run classical detector once; store (crop, raw features, residual targets).

    Uses ``AblationDataset``, which pairs Fiji CSV row *k* with TIFF frame
    ``k + 5`` (CSV 1 → image 6 through CSV 40 → image 45 by default).
    """
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
                "target": residual_targets(raw, sample.ground_truth, crop_size),
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


class _HybridPairDataset(TorchDataset):
    def __init__(self, pairs: list[dict[str, Any]], torch_mod) -> None:
        self.pairs = pairs
        self._torch = torch_mod

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = self._torch
        item = self.pairs[index]
        return {
            "crop": torch.from_numpy(item["crop"]).unsqueeze(0),
            "raw_feat": torch.from_numpy(item["raw_feat"]),
            "target": torch.from_numpy(item["target"]),
        }


def train_hybrid_refiner(
    data_dir: Path | str = "Data",
    split_path: Path | str = "splits/train_test_split.json",
    detector_model_path: Path | str = "splits/detector_model.json",
    output_dir: Path | str = "splits",
    *,
    crop_size: int = 256,
    batch_size: int = 16,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    frame_min: int = DEFAULT_FRAME_MIN,
    frame_max: int = DEFAULT_FRAME_MAX,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    device: str | None = None,
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    torch = _require_torch()
    from torch.utils.data import DataLoader, random_split

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids, test_ids = load_split(split_path)
    assert_disjoint_disc_splits(train_ids, test_ids)
    print(f"Train discs: {len(train_ids)} | Test discs: {len(test_ids)} (no leakage)")

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

    full_ds = _HybridPairDataset(train_pairs, torch)
    n_val = max(1, int(round(len(full_ds) * val_fraction)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        _HybridPairDataset(test_pairs, torch), batch_size=batch_size, shuffle=False
    )

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device_t = torch.device(device)
    print(f"Device: {device_t}")

    model = build_hybrid_refiner(crop_size=crop_size).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val = float("inf")
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "hybrid_model.pt"
    config_path = output_dir / "hybrid_model.json"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            crop = batch["crop"].to(device_t)
            raw_feat = batch["raw_feat"].to(device_t)
            target = batch["target"].to(device_t)
            optimizer.zero_grad(set_to_none=True)
            pred = model(crop, raw_feat)
            loss, _ = hybrid_residual_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = _loader_loss(model, val_loader, device_t)
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
                    "frame_min": frame_min,
                    "frame_max": frame_max,
                    "train_discs": train_ids,
                    "test_discs": test_ids,
                    "best_val_loss": best_val,
                    "detector_model": str(detector_model_path),
                },
                checkpoint_path,
            )

    config = {
        "model_type": "hybrid_classical_cnn_residual",
        "crop_size": crop_size,
        "frame_min": frame_min,
        "frame_max": frame_max,
        "train_discs": train_ids,
        "test_discs": test_ids,
        "checkpoint": str(checkpoint_path),
        "best_val_loss": best_val,
        "history": history,
        "split_policy": "disc_level_no_frame_leakage",
        "strategy": "classical_ring_fit + CNN residual (Fiji - raw)",
    }
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved {checkpoint_path}")

    # Held-out test ellipse metrics: classical vs hybrid
    metrics = evaluate_hybrid_on_pairs(test_pairs, checkpoint_path, crop_size, device_t)
    config["test_metrics"] = metrics
    config_path.write_text(json.dumps(config, indent=2))
    return config


def _loader_loss(model, loader, device_t) -> float:
    torch = _require_torch()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["crop"].to(device_t), batch["raw_feat"].to(device_t))
            loss, _ = hybrid_residual_loss(pred, batch["target"].to(device_t))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def evaluate_hybrid_on_pairs(
    pairs: list[dict[str, Any]],
    checkpoint_path: Path | str,
    crop_size: int,
    device_t,
) -> dict[str, Any]:
    torch = _require_torch()
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = build_hybrid_refiner(crop_size=crop_size)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device_t)
    model.eval()

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
    print_metrics(raw_df, title="Classical only (held-out)")
    print_metrics(hyb_df, title="Hybrid classical+CNN (held-out)")

    def _mae(df):
        return {
            col: float(df.loc[df["detected"], f"err_{col}"].mean())
            for col in GT_COLUMNS
            if df["detected"].any()
        }

    return {"classical": _mae(raw_df), "hybrid": _mae(hyb_df)}
