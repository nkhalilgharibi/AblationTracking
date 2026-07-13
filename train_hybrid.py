#!/usr/bin/env python3
"""Train hybrid refiner: classical ring fit + CNN residual vs Fiji GT."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_edge.data import DEFAULT_FRAME_MAX, DEFAULT_FRAME_MIN
from ablation_edge.hybrid_train import train_hybrid_refiner


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train CNN residual on top of classical AblationEdgeDetector. "
            "Train/test discs are disjoint (no frame leakage)."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--split", type=Path, default=Path("splits/train_test_split.json"))
    parser.add_argument("--detector-model", type=Path, default=Path("splits/detector_model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--frame-min", type=int, default=DEFAULT_FRAME_MIN)
    parser.add_argument("--frame-max", type=int, default=DEFAULT_FRAME_MAX)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: fewer samples and epochs",
    )
    args = parser.parse_args()

    train_hybrid_refiner(
        data_dir=args.data_dir,
        split_path=args.split,
        detector_model_path=args.detector_model,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        batch_size=8 if args.quick else args.batch_size,
        epochs=5 if args.quick else args.epochs,
        learning_rate=args.lr,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
        max_train_samples=400 if args.quick else None,
        max_test_samples=200 if args.quick else None,
        device=args.device,
    )


if __name__ == "__main__":
    main()
