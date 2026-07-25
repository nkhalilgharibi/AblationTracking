#!/usr/bin/env python3
"""Train hybrid refiner: this-branch classical ring fit + modular CNN residual."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_edge.data import DEFAULT_FRAME_MAX, DEFAULT_FRAME_MIN
from ablation_edge.hybrid_refine import (
    DEFAULT_DROPOUT,
    DEFAULT_ENCODER_CHANNELS,
    DEFAULT_MLP_HIDDEN,
)
from ablation_edge.hybrid_train import train_hybrid_refiner


def _parse_int_list(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated list of ints")
    return tuple(int(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train CNN residual on top of this branch's AblationEdgeDetector. "
            "Train/test discs are disjoint (no frame leakage)."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--split", type=Path, default=Path("splits/train_test_split.json"))
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=Path("splits/detector_model_sincos.json"),
        help="Classical detector artifact (default: sincos-tuned model).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--model-basename",
        type=str,
        default="hybrid_model",
        help="Checkpoint/config stem under --output-dir "
        "(default: hybrid_model → hybrid_model.pt / .json).",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--encoder-channels",
        type=_parse_int_list,
        default=DEFAULT_ENCODER_CHANNELS,
        help="Comma-separated CNN encoder widths (default: 32,64,128).",
    )
    parser.add_argument(
        "--mlp-hidden",
        type=_parse_int_list,
        default=DEFAULT_MLP_HIDDEN,
        help="Comma-separated MLP hidden widths (default: 128,64).",
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
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
        encoder_channels=args.encoder_channels,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        batch_size=8 if args.quick else args.batch_size,
        epochs=5 if args.quick else args.epochs,
        learning_rate=args.lr,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
        max_train_samples=400 if args.quick else None,
        max_test_samples=200 if args.quick else None,
        device=args.device,
        model_basename=args.model_basename,
    )


if __name__ == "__main__":
    main()
