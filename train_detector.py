#!/usr/bin/env python3
"""Train the annular ablation edge detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_edge.data import DEFAULT_FRAME_MAX, DEFAULT_FRAME_MIN
from ablation_edge.train import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ablation edge detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--split-path", type=Path, default=Path("splits/train_test_split.json"))
    parser.add_argument("--tune-samples", type=int, default=400)
    parser.add_argument("--frame-min", type=int, default=DEFAULT_FRAME_MIN)
    parser.add_argument("--frame-max", type=int, default=DEFAULT_FRAME_MAX)
    args = parser.parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        split_path=args.split_path,
        tune_sample_limit=args.tune_samples,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
    )


if __name__ == "__main__":
    main()
