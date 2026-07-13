#!/usr/bin/env python3
"""Run trained detector on one image or an entire disc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ablation_edge.data import fiji_csv_to_ellipse_params, load_gray_image
from ablation_edge.train import load_trained_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect ablation ellipse on TIFF frames.")
    parser.add_argument("image", type=Path, help="Path to a frame TIFF")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("splits/detector_model.json"),
        help="Trained detector artifact",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        default=None,
        help="Optional Fiji CSV for the same disc",
    )
    args = parser.parse_args()

    detector = load_trained_detector(args.model)
    image = load_gray_image(args.image)
    result = detector.detect(image)
    if result is None:
        print("Detection failed.")
        return

    print(json.dumps(result.as_dict(), indent=2))

    if args.compare_csv is not None:
        disc_id = args.image.name.split("_frame")[0]
        df = pd.read_csv(args.compare_csv)
        frame_col = df.columns[0]
        frame = int(args.image.stem.split("_frame")[-1])
        row = df[df[frame_col] == frame].iloc[0]
        gt = fiji_csv_to_ellipse_params(row)
        print("\nFiji ground truth (ellipse center):")
        for key in ["BX", "BY", "Major", "Minor", "Angle"]:
            print(f"  {key}: gt={gt[key]:.2f}  pred={getattr(result, key):.2f}")


if __name__ == "__main__":
    main()
