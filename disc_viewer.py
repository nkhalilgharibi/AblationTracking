#!/usr/bin/env python3
"""Interactive viewer: scroll through frames and compare Fiji GT vs detected ellipses."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_edge.data import DEFAULT_FRAME_MAX, DEFAULT_FRAME_MIN
from ablation_edge.viewer import launch_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive ablation ellipse viewer (GT vs prediction per frame)."
    )
    parser.add_argument(
        "disc",
        nargs="?",
        default=None,
        help="Optional disc ID to load on startup (e.g. disc202207050101 or 202207050101)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--model", type=Path, default=Path("splits/detector_model.json"))
    parser.add_argument("--frame-min", type=int, default=DEFAULT_FRAME_MIN)
    parser.add_argument("--frame-max", type=int, default=DEFAULT_FRAME_MAX)
    args = parser.parse_args()

    launch_viewer(
        data_dir=args.data_dir,
        model_path=args.model,
        disc=args.disc,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
    )


if __name__ == "__main__":
    main()
