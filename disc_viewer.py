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
    parser.add_argument(
        "--hybrid-model",
        type=Path,
        default=Path("splits/hybrid_model.pt"),
        help="Classical+CNN hybrid checkpoint (cyan overlay).",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable cyan hybrid overlay",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional prediction cache CSV (default: live detection)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use reviews/predictions.csv if present",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Run live detection (ignore --cache and --use-cache)",
    )
    parser.add_argument("--frame-min", type=int, default=DEFAULT_FRAME_MIN)
    parser.add_argument("--frame-max", type=int, default=DEFAULT_FRAME_MAX)
    args = parser.parse_args()

    if args.no_cache:
        cache = None
    else:
        cache = args.cache
        if cache is None and args.use_cache:
            cache = Path("reviews/predictions.csv")
        if cache is not None and not cache.exists():
            cache = None

    launch_viewer(
        data_dir=args.data_dir,
        model_path=args.model,
        hybrid_model_path=None if args.no_hybrid else args.hybrid_model,
        cache_path=cache,
        disc=args.disc,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
    )


if __name__ == "__main__":
    main()
