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
    parser.add_argument("--tune-samples", type=int, default=None)
    parser.add_argument("--frame-min", type=int, default=DEFAULT_FRAME_MIN)
    parser.add_argument("--frame-max", type=int, default=DEFAULT_FRAME_MAX)
    parser.add_argument(
        "--tune-frame-stride",
        type=int,
        default=5,
        help="Use every N-th frame for hyperparameter search within "
        "--frame-min/--frame-max (default: 5 → frames 6,11,...,41).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        dest="cv_folds",
        help="GroupKFold folds over discs (default: 3).",
    )
    parser.add_argument(
        "--search",
        choices=("grid", "random"),
        default="grid",
        help="Hyperparameter search strategy (default: grid).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=24,
        help="RandomizedSearchCV iterations when --search random.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--verbose",
        type=int,
        default=3,
        help="sklearn search verbosity (0–3; default: 3).",
    )
    args = parser.parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        split_path=args.split_path,
        tune_sample_limit=args.tune_samples,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
        tune_frame_stride=args.tune_frame_stride,
        cv_folds=args.cv_folds,
        search=args.search,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
