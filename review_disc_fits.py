#!/usr/bin/env python3
"""
Review ellipse fits per disc: BX/BY stability and GT vs prediction plots.

Examples:
  # Ground-truth center stability for all discs (fast, no detection)
  python review_disc_fits.py --stability-only

  # Full review for one disc (plots + overlay images)
  python review_disc_fits.py --disc disc202207050101

  # All discs, save to reviews/
  python review_disc_fits.py --all --output-dir reviews

  # Reuse cached predictions
  python review_disc_fits.py --all --cache reviews/predictions.parquet --no-detect
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ablation_edge.data import AblationDataset, list_disc_ids, load_gray_image
from ablation_edge.evaluate import evaluate_predictions
from ablation_edge.stability import (
    flag_unstable_centers,
    pivot_stability_wide,
    print_stability_report,
    summarize_all_discs,
)
from ablation_edge.train import load_trained_detector
from ablation_edge.viz import plot_center_drift, plot_disc_parameters, save_overlay_frames


def run_detection(
    data_dir: Path,
    disc_ids: list[str],
    model_path: Path,
) -> pd.DataFrame:
    detector = load_trained_detector(model_path)
    rows = []
    for disc_id in tqdm(disc_ids, desc="Detecting"):
        dataset = AblationDataset(data_dir, [disc_id])
        samples = sorted(list(dataset), key=lambda s: s.frame)
        detector.reset_temporal_state()
        for sample in samples:
            image = load_gray_image(sample.image_path)
            prediction = detector.detect(image, temporal_smooth=True)
            rows.append(
                {
                    "disc_id": sample.disc_id,
                    "frame": sample.frame,
                    "ground_truth": sample.ground_truth,
                    "prediction": prediction,
                }
            )
    return evaluate_predictions(rows)


def load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.suffix == ".parquet":
        return pd.read_parquet(cache_path)
    return pd.read_csv(cache_path)


def save_cache(eval_df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix == ".parquet":
        try:
            eval_df.to_parquet(cache_path, index=False)
            return
        except ImportError:
            cache_path = cache_path.with_suffix(".csv")
    eval_df.to_csv(cache_path, index=False)


def review_disc(
    eval_df: pd.DataFrame,
    disc_id: str,
    data_dir: Path,
    output_dir: Path,
    overlay_frames: int,
) -> None:
    disc_out = output_dir / disc_id
    disc_out.mkdir(parents=True, exist_ok=True)
    plot_disc_parameters(eval_df, disc_id, disc_out / "parameters.png")
    plot_center_drift(eval_df, disc_id, disc_out / "center_drift.png")
    save_overlay_frames(
        eval_df,
        data_dir,
        disc_id,
        disc_out,
        max_frames=overlay_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Review per-disc ellipse fits and center stability.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--model", type=Path, default=Path("splits/detector_model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reviews"))
    parser.add_argument("--disc", action="append", default=[], help="Disc ID to review (repeatable)")
    parser.add_argument("--all", action="store_true", help="Review every disc")
    parser.add_argument(
        "--stability-only",
        action="store_true",
        help="Only print BX/BY stability table (GT; add --no-detect with cache for predictions)",
    )
    parser.add_argument("--max-step", type=float, default=5.0, help="Flag threshold for |ΔBX|/|ΔBY| (px)")
    parser.add_argument("--max-range", type=float, default=10.0, help="Flag threshold for total BX/BY range (px)")
    parser.add_argument("--overlay-frames", type=int, default=6, help="Number of overlay images per disc")
    parser.add_argument("--cache", type=Path, default=None, help="Path to save/load prediction cache (.csv or .parquet)")
    parser.add_argument("--no-detect", action="store_true", help="Skip detection; requires --cache")
    args = parser.parse_args()

    all_discs = list_disc_ids(args.data_dir)
    if args.all:
        disc_ids = all_discs
    elif args.disc:
        disc_ids = args.disc
    else:
        disc_ids = all_discs

    eval_df: pd.DataFrame | None = None
    if args.cache and args.no_detect:
        if not args.cache.exists():
            raise FileNotFoundError(f"Cache not found: {args.cache}")
        eval_df = load_cache(args.cache)
        if not args.all and args.disc:
            eval_df = eval_df[eval_df["disc_id"].isin(args.disc)]
    elif not args.stability_only or args.cache is not None:
        if args.no_detect:
            raise ValueError("--no-detect requires an existing --cache file")
        eval_df = run_detection(args.data_dir, disc_ids, args.model)
        if args.cache:
            save_cache(eval_df, args.cache)

    stability_df = summarize_all_discs(args.data_dir, eval_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stability_wide = pivot_stability_wide(stability_df)
    stability_wide.to_csv(args.output_dir / "center_stability.csv", index=False)
    flagged = flag_unstable_centers(stability_df, args.max_step, args.max_range)
    flagged.to_csv(args.output_dir / "center_stability_flagged.csv", index=False)
    print_stability_report(stability_df, args.max_step, args.max_range)
    print(f"\nSaved stability tables to {args.output_dir}")

    if args.stability_only:
        return

    if eval_df is None:
        raise RuntimeError("Internal error: eval_df missing for review plots")

    for disc_id in tqdm(disc_ids, desc="Generating reviews"):
        review_disc(eval_df, disc_id, args.data_dir, args.output_dir, args.overlay_frames)

    print(f"\nSaved per-disc plots and overlays to {args.output_dir}/<disc_id>/")


if __name__ == "__main__":
    main()
