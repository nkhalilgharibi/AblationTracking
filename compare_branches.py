#!/usr/bin/env python3
"""Compare newEdit vs withNN branch metrics on the shared disc split."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
SPLIT_PATH = ROOT / "splits" / "train_test_split.json"
OUT_PATH = ROOT / "COMPARISON.md"

NEWEDIT_ROOT = ROOT  # current checkout is feature/ablation-edge-detector-newEdit
WITHNN_ROOT = Path("/Users/nargess/AblationTracking-withNN")

GT_COLUMNS = ["BX", "BY", "Major", "Minor", "Angle"]


def _load_module_from(root: Path, package: str = "ablation_edge"):
    """Import ablation_edge from a specific worktree root."""
    root_s = str(root)
    # Drop any previously loaded ablation_edge modules.
    to_drop = [k for k in sys.modules if k == package or k.startswith(package + ".")]
    for k in to_drop:
        del sys.modules[k]
    sys.path.insert(0, root_s)
    try:
        mod = importlib.import_module(package)
        return mod
    finally:
        if sys.path and sys.path[0] == root_s:
            sys.path.pop(0)


def _signed_angle_delta(angle: float, reference: float) -> float:
    return float((float(angle) - float(reference) + 90.0) % 180.0 - 90.0)


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """MAE/RMSE matching newEdit evaluate.py (circular Angle residual for RMSE)."""
    abs_errs: dict[str, list[float]] = {c: [] for c in GT_COLUMNS}
    signed: dict[str, list[float]] = {c: [] for c in GT_COLUMNS}
    n_total = len(rows)
    n_detected = 0
    for row in rows:
        gt = row["ground_truth"]
        pred = row["prediction"]
        if pred is None:
            continue
        if hasattr(pred, "as_dict"):
            pred_d = pred.as_dict()
        else:
            pred_d = pred
        if pred_d.get("BX") is None:
            continue
        n_detected += 1
        for col in GT_COLUMNS:
            p = float(pred_d[col])
            g = float(gt[col])
            if col == "Angle":
                abs_errs[col].append(min(abs(p - g) % 180.0, 180.0 - (abs(p - g) % 180.0)))
                signed[col].append(_signed_angle_delta(p, g))
            else:
                abs_errs[col].append(abs(p - g))
                signed[col].append(p - g)

    out: dict[str, float] = {
        "frames_total": float(n_total),
        "frames_detected": float(n_detected),
        "detection_rate": n_detected / max(n_total, 1),
    }
    for col in GT_COLUMNS:
        ae = np.asarray(abs_errs[col], dtype=np.float64)
        se = np.asarray(signed[col], dtype=np.float64)
        if len(ae) == 0:
            out[f"mae_{col}"] = float("nan")
            out[f"rmse_{col}"] = float("nan")
        else:
            out[f"mae_{col}"] = float(np.mean(ae))
            out[f"rmse_{col}"] = float(np.sqrt(np.mean(se**2)))
    return out


def _collect_samples(root: Path, disc_ids: list[str]):
    sys.path.insert(0, str(root))
    try:
        from ablation_edge.data import AblationDataset, load_gray_image

        dataset = AblationDataset(DATA_DIR, disc_ids, frame_min=6, frame_max=45)
        samples = list(dataset)
        images = [load_gray_image(s.image_path) for s in samples]
        return samples, images
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)


def _eval_predictor(
    samples,
    images,
    predict_fn: Callable[[Any], Any],
) -> dict[str, float]:
    rows = []
    for sample, image in zip(samples, images):
        rows.append(
            {
                "disc_id": sample.disc_id,
                "frame": sample.frame,
                "ground_truth": sample.ground_truth,
                "prediction": predict_fn(image),
            }
        )
    return _metrics_from_rows(rows)


def evaluate_newedit(split: dict[str, list[str]]) -> dict[str, Any]:
    print("\n=== Evaluating feature/ablation-edge-detector-newEdit ===", flush=True)
    sys.path.insert(0, str(NEWEDIT_ROOT))
    try:
        # Clear modules
        for k in [k for k in sys.modules if k == "ablation_edge" or k.startswith("ablation_edge.")]:
            del sys.modules[k]
        from ablation_edge.data import AblationDataset, load_gray_image
        from ablation_edge.train import load_trained_detector

        detector = load_trained_detector(NEWEDIT_ROOT / "splits" / "detector_model.json")

        def run_split(disc_ids: list[str], label: str) -> tuple[int, dict[str, float]]:
            samples = list(AblationDataset(DATA_DIR, disc_ids, frame_min=6, frame_max=45))
            images = [load_gray_image(s.image_path) for s in samples]
            print(f"  {label}: {len(samples)} frames...", flush=True)
            metrics = _eval_predictor(samples, images, lambda img: detector.detect(img))
            return len(samples), metrics

        n_train, train_m = run_split(split["train"], "train")
        n_test, test_m = run_split(split["test"], "test")
        return {
            "name": "feature/ablation-edge-detector-newEdit",
            "n_train": n_train,
            "n_test": n_test,
            "train": train_m,
            "test": test_m,
        }
    finally:
        if sys.path and sys.path[0] == str(NEWEDIT_ROOT):
            sys.path.pop(0)


def evaluate_withnn(split: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n=== Evaluating feature/ablation-edge-detector-withNN ===", flush=True)
    root = WITHNN_ROOT
    sys.path.insert(0, str(root))
    try:
        for k in [k for k in sys.modules if k == "ablation_edge" or k.startswith("ablation_edge.")]:
            del sys.modules[k]
        from ablation_edge.data import AblationDataset, load_gray_image
        from ablation_edge.hybrid_refine import HybridEllipsePredictor
        from ablation_edge.train import load_trained_detector

        classical = load_trained_detector(root / "splits" / "detector_model.json")
        hybrid = HybridEllipsePredictor(
            root / "splits" / "hybrid_model.pt",
            detector=classical,
            detector_model_path=root / "splits" / "detector_model.json",
        )

        def run_split(disc_ids: list[str], label: str):
            samples = list(AblationDataset(DATA_DIR, disc_ids, frame_min=6, frame_max=45))
            images = [load_gray_image(s.image_path) for s in samples]
            print(f"  {label}: {len(samples)} frames (classical)...", flush=True)
            classical_m = _eval_predictor(samples, images, lambda img: classical.detect(img))
            print(f"  {label}: {len(samples)} frames (hybrid)...", flush=True)
            hybrid.reset_temporal_state()
            hybrid_m = _eval_predictor(
                samples, images, lambda img: hybrid.detect(img, temporal_smooth=False)
            )
            return len(samples), classical_m, hybrid_m

        n_train, tr_c, tr_h = run_split(split["train"], "train")
        n_test, te_c, te_h = run_split(split["test"], "test")
        hybrid_result = {
            "name": "feature/ablation-edge-detector-withNN (hybrid)",
            "n_train": n_train,
            "n_test": n_test,
            "train": tr_h,
            "test": te_h,
        }
        classical_result = {
            "name": "feature/ablation-edge-detector-withNN (classical only)",
            "n_train": n_train,
            "n_test": n_test,
            "train": tr_c,
            "test": te_c,
        }
        return hybrid_result, classical_result
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)


def _fmt_block(title: str, n: int, metrics: dict[str, float]) -> str:
    lines = [
        f"## {title}",
        f"Samples: {n}",
        f"Detected: {int(metrics['frames_detected'])} / {int(metrics['frames_total'])} "
        f"({metrics['detection_rate']:.1%})",
        "",
        "| Param | MAE | RMSE |",
        "|-------|-----|------|",
    ]
    for col in GT_COLUMNS:
        lines.append(
            f"| {col} | {metrics[f'mae_{col}']:.2f} | {metrics[f'rmse_{col}']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section(result: dict[str, Any]) -> str:
    return (
        f"# {result['name']}\n\n"
        + _fmt_block("Train", result["n_train"], result["train"])
        + _fmt_block("Test", result["n_test"], result["test"])
        + "\n"
    )


def _summary_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Compact test summary",
        "",
        "| Branch | Test n | Det% | MAE Major | RMSE Major | MAE Minor | RMSE Minor | MAE Angle | RMSE Angle |",
        "|--------|--------|------|-----------|------------|-----------|------------|-----------|------------|",
    ]
    for r in results:
        t = r["test"]
        lines.append(
            f"| {r['name']} | {r['n_test']} | {t['detection_rate']*100:.1f}% | "
            f"{t['mae_Major']:.2f} | {t['rmse_Major']:.2f} | "
            f"{t['mae_Minor']:.2f} | {t['rmse_Minor']:.2f} | "
            f"{t['mae_Angle']:.2f} | {t['rmse_Angle']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    print(
        f"Split: {len(split['train'])} train discs, {len(split['test'])} test discs",
        flush=True,
    )
    print(f"Data: {DATA_DIR}", flush=True)

    newedit = evaluate_newedit(split)
    hybrid, classical = evaluate_withnn(split)
    results = [newedit, hybrid, classical]

    body = (
        "# Branch metrics comparison\n\n"
        "Evaluated on the shared disc-level split (78 train / 20 test), "
        "frames 6–45, using each branch’s checked-in artifacts (no retrain).\n\n"
        "- **newEdit**: classical OpenCV detector (`detector_model.json`, no Ridge).\n"
        "- **withNN (hybrid)**: classical + CNN residual (`hybrid_model.pt`).\n"
        "- **withNN (classical only)**: same branch classical detector without CNN.\n\n"
        "Metrics: MAE = mean absolute error; RMSE from signed residuals "
        "(Angle uses shortest circular residual on a 180° period).\n\n"
        + _summary_table(results)
        + "\n---\n\n"
        + "".join(_section(r) for r in results)
    )
    OUT_PATH.write_text(body)
    print("\n" + body)
    print(f"\nWrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
