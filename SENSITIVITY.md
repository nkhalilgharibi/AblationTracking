# Hyperparameter sensitivity search

Summary of one-at-a-time (OAT) sensitivity studies on the OpenCV ablation-ring detector.
All runs used the same fixed eval set: **400 test frames** (seed=1), frames 6–45, images **512×512**.

**Baseline** (from `splits/detector_model.json` at time of study):

| Parameter | Value |
|-----------|-------|
| `dark_threshold` | 18 |
| `bright_threshold` | 22 |
| `r_min` / `r_max` | 70 / 200 |
| `min_dark_run` | 5 |
| `edge_blur` | 9 |
| `radius_outlier_sigma` | 2.7 |

**Primary metric:** full-ellipse quadratic loss  
`mean(ΔBX² + ΔBY² + ΔMajor² + ΔMinor² + ΔAngle²)` with circular `ΔAngle` on [0, 180).  
Also reported: detection rate, MAE, RMSE per ellipse parameter.

---

## 1. Coarse magnitude sweep (0.1× / 1× / 10× / 100×)

True multiplicative scales of baseline. Clipping only where physically required:

- `r_min` / `r_max` → `[0, 512]` (image size)
- `dark_threshold` / `bright_threshold` → `[0, 255]`
- `edge_blur` → odd positive integer (OpenCV Gaussian kernel)

### Ranking by impact (worst-case Δquad vs 1×)

| Rank | Parameter | Effect |
|------|-----------|--------|
| 1 | `r_min` / `r_max` | Dominates; wrong band → ~60% det, quad ~10⁵ |
| 2 | `edge_blur` | Large degradation at 10×–100× |
| 3 | `dark_threshold` | Large degradation at high cuts |
| 4 | `min_dark_run` | Moderate at 10×+ |
| 5 | `radius_outlier_sigma` | Weak; 10×≈100× (filter effectively off) |
| 6 | `bright_threshold` | Almost flat on this set |

### Compact results (quad / det%)

| Param | 0.1× | 1× | 10× | 100× |
|-------|------|----|-----|------|
| `r_min`/`r_max` | 123785 / 59.5% | 2425 / 100% | 123466 / 59.5%† | 123466 / 59.5%† |
| `edge_blur` | 2694 / 100% | 2425 / 100% | 5204 / 100% | 32076 / 95.2% |
| `dark_threshold` | 6250 / 100% | 2425 / 100% | 28576 / 100%‡ | 28576 / 100%‡ |
| `min_dark_run` | 2586 / 100% | 2425 / 100% | 12365 / 100% | 7256 / 100% |
| `radius_outlier_sigma` | 3048 / 100% | 2425 / 100% | 2409 / 100% | 2409 / 100% |
| `bright_threshold` | 2358 / 100% | 2425 / 100% | 2358 / 100% | 2358 / 100% |

† Image-size clip (`r_min=r_max=512`).  
‡ Intensity clip (`dark_threshold=255`).

**Working neighborhoods from this stage:**

- `r_min`/`r_max`: stay near **1×** (70 / 200)
- `min_dark_run`: **0.1×–1×** (0–5 looked improving)
- `dark_threshold`: near **1×** (~18)
- `edge_blur`: roughly **0.1×–10×** (1–91), best near baseline
- `radius_outlier_sigma`, `bright_threshold`: low priority for search

---

## 2. Finer one-at-a-time sweep

Others held at baseline. Best per parameter marked with `*`.

### `r_band` (scale both radii together)

| Scale | r_min, r_max | det% | quad | MAE Major | RMSE Major | MAE Angle |
|-------|--------------|------|------|-----------|------------|-----------|
| 0.70 | 49, 140 | 100 | 4795 | 41.9 | 53.5 | 30.1 |
| 0.85 | 60, 170 | 100 | 2738 | 18.6 | 27.5 | 34.7 |
| *1.00* | *70, 200* | *100* | *2425* | *13.4* | *18.9* | *35.7* |
| 1.15 | 80, 230 | 100 | 2737 | 15.2 | 21.2 | 37.4 |
| 1.30 | 91, 260 | 100 | 2850 | 16.0 | 22.6 | 37.8 |

Clear U-shape; ±15% already hurts.

### `dark_threshold`

| Value | quad | MAE Major | RMSE Major | MAE Angle |
|-------|------|-----------|------------|-----------|
| 14 | 2509 | 15.3 | 24.0 | 34.0 |
| 16 | 2526 | 14.1 | 21.3 | 35.8 |
| *18* | *2425* | *13.4* | *18.9* | *35.7* |
| 20 | 2654 | 14.5 | 19.3 | 37.5 |
| 22 | 2925 | 16.1 | 21.5 | 39.5 |
| 24 | 3271 | 17.9 | 24.2 | 40.8 |

Best at **18**; rising above 20 steadily worsens.

### `edge_blur` (odd kernels)

| Value | quad | MAE Major | RMSE Major | MAE Angle |
|-------|------|-----------|------------|-----------|
| 1 | 2694 | 16.8 | 25.6 | 34.5 |
| 3 | 2623 | 15.5 | 23.2 | 35.4 |
| 5 | 2546 | 15.0 | 21.9 | 35.5 |
| 7 | 2484 | 14.1 | 19.5 | 36.0 |
| *9* | *2425* | 13.4 | 18.9 | *35.7* |
| 11 | 2519 | *13.2* | *18.3* | 36.8 |
| 15 | 2676 | 13.8 | 18.7 | 38.4 |
| 21 | 2852 | 14.5 | 19.2 | 40.4 |
| 31+ | ≥3396 | rising | | |

- **Quad loss:** best at **9**
- **MAE / RMSE Major:** slightly better at **11**
- Sweet band: **7–11**

### `min_dark_run` (0–5 only in first fine sweep)

Monotonic improvement 0 → 5 on that short range (see extended sweep below).

---

## 3. Extended `min_dark_run` (0 → 50 = 10×)

Same 400 frames; detection stayed 100% throughout.

| Value | quad | MAE Major | MAE Minor | MAE Angle |
|-------|------|-----------|-----------|-----------|
| 0–1 | 2586 | 14.6 | 9.6 | 36.4 |
| 3 | 2523 | 14.0 | 9.8 | 36.2 |
| 4 | 2477 | 13.7 | 9.6 | 36.0 |
| *5* | *2425* | *13.4* | *9.6* | *35.7* |
| 6 | 2455 | 13.6 | 9.7 | 35.8 |
| 8 | 2481 | 13.6 | 9.6 | 36.1 |
| 10 | 2523 | 13.9 | 9.8 | 36.4 |
| 15 | 2736 | 14.3 | 11.4 | 38.2 |
| 20 | 3035 | 14.6 | 13.2 | 40.9 |
| 30 | 4598 | 19.2 | 24.1 | 47.5 |
| 50 | 12365 | 46.7 | 65.2 | 56.6 |

The 0→5 trend was the **left side of a U-shape**, not open-ended gain. Best at **5**; degradation accelerates after ~20.

---

## Conclusions

1. **Baseline is near a local optimum** on this 400-frame test subset for OAT sweeps.
2. **Highest-value knobs for GridSearch:** `r_min`/`r_max`, `dark_threshold`, `edge_blur`, `min_dark_run`.
3. **Low-value / leave fixed:** `bright_threshold`, `radius_outlier_sigma`, blob-fallback knobs (`blur`, `threshold`, `morph_*`), `temporal_alpha` (unused during non-temporal `detect()` in CV).
4. Suggested compact grid for `DEFAULT_PARAM_GRID` in [`ablation_edge/model.py`](ablation_edge/model.py):

```text
r_min / r_max:     near 70 / 200  (± small if searched)
dark_threshold:    16, 18, 20
min_dark_run:      4, 5  (maybe 6)
edge_blur:         7, 9, 11
```

5. Prefer **relative** neighborhoods around baseline over extreme magnitude scales for production search; extreme scales mainly confirm which knobs break the detector.

---

## Notes

- Tuning data for `train_detector.py` can be subsampled with `--tune-frame-stride` (default every 5th frame in 6–45: 6, 11, …, 41). Sensitivity studies above used **every** labeled frame in the 400-sample eval draw, not the stride filter.
- Failed detections contribute zeros to the quadratic loss (heavily penalized).
- Results are OAT: interactions between parameters are not fully explored; a small joint GridSearch in the compact ranges above is the next step if retuning.
