# Branch metrics comparison

Evaluated on the shared disc-level split (**78** train / **20** test discs), frames **6–45**.

- **newEdit (circular)**: classical OpenCV detector tuned with circular `ΔAngle²` (`detector_model.json`).
- **newEdit (sincos)**: classical retuned with quadratic sin/cos(2θ) angle residual (`detector_model_sincos.json`).
- **newEdit-withNN hybrid**: this branch’s classical (sincos artifact) + modular CNN residual (`hybrid_model.pt`); equal-weight quadratic sincos loss.
- **withNN (hybrid / classical)**: older branch artifacts (unchanged reference).

Metrics: MAE = mean absolute error; RMSE from signed residuals (Angle uses shortest circular residual on a 180° period).

## Compact test summary

| Branch | Test n | Det% | MAE Major | RMSE Major | MAE Minor | RMSE Minor | MAE Angle | RMSE Angle |
|--------|--------|------|-----------|------------|-----------|------------|-----------|------------|
| newEdit (circular ΔAngle²) | 800 | 100.0% | 14.06 | 19.47 | 9.96 | 12.62 | 35.48 | 43.11 |
| newEdit (sincos 2θ quadratic) | 800 | 100.0% | 14.12 | 19.60 | 10.05 | 12.83 | 35.38 | 43.22 |
| newEdit-withNN hybrid (sincos quadratic CNN) | 800 | 100.0% | 14.69 | 19.20 | 8.50 | 10.92 | 11.27 | 16.08 |
| withNN (hybrid) | 800 | 100.0% | 13.82 | 17.96 | 8.12 | 10.22 | 11.65 | 15.99 |
| withNN (classical only) | 800 | 100.0% | 16.53 | 22.32 | 11.53 | 15.15 | 38.50 | 45.99 |

---

## feature/ablation-edge-detector-newEdit (circular ΔAngle²)

Tuning loss: `mean(dBX²+dBY²+dMajor²+dMinor²+dAngle²)` with circular `dAngle`.  
Artifact: `splits/detector_model.json` (`edge_blur=7`, `dark_threshold=19`, `min_dark_run=5`).

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.37 | 13.87 |
| BY | 3.57 | 5.75 |
| Major | 14.57 | 22.57 |
| Minor | 9.79 | 13.80 |
| Angle | 27.68 | 34.89 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 4.86 | 6.14 |
| BY | 2.68 | 3.54 |
| Major | 14.06 | 19.47 |
| Minor | 9.96 | 12.62 |
| Angle | 35.48 | 43.11 |

---

## feature/ablation-edge-detector-newEdit (sincos 2θ quadratic)

Tuning loss: `mean(dBX²+dBY²+dMajor²+dMinor²+(sin2θ_pred−sin2θ_gt)²+(cos2θ_pred−cos2θ_gt)²)`.  
Artifact: `splits/detector_model_sincos.json` (`edge_blur=9`, `dark_threshold=19`, `min_dark_run=4`).

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.49 | 14.54 |
| BY | 3.64 | 5.81 |
| Major | 14.27 | 22.35 |
| Minor | 9.82 | 13.84 |
| Angle | 27.79 | 35.12 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 4.76 | 6.02 |
| BY | 2.82 | 3.71 |
| Major | 14.12 | 19.60 |
| Minor | 10.05 | 12.83 |
| Angle | 35.38 | 43.22 |

---

## feature/ablation-edge-detector-newEdit-withNN (hybrid)

Classical base: `splits/detector_model_sincos.json` (this branch).  
CNN: modular encoder `(32,64,128)`, MLP `(128,64)`, dropout `0.2`.  
Train loss: equal-weight quadratic refined-vs-GT with sin/cos(2θ).  
Artifacts: `splits/hybrid_model.pt`, `splits/hybrid_model.json`.

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.90 | 12.19 |
| BY | 3.38 | 4.92 |
| Major | 11.39 | 15.48 |
| Minor | 7.61 | 11.07 |
| Angle | 6.88 | 13.24 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.73 | 8.06 |
| BY | 3.37 | 4.35 |
| Major | 14.69 | 19.20 |
| Minor | 8.50 | 10.92 |
| Angle | 11.27 | 16.08 |

---

## feature/ablation-edge-detector-withNN (hybrid)

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 4.82 | 13.76 |
| BY | 3.00 | 4.74 |
| Major | 10.86 | 15.75 |
| Minor | 7.06 | 10.58 |
| Angle | 7.03 | 12.44 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.60 | 7.31 |
| BY | 4.30 | 5.60 |
| Major | 13.82 | 17.96 |
| Minor | 8.12 | 10.22 |
| Angle | 11.65 | 15.99 |

---

## feature/ablation-edge-detector-withNN (classical only)

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.67 | 14.72 |
| BY | 3.96 | 6.41 |
| Major | 15.53 | 24.19 |
| Minor | 12.20 | 17.28 |
| Angle | 29.02 | 36.72 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.04 | 6.43 |
| BY | 3.64 | 4.98 |
| Major | 16.53 | 22.32 |
| Minor | 11.53 | 15.15 |
| Angle | 38.50 | 45.99 |
