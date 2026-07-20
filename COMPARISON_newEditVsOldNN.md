# Branch metrics comparison

Evaluated on the shared disc-level split (**78** train / **20** test discs), frames **6–45**, using each branch’s checked-in artifacts (no retrain).

- **newEdit**: classical OpenCV detector (`detector_model.json`, no Ridge).
- **withNN (hybrid)**: classical + CNN residual (`hybrid_model.pt`).
- **withNN (classical only)**: same branch classical detector without CNN.

Metrics: MAE = mean absolute error; RMSE from signed residuals (Angle uses shortest circular residual on a 180° period).

## Compact test summary

| Branch | Test n | Det% | MAE Major | RMSE Major | MAE Minor | RMSE Minor | MAE Angle | RMSE Angle |
|--------|--------|------|-----------|------------|-----------|------------|-----------|------------|
| feature/ablation-edge-detector-newEdit | 800 | 100.0% | 14.06 | 19.47 | 9.96 | 12.62 | 35.48 | 43.11 |
| feature/ablation-edge-detector-withNN (hybrid) | 800 | 100.0% | 13.82 | 17.96 | 8.12 | 10.22 | 11.65 | 15.99 |
| feature/ablation-edge-detector-withNN (classical only) | 800 | 100.0% | 16.53 | 22.32 | 11.53 | 15.15 | 38.50 | 45.99 |

---

## feature/ablation-edge-detector-newEdit

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
