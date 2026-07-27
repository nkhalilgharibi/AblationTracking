# Branch metrics comparison

Evaluated on the shared disc-level split (**78** train / **20** test discs), frames **6–45**.

- **newEdit (circular)**: classical tuned with circular `ΔAngle²` (`detector_model.json`).
- **newEdit (sincos)**: classical with quadratic sin/cos(2θ), **unnormalized** geometry (`detector_model_sincos.json`).
- **newEdit (sincos, norm512)**: classical with geometry `/512` + sin/cos(2θ) (`detector_model_sincos_norm512.json`, tune stride=5).
- **newEdit (sincos, norm512, full-grid stride1)**: same loss/grid (24 combos), tune stride=1 (`detector_model_sincos_norm512_full.json`).
- **newEdit-withNN hybrid**: CNN on unnormalized-sincos classical (`hybrid_model.pt`).
- **newEdit-withNN hybrid (norm512)**: CNN on norm512 classical (`hybrid_model_norm512.pt`).
- **hybrid wide+aug (full classical)**: encoder `(64,128,256)` / MLP `(256,128)` + photometric/jitter (`hybrid_model_full_wide_aug.pt`).
- **hybrid deep+aug (full classical)**: encoder `(16,32,64,128)` / MLP `(128,64)` + aug (`hybrid_model_full_deep_aug.pt`).
- **withNN (hybrid / classical)**: older branch reference.

Metrics: MAE = mean absolute error; RMSE from signed residuals (Angle uses shortest circular residual on a 180° period).

## Compact test summary

| Branch | Test n | Det% | MAE Major | RMSE Major | MAE Minor | RMSE Minor | MAE Angle | RMSE Angle |
|--------|--------|------|-----------|------------|-----------|------------|-----------|------------|
| newEdit (circular ΔAngle²) | 800 | 100.0% | 14.06 | 19.47 | 9.96 | 12.62 | 35.48 | 43.11 |
| newEdit (sincos, unnormalized) | 800 | 100.0% | 14.12 | 19.60 | 10.05 | 12.83 | 35.38 | 43.22 |
| newEdit (sincos, norm512) | 800 | 100.0% | 13.22 | 19.96 | 9.27 | 11.98 | 34.15 | 41.74 |
| newEdit (sincos, norm512, full-grid stride1) | 800 | 100.0% | 14.37 | 20.67 | 9.87 | 12.52 | 34.75 | 42.31 |
| newEdit-withNN hybrid (on unnormalized sincos) | 800 | 100.0% | 14.69 | 19.20 | 8.50 | 10.92 | 11.27 | 16.08 |
| newEdit-withNN hybrid (on norm512 classical) | 800 | 100.0% | 13.07 | 18.01 | 8.81 | 11.36 | 10.49 | 13.93 |
| hybrid wide+aug (on full-grid classical) | 800 | 100.0% | 15.55 | 20.36 | 9.15 | 11.18 | 11.36 | 15.09 |
| hybrid deep+aug (on full-grid classical) | 800 | 100.0% | 15.38 | 20.44 | 8.98 | 11.13 | 11.83 | 16.74 |
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

## feature/ablation-edge-detector-newEdit (sincos, unnormalized)

Tuning loss: `mean(dBX²+dBY²+dMajor²+dMinor²+(Δsin2θ)²+(Δcos2θ)²)`.  
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

## feature/ablation-edge-detector-newEdit (sincos, norm512)

Tuning loss: `mean((dBX/512)²+(dBY/512)²+(dMajor/512)²+(dMinor/512)²+(Δsin2θ)²+(Δcos2θ)²)`.  
Artifact: `splits/detector_model_sincos_norm512.json` (`edge_blur=11`, `dark_threshold=16`, `min_dark_run=4`).

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.61 | 15.11 |
| BY | 3.49 | 5.65 |
| Major | 14.96 | 22.99 |
| Minor | 8.60 | 12.75 |
| Angle | 27.44 | 34.59 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 4.96 | 6.40 |
| BY | 2.59 | 3.43 |
| Major | 13.22 | 19.96 |
| Minor | 9.27 | 11.98 |
| Angle | 34.15 | 41.74 |

---

## feature/ablation-edge-detector-newEdit (sincos, norm512, full-grid stride1)

Tuning loss: same norm512 sincos as above.  
Search: full `GridSearchCV` over all **24** `DEFAULT_PARAM_GRID` combos, `GroupKFold` k=4, **tune-frame-stride=1** (all frames 6–45).  
Artifact: `splits/detector_model_sincos_norm512_full.json` (`edge_blur=7`, `dark_threshold=18`, `min_dark_run=4`).  
CV quadratic loss: **1.0456** (prior stride-5 norm512 was 1.0510). Search wall time ~1h 21m (`n_jobs=4`).

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.50 | 13.95 |
| BY | 3.64 | 5.76 |
| Major | 14.89 | 22.79 |
| Minor | 9.63 | 13.79 |
| Angle | 27.48 | 34.57 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 4.83 | 6.28 |
| BY | 2.70 | 3.57 |
| Major | 14.37 | 20.67 |
| Minor | 9.87 | 12.52 |
| Angle | 34.75 | 42.31 |

---

## feature/ablation-edge-detector-newEdit-withNN (hybrid on unnormalized sincos)

Classical base: `splits/detector_model_sincos.json`.  
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

## feature/ablation-edge-detector-newEdit-withNN (hybrid on norm512 classical)

Classical base: `splits/detector_model_sincos_norm512.json`.  
CNN: encoder `(32,64,128)`, MLP `(128,64)`, dropout `0.2`.  
Artifacts: `splits/hybrid_model_norm512.pt`, `splits/hybrid_model_norm512.json`.

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.89 | 11.17 |
| BY | 3.07 | 4.57 |
| Major | 11.41 | 15.93 |
| Minor | 7.36 | 11.02 |
| Angle | 6.62 | 12.10 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.58 | 7.64 |
| BY | 2.82 | 3.71 |
| Major | 13.07 | 18.01 |
| Minor | 8.81 | 11.36 |
| Angle | 10.49 | 13.93 |

---

## hybrid wide+aug (on full-grid classical)

Classical base: `splits/detector_model_sincos_norm512_full.json`.  
CNN: encoder `(64,128,256)`, MLP `(256,128)`, dropout `0.2`, **50** epochs, train-only photometric + ±8px jitter (`--augment`).  
Artifacts: `splits/hybrid_model_full_wide_aug.pt`, `splits/hybrid_model_full_wide_aug.json`.  
Best val loss: **0.0145**.

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 4.74 | 8.14 |
| BY | 3.07 | 4.39 |
| Major | 11.21 | 14.88 |
| Minor | 7.78 | 11.53 |
| Angle | 6.29 | 12.84 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.72 | 9.01 |
| BY | 2.94 | 3.79 |
| Major | 15.55 | 20.36 |
| Minor | 9.15 | 11.18 |
| Angle | 11.36 | 15.09 |

---

## hybrid deep+aug (on full-grid classical)

Classical base: `splits/detector_model_sincos_norm512_full.json` (same as wide).  
CNN: encoder `(16,32,64,128)`, MLP `(128,64)`, dropout `0.2`, **50** epochs, same `--augment`.  
Artifacts: `splits/hybrid_model_full_deep_aug.pt`, `splits/hybrid_model_full_deep_aug.json`.  
Best val loss: **0.0141**.

Train samples: **3120** (detected 3120 / 3120, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.06 | 9.20 |
| BY | 3.19 | 4.71 |
| Major | 11.69 | 15.65 |
| Minor | 8.20 | 12.04 |
| Angle | 5.71 | 11.58 |

Test samples: **800** (detected 800 / 800, 100.0%)

| Param | MAE | RMSE |
|-------|-----|------|
| BX | 5.47 | 8.49 |
| BY | 2.95 | 3.90 |
| Major | 15.38 | 20.44 |
| Minor | 8.98 | 11.13 |
| Angle | 11.83 | 16.74 |

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

---

## Takeaways (full-grid classical → width vs depth + aug)

- **Full-grid classical (stride=1)** found different params (`edge_blur=7`, `dark_threshold=18`, `min_dark_run=4`) and a slightly better CV loss (1.0456 vs 1.0510), but **held-out test axes/angle did not improve** vs prior stride-5 norm512 (`Major` 14.37 vs 13.22). Thorough search ≠ better test classical.
- **Wide+aug vs deep+aug** on that full-grid base (same aug, 50 epochs): differences are small. Deep is slightly better on test **Major/Minor**; wide is slightly better on test **Angle**. Neither beats the prior hybrid on the stronger stride-5 classical (`hybrid_model_norm512`: Major 13.07 / Angle 10.49).
- **Verdict:** on this base, **neither width nor depth clearly wins**; classical quality dominates. Prefer expanding classical search (e.g. `r_min`/`r_max`) or sticking with the stride-5 norm512 classical + baseline hybrid over stacking a wider/deeper CNN on the weaker full-grid base.
