# Summary: what we tried and what we kept

This document records experiment history on the ablation-edge detector branch and the recipe retained for main.

## What this branch keeps

| Piece | Path / setting | Notes |
|-------|----------------|-------|
| Classical detector | `splits/detector_model.json` | Sincos(2θ) angle loss, geometry `/512`, GridSearch over 24 param combos, **tune-frame-stride=5**, GroupKFold cv=4 |
| Best classical params | `edge_blur=11`, `dark_threshold=16`, `min_dark_run=4` | From the stride-5 norm512 search |
| Hybrid CNN | `splits/hybrid_model.pt` / `.json` | Residual on the classical fit; encoder `(32,64,128)`, MLP `(128,64)`, dropout `0.2` |
| Split | `splits/train_test_split.json` | 78 train / 20 test discs (no frame leakage) |
| CLIs | `train_detector.py`, `train_hybrid.py`, `disc_viewer.py` | Defaults point at the artifacts above |
| Docs | `USAGE.md` (how to run), this file (history) | |

Pipeline:

```text
Data (CSV + TIFFs)
  → classical OpenCV ellipse fit (detector_model.json)
  → optional CNN residual (hybrid_model.pt)
  → viewer / metrics
```

## What we tried

### Classical tuning losses / schedules

1. **Circular ΔAngle²** (geometry unnormalized in early artifacts) — workable baseline, weaker Angle/axes than later recipes.  
2. **Sincos(2θ) without `/512` geometry** — similar to circular on held-out axes; Angle still poor classically.  
3. **Sincos + geometry `/512`, stride=5** — **best classical** on the shared test set (lower Major/Minor/Angle MAE than 1–2).  
4. **Same 24-combo grid, stride=1 (all frames 6–45)** — slightly better CV loss, but **worse held-out test** axes than stride-5; different OpenCV params (`edge_blur=7`, `dark_threshold=18`). Thorough search did not beat stride-5 on test.

### Hybrid / CNN

1. CNN residual on unnormalized-sincos classical — large Angle gain; axes mixed.  
2. CNN residual on stride-5 norm512 classical (`hybrid_model`, kept) — **best overall** among hybrids tried (test Major ≈ 13.1, Angle ≈ 10.5).  
3. Wider net `(64,128,256)/(256,128)` + photometric/jitter aug on the weaker full-grid classical — did not beat (2).  
4. Deeper net `(16,32,64,128)/(128,64)` + same aug on full-grid classical — similarly no clear win over (2); width vs depth differences were small.

### Other ideas that did not stick

- **Ridge residual** on the classical detector — unused once the CNN hybrid existed; removed.  
- **Photometric + crop-jitter augmentation** — did not outperform the unaugmented baseline hybrid on the winning classical base.  
- Expanding classical search to `r_min`/`r_max` was noted as a possible next step but not kept in this cleanup.

## Takeaways

- Classical quality dominates: if the OpenCV ellipse fit is far off, the CNN residual cannot fully recover.  
- Full-grid stride-1 retuning did not improve held-out classical metrics vs stride-5.  
- On the full-grid base, neither “more width” nor “more depth” clearly won; both trailed the baseline hybrid on the stronger classical.  
- Keep stride-5 sincos-norm512 classical + mid-size hybrid; retrain hybrid after any classical change.

## What was removed in this cleanup

**Artifacts:** circular / unnormalized / full-grid classical JSONs; hybrids on those bases; wide/deep+aug and smoke checkpoints.

**Code:** Ridge correction path; circular GridSearch loss CLI; `--augment` hybrid path; `compare_branches.py`.

**Docs:** `COMPARISON_newEditVsOldNN.md`, `SENSITIVITY.md` (replaced by this summary + updated `USAGE.md`).

Canonical names after cleanup: `splits/detector_model.json` and `splits/hybrid_model.{pt,json}` (formerly `*_sincos_norm512` / `*_norm512`).
