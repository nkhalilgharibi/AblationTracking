# Using the ablation edge detector

This project fits Fiji-style ellipses to annular ablation microscopy frames and compares them to ImageJ Results CSVs. The kept pipeline is:

1. **Classical** OpenCV ring fit (`splits/detector_model.json`) — GridSearch with geometry `/512` and sin/cos(2θ) angle loss  
2. **Hybrid** CNN residual on top (`splits/hybrid_model.pt`) — optional refinement of the classical ellipse  

See [SUMMARY.md](SUMMARY.md) for what was tried and why this recipe was kept.

## Setup

```bash
cd AblationTracking
python3 -m pip install -r requirements.txt
# Hybrid CNN also needs PyTorch:
python3 -m pip install torch
```

Default data folder is `Data/` at the repo root. Override with `--data-dir` on the CLIs below.

## Expected file layout

For each disc (movie), put files in the same directory (e.g. `Data/`):

| File | Naming | Role |
|------|--------|------|
| TIFF frames | `{disc_id}_frameXXXX.tif` | One grayscale image per frame (`XXXX` is zero-padded, e.g. `0006`) |
| Fiji CSV | `{disc_id}.csv` | Ellipse fits for that disc (ImageJ Results table) |

**Disc ID** must match: start with `disc`, then digits only, e.g. `disc202302140302`.

Example:

```text
Data/
  disc202302140302.csv
  disc202302140302_frame0001.tif
  disc202302140302_frame0002.tif
  …
  disc202302140302_frame0045.tif
```

### CSV format

Export Fiji / ImageJ **Fit Ellipse** Results with at least these columns:

- First column: frame index within the ablation window (`1`, `2`, `3`, …)
- `BX`, `BY`, `Width`, `Height`, `Major`, `Minor`, `Angle`

`BX`/`BY` are the **upper-left** of the axis-aligned bounding box (Fiji convention). The code converts them to the ellipse center internally.

### Frame pairing (important)

TIFF numbers are from the start of the movie. CSV row indices are for the ablation window only.

**Default mapping: CSV frame `k` ↔ TIFF `frame{k+5}`**

| CSV row | TIFF file |
|---------|-----------|
| 1 | `…_frame0006.tif` |
| 2 | `…_frame0007.tif` |
| … | … |
| 40 | `…_frame0045.tif` |

By default the tools only use image frames **6–45**. If your window differs, pass `--frame-min` / `--frame-max` (and keep CSV indices consistent with the same offset of 5, or change `CSV_TO_IMAGE_FRAME_OFFSET` in `ablation_edge/data.py`).

## Adding a new disc

1. Split the movie into single-frame TIFFs named `{disc_id}_frameXXXX.tif` (you can adapt `RenameAndSaveFrames.bash` if you have multi-page aligned TIFFs).
2. Export the Fiji ellipse Results CSV as `{disc_id}.csv` into the same folder.
3. Confirm pairing: CSV row `1` should match `…_frame0006.tif` visually (green GT overlay in the viewer).
4. Inspect with the viewer (no retrain required for a quick check):

```bash
python3 disc_viewer.py disc202302140302
# or
python3 disc_viewer.py 202302140302 --data-dir Data
```

Controls: enter disc ID → **Load**, then slider / ← → to step frames.  
Overlays: **green** = Fiji GT, **red** = classical detector, **yellow** = Fiji CSV bounding box.

```bash
python3 disc_viewer.py discYYYYMMDDNNNN \
  --data-dir Data \
  --model splits/detector_model.json \
  --frame-min 6 \
  --frame-max 45
```

## Classical inference (Python)

```python
from pathlib import Path
from ablation_edge.data import load_gray_image
from ablation_edge.train import load_trained_detector

detector = load_trained_detector("splits/detector_model.json")
image = load_gray_image(Path("Data/disc202302140302_frame0006.tif"))
result = detector.detect(image)  # BX/BY = ellipse center; Angle = Fiji clockwise from +x
print(result)
```

For a full disc with temporal smoothing:

```python
from ablation_edge.data import AblationDataset, load_gray_image
from ablation_edge.train import load_trained_detector

detector = load_trained_detector("splits/detector_model.json")
detector.reset_temporal_state()
for sample in AblationDataset("Data", ["disc202302140302"]):
    image = load_gray_image(sample.image_path)
    pred = detector.detect(image, temporal_smooth=True)
```

## Hybrid inference (Python)

Requires PyTorch. Classical fit first, then CNN residual:

```python
from pathlib import Path
from ablation_edge.data import load_gray_image
from ablation_edge.hybrid_refine import HybridEllipsePredictor

predictor = HybridEllipsePredictor(
    checkpoint_path="splits/hybrid_model.pt",
    detector_model_path="splits/detector_model.json",
)
image = load_gray_image(Path("Data/disc202302140302_frame0006.tif"))
result = predictor.detect(image)  # Fiji-style EllipseResult after CNN residual
print(result)
```

## Retrain classical detector

After adding many new labeled discs:

```bash
# Optional: force a new disc-level split
rm -f splits/train_test_split.json

python3 train_detector.py \
  --data-dir Data \
  --split-path splits/train_test_split.json \
  --tune-frame-stride 5 \
  --cv 4 \
  --search grid \
  --model-out splits/detector_model.json \
  --frame-min 6 \
  --frame-max 45
```

Tuning uses `GridSearchCV` with **GroupKFold** over discs (no frame leakage).  
Default search uses every 5th frame in 6–45. Loss is:

`mean( (ΔBX/512)² + (ΔBY/512)² + (ΔMajor/512)² + (ΔMinor/512)² + (Δsin2θ)² + (Δcos2θ)² )`

Writes / updates:

- `splits/train_test_split.json` — disc-level train/test IDs  
- `splits/detector_model.json` — tuned OpenCV params  

## Retrain hybrid CNN

After the classical model is updated (or when you want a fresh residual net):

```bash
python3 train_hybrid.py \
  --detector-model splits/detector_model.json \
  --model-basename hybrid_model \
  --encoder-channels 32,64,128 \
  --mlp-hidden 128,64 \
  --epochs 30
```

Default architecture matches the kept checkpoint: encoder `(32,64,128)`, MLP `(128,64)`.  
Writes `splits/hybrid_model.pt` and `splits/hybrid_model.json`.

Smoke test:

```bash
python3 train_hybrid.py --quick
```

## Checklist for new data

- [ ] Files named `disc… .csv` and `disc…_frameXXXX.tif` with the **same** disc id  
- [ ] CSV has `BX`, `BY`, `Major`, `Minor`, `Angle` (and ideally `Width`, `Height`)  
- [ ] CSV row `1` matches TIFF frame `6` (or adjust frame range / offset)  
- [ ] Frames of interest are present for `--frame-min` / `--frame-max`  
- [ ] Viewer GT (green) looks correctly aligned before trusting metrics or retraining  

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| No labeled frames / empty viewer | CSV↔TIFF pairing wrong, or frames outside 6–45 |
| GT ellipse far from the ring | Wrong CSV for that disc, or BX/BY not Fiji bbox upper-left |
| Disc not found | ID typo; must match `disc*.csv` stem |
| Import / dependency errors | `pip install -r requirements.txt` (+ `torch` for hybrid) |
| Hybrid import fails | Install PyTorch: `pip install torch` |
