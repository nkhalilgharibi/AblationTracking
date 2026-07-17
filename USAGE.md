# Using the ablation edge detector with new data

This project fits Fiji-style ellipses to annular ablation microscopy frames and compares them to ImageJ Results CSVs.

## Setup

```bash
cd AblationTracking
python3 -m pip install -r requirements.txt
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

Example header:

```text
 ,Area,Perim.,BX,BY,Width,Height,Major,Minor,Angle
1,…
```

### Frame pairing (important)

TIFF numbers are from the start of the movie. CSV row indices are for the ablation window only.

**Default mapping: CSV frame `k` ↔ TIFF `frame{k+5}`**

| CSV row | TIFF file |
|---------|-----------|
| 1 | `…_frame0006.tif` |
| 2 | `…_frame0007.tif` |
| … | … |
| 40 | `…_frame0045.tif` |

By default the tools only use image frames **6–45** (`DEFAULT_FRAME_MIN` / `DEFAULT_FRAME_MAX` in `ablation_edge/data.py`). If your window differs, pass `--frame-min` / `--frame-max` (and keep CSV indices consistent with the same offset of 5, or change `CSV_TO_IMAGE_FRAME_OFFSET` in code).

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
Overlays: **green** = Fiji GT, **red** = detector, **yellow** = Fiji CSV bounding box.

Optional flags:

```bash
python3 disc_viewer.py discYYYYMMDDNNNN \
  --data-dir Data \
  --model splits/detector_model.json \
  --frame-min 6 \
  --frame-max 45
```

## Running the detector without the viewer

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

## Retraining after adding many new discs

If you add a substantial amount of new labeled data and want updated hyperparameters / split:

```bash
# Deletes / regenerates split only if you remove the existing file first,
# or point --split-path at a new file.
rm -f splits/train_test_split.json   # optional: force a new disc-level split

python3 train_detector.py \
  --data-dir Data \
  --split-path splits/train_test_split.json \
  --tune-samples 400 \
  --tune-frame-stride 5 \
  --cv 3 \
  --search grid \
  --frame-min 6 \
  --frame-max 45
```

Tuning uses sklearn `GridSearchCV` (or `--search random` → `RandomizedSearchCV`) with
**GroupKFold** over discs so frames from the same movie stay in one fold.
By default only every 5th frame in 6–45 is used for the search
(`6, 11, 16, 21, 26, 31, 36, 41`; override with `--tune-frame-stride`, or `1` for all frames).
The search loss is mean quadratic error on all ellipse params
`[BX, BY, Major, Minor, Angle]`:
`mean( (ΔBX)² + (ΔBY)² + (ΔMajor)² + (ΔMinor)² + (ΔAngle)² )`,
where `ΔAngle` is the shortest circular difference on the 180° period.
(OpenCV `fitEllipse` angle is converted to Fiji clockwise-from-+x before scoring.)

This writes:

- `splits/train_test_split.json` — disc-level train/test IDs (no frame leakage)
- `splits/detector_model.json` — tuned detector settings (used by the viewer)

You can then re-open `disc_viewer.py`; it loads the new model by default.

## Checklist for new data

- [ ] Files named `disc… .csv` and `disc…_frameXXXX.tif` with the **same** disc id  
- [ ] CSV has `BX`, `BY`, `Major`, `Minor`, `Angle` (and ideally `Width`, `Height`)  
- [ ] CSV row `1` matches TIFF frame `6` (or adjust frame range / offset)  
- [ ] Frames of interest are present on disk for the range you pass to `--frame-min` / `--frame-max`  
- [ ] Viewer GT (green) looks correctly aligned before trusting metrics or retraining  

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| No labeled frames / empty viewer | CSV↔TIFF pairing wrong, or frames outside 6–45 |
| GT ellipse far from the ring | Wrong CSV for that disc, or BX/BY not Fiji bbox upper-left |
| Disc not found | ID typo; must match `disc*.csv` stem |
| Import / dependency errors | `pip install -r requirements.txt` from repo root |
