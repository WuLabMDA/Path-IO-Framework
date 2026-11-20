# UNI tissue inference (without Macenko normalization)

This script classifies previously extracted H&E WSI patches into eight tissue classes using the **frozen UNI foundation model** and the released, fitted `UNI_8class_classifier.joblib` from `pathhab_UNI_patient_split`. It does not retrain the classifier or perform stain normalization.

## Requirements

Python packages: `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `Pillow`, `scikit-learn`, `joblib`, `scikit-image`, and `tqdm`. Obtain the original UNI model checkpoint separately, according to its distribution terms. Use compatible library versions with those used to create the classifier artifact.

## Input

One directory per slide, containing patches with filenames created by `svs2tile.py`, e.g. `SLIDE_256_x0_0_10_y0_0_20.png`. This filename convention is required for recovering patch coordinates and creating maps.

```
patches/
  SLIDE_A/
    SLIDE_A_256_x0_0_10_y0_0_20.png
    ...
```

## Run

```bash
python predict_tissue_classes.py \
  --patch-root /path/to/patches \
  --output /path/to/UNI_predictions \
  --uni-checkpoint /path/to/UNI/pytorch_model.bin \
  --classifier /path/to/pathhab_UNI_patient_split/UNI_8class_classifier.joblib \
  --device cuda:0 \
  --batch-size 128 \
  --workers 8 \
  --background-threshold 0.997 \
  --save-probs
```

For CPU execution use `--device cpu --workers 0` and reduce the batch size if necessary.

## Outputs

For each slide:
- `<slide>_prediction_256.csv`: `x`, `y`, `x_idx`, `y_idx`, `patch_size`, `pred`, `pred_class`, `prob_max`, and QC columns; optionally `probs`.
- `<slide>_map_256.png`: eight-class patch-grid color map.
- `<slide>_corrupt.csv`: unreadable images, if any.

The root output also contains `UNI_inference_QC.csv` with per-slide counts. Corrupt patches are represented as background-colored cells in the map, but remain marked `readable=0` in the CSV.

The optical-background rule is unchanged: pixels with maximum RGB channel ≥220 and channel range ≤20 are counted as near-white; patches with ≥99.7% such pixels are assigned to Background without UNI inference. This is a heuristic, not a calibrated probability.

The prediction CSV is compatible with the subsequent GeoJSON postprocessing script, which reads `x`, `y`, `x_idx`, `y_idx`, `patch_size`, and `pred`. GeoJSON and patch coordinates must use the same reference coordinate system.

**Security:** Only load `.joblib` files from trusted sources. Joblib loading can execute arbitrary code.
