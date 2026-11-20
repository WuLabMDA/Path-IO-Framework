# UNI-based eight-class tissue classification

This module extracts frozen UNI ViT-L/16 patch embeddings and optionally trains a linear eight-class classifier. It does **not** perform or enforce a patient split: users are responsible for defining their own training and testing partitions and preventing leakage between patients/slides.

Class order (labels 0–7): `Background`, `Bronchi`, `Inflammatory`, `Lung`, `Necrosis`, `Stroma`, `Tumor`, `Vessel`.

## Inputs

- Download the authorized UNI checkpoint separately and supply its path with `--uni-checkpoint`. This implementation expects the `timm` ViT-L/16 state dict matching the original code.
- For labelled extraction or evaluation, arrange patches in eight named subfolders under each split root, e.g. `my_train/Tumor/*.png`, `my_test/Stroma/*.png`. Use **your own** train/test split; the script never creates one.
- For prediction, point `--input` to any directory of patches (recursive scan).
- Your pretrained classifier is `pathhab_UNI_patient_split/UNI_8class_classifier.joblib`. Place a copy in `weights/UNI_8class_classifier.joblib` if redistribution is permitted. The file includes both `StandardScaler` and logistic regression.

## Setup

Install compatible versions of PyTorch, torchvision, timm, NumPy, pandas, Pillow, scikit-learn, and joblib. Pin exact versions of PyTorch/timm/scikit-learn used for the released UNI checkpoint and classifier before publishing a reproducible environment.

## Extract UNI features from your training split

```bash
python uni_tissue_classifier.py --mode extract \
  --input /path/to/my_train --split-name train \
  --uni-checkpoint /path/to/UNI/pytorch_model.bin \
  --features-dir features
```

## Option A: Train a new classifier on your own split

```bash
python uni_tissue_classifier.py --mode train \
  --features-dir features --split-name train --output-dir outputs
```

## Option B: Evaluate the released Path-IO classifier on your test split

```bash
python uni_tissue_classifier.py --mode evaluate \
  --input /path/to/my_test \
  --uni-checkpoint /path/to/UNI/pytorch_model.bin \
  --classifier weights/UNI_8class_classifier.joblib \
  --output-dir evaluation
```

For evaluation of a newly trained classifier, substitute `outputs/UNI_8class_classifier.joblib`.

## Predict tissue classes for new WSI patches

```bash
python uni_tissue_classifier.py --mode predict \
  --input /path/to/wsi_patches \
  --uni-checkpoint /path/to/UNI/pytorch_model.bin \
  --classifier weights/UNI_8class_classifier.joblib \
  --output-dir predictions
```

`predictions/tissue_predictions.csv` contains filenames and predicted labels/classes. This is **not** yet the habitat-map reconstruction step; a separate map reconstruction script must map predictions to WSI coordinates according to the patch filename conventions.

## Notes

- UNI images are resized to 224×224 and normalized with ImageNet mean/std, matching the supplied training implementation.
- This code does not distribute any patient data, original patch images, cached UNI features, or study manifests.
- Classifier loading with joblib requires a trusted file and a compatible scikit-learn environment.
- Confirm that the released classifier is the same fitted artifact used for the reported habitat maps.
