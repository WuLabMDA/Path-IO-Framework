# Path-IO — Module 3: Four-ResNet survival prediction

This module trains four custom GroupNorm CNN architectures on **patient-level seven-channel habitat patch caches** from Module 2 and exports **per-patch continuous risk values** for subsequent Fisher-vector / random survival forest modeling.

## Files

- `Macro_networks_new.py` — checkpoint-compatible custom R18, R34, R50, RX50 definitions, adapted from the provided source; unused architectures omitted.
- `models.py` — original outer Path-IO survival head (`self.h`) and patient mean-risk aggregation.
- `common.py` — cohort data loading, patch sampling, and concordance index.
- `train_survival_models.py` — Cox partial-likelihood training and validation-based model selection.
- `extract_patch_risks.py` — selects the highest validation-C-index epoch for each architecture and produces four-column patch risk arrays.
- `weights/` — place released checkpoint files here; **weights are not bundled**.

## Install

Python 3.10+ recommended. Install PyTorch appropriate for your CUDA setup, then:

```bash
pip install -r requirements.txt
```

## Input data

Clinical CSV: `case_ID`, `Group` (`train`, `valid`, `test`), `OS`, `OS_Status` and/or `PFS`, `PFS_Status`. Events are coded **1=event, 0=censored**; times must be positive.


## Train OS models

```bash
python train_survival_models.py \
  --clinical /data/clinical_split.csv \
  --cache /data/patch_cache \
  --output /data/training_runs \
  --endpoint OS \
  --models R18 R34 R50 RX50 \
  --device cuda:0
```


## Export patch risks

```bash
python extract_patch_risks.py \
  --cache /data/patch_cache \
  --training-root /data/training_runs \
  --output /data/patch_risks \
  --endpoint both \
  --models R18 R34 R50 RX50 \
  --device cuda:0
```


Output files: `OS_4D/<prefix>_<case_ID>.npz` and `PFS_4D/...`, each containing:

- `risk`: float32 array `[N_patches, 4]`, columns ordered R18, R34, R50, RX50.
- `coords`: patch coordinates from the original cache.



## Random Survival Forest (four CNNs)

`train_fisher_rsf.py` trains the downstream model using four patch-risk columns
(`R18`, `R34`, `R50`, `RX50`) in that order.The saved `.joblib` contains every fitted component.


### Train and save a locked model

```bash
python train_fisher_rsf.py \
  --risk-dir /path/to/OS_4D \
  --output /path/to/rsf_results \
  --endpoint OS
```

The CSV must contain `case_ID`, `Group` (`train`, `valid`, `test`),
`OS`, and `OS_Status` (1=event, 0=censored). Optional external CSVs require
`case_ID`, `OS`, and `OS_Status`. 

### Predict without retraining

```bash
python predict_fisher_rsf.py \
  --pipeline /path/to/RSF_pipeline.joblib \
  --risk-dir /path/to/OS_4D \
  --output /path/to/patient_pathio_risk.csv
```


