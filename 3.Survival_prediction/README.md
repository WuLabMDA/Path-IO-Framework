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

Clinical CSV: `case_ID`, `Group` (`train`, `valid`, `test`), `OS`, `OS_Status` and/or `PFS`, `PFS_Status`. Events are coded **1=event, 0=censored**; times must be positive. Group membership must be determined independently of outcomes. Optional external cohorts use the same columns except `Group`.

Patient patch cache names: `<prefix>_<case_ID>.pt`, e.g. `mda_409.pt`. Each file contains `{"patches": FloatTensor[N,7,64,64], "coords": IntTensor[N,2]}`. Channels correspond to Bronchi, Inflammatory, Lung, Necrosis, Stroma, Tumor, Vessel; background is represented by all zeros.

## Train OS models

```bash
python train_survival_models.py \
  --clinical /data/clinical_split.csv \
  --cache /data/patch_cache \
  --output /data/training_runs \
  --endpoint OS \
  --prefix mda \
  --models R18 R34 R50 RX50 \
  --seeds 42 \
  --device cuda:0
```

To train PFS models, change `--endpoint PFS`. Optional external evaluation: `--external mayo:/data/mayo.csv:mayo` (repeatable). External cohorts are **not** used to select checkpoints. `--resume` restores the latest optimizer, gradient-scaler, and model states; random-number-generator state is not saved, so exact stochastic replay is not guaranteed.

Training saves per-epoch checkpoints named `S42_E001.pt`, patient risk CSVs, and `all_results.csv` with `valid_C`. Validation is performed on the `valid` group, not the test group. The script follows the original patient-level Cox objective, patch selection, augmentation, and 32-patient gradient updates. Patient patch sampling uses a maximum of 160 patches by default. This is a computationally expensive training process.

## Export patch risks

```bash
python extract_patch_risks.py \
  --cache /data/patch_cache \
  --training-root /data/training_runs \
  --output /data/patch_risks \
  --endpoint both \
  --models R18 R34 R50 RX50 \
  --seeds 42 \
  --device cuda:0
```

For each architecture, select the checkpoint with **highest validation C-index** at epoch >= 3. Ties use the earliest epoch and then lowest seed. No internal or external C-indices influence selection.

Output files: `OS_4D/<prefix>_<case_ID>.npz` and `PFS_4D/...`, each containing:

- `risk`: float32 array `[N_patches, 4]`, columns ordered R18, R34, R50, RX50.
- `coords`: patch coordinates from the original cache.
- `columns`: labels such as `OS_R18_S42`.

Selected checkpoints are recorded in `OS_selected_checkpoints.csv` and `PFS_selected_checkpoints.csv`.

## Compatibility caveats

1. **Original architecture preserved.** The provided `Macro_networks_new.py` returns `(embedding, hazard)`. The original training code uses `embedding` and an extra `Net.h` head, not the internal hazard output. Both heads are retained for strict checkpoint loading.
2. The custom architecture's pooling layer is called `pool`; the original outer wrapper adds an unused `maxpool=Identity`. We intentionally preserve this behavior for existing checkpoint compatibility.
3. The custom model uses GroupNorm and is **not** interchangeable with torchvision ResNet.
4. The public architecture does not implement ImageNet pretrained initialization. This workflow trains from scratch or loads separately supplied Path-IO checkpoints.
5. Four-model risk outputs are **not** compatible with a Fisher-RSF pipeline trained on six model columns or a different column order. Refit or use a separately locked four-model Fisher-RSF pipeline.
6. **No clinical records, trained weights, or Fisher-RSF model are included.** This module alone does not produce the final Path-IO patient risk score.
7. Use appropriately licensed code and weights for redistribution. Training/test cohort separation and censoring definitions are the user's responsibility.

## Fisher Vector + Random Survival Forest (four CNNs)

`train_fisher_rsf.py` trains the downstream model using four patch-risk columns
(`R18`, `R34`, `R50`, `RX50`) in that order. It fits patch-level mean/SD,
a diagonal 25-component Gaussian mixture, signed-square-root/L2 Fisher
encoding (200 features), a feature scaler, and a random survival forest **only
on the training cohort**. The saved `.joblib` contains every fitted component.
The optional external cohorts are evaluated, never used to fit or select the model.

### Train and save a locked model

```bash
python train_fisher_rsf.py \
  --risk-dir /path/to/OS_4D \
  --mda-csv /path/to/Table_best.csv \
  --mayo-csv /path/to/mayo.csv \
  --cimac-csv /path/to/cimac.csv \
  --roussy-csv /path/to/roussy.csv \
  --output /path/to/rsf_results \
  --endpoint OS
```

The MDA CSV must contain `case_ID`, `Group` (`train`, `valid`, `test`),
`OS`, and `OS_Status` (1=event, 0=censored). Optional external CSVs require
`case_ID`, `OS`, and `OS_Status`. Cache filenames must follow
`mda_<case_ID>.npz`, `mayo_<case_ID>.npz`, etc. Training writes
`OS_locked_Fisher_RSF_pipeline.joblib`, cohort risk CSVs, and a Cox/C-index
summary. For PFS, use `--endpoint PFS`, `PFS_4D` and PFS outcome columns.

### Predict without retraining

```bash
python predict_fisher_rsf.py \
  --pipeline /path/to/OS_locked_Fisher_RSF_pipeline.joblib \
  --risk-dir /path/to/OS_4D \
  --output /path/to/patient_pathio_risk.csv
```

This produces raw RSF risk (`risk`) and training-standardized risk (`risk_z`).
The input must contain patch-level risks generated with the **same four CNN
checkpoints** as those used to fit the saved pipeline, with matching `columns`
identifiers. A final RSF model alone cannot generate patch-level CNN risks from
WSIs. Use the pretrained CNN checkpoints from the GitHub Releases page, placed
under `weights/OS/`, to run the preceding inference stage.

**Compatibility:** A saved 6-model Fisher-RSF pipeline is not interchangeable
with the new four-model pipeline. Retrain the downstream model using the four
CNN outputs, or use the original exact feature set for the original pipeline.

**Security:** Only load `.joblib` files from trusted sources; joblib uses pickle
and can execute code when deserializing untrusted files.
