# Path-IO

**Pathology-Driven Immunotherapy Optimization**

Path-IO is a computational pathology framework for survival risk prediction in patients with metastatic non–small cell lung cancer treated with immune checkpoint inhibitors. This repository contains the **source code and a TCGA-LUAD/LUSC inference tutorial**, but **does not distribute pretrained model weights or patient-level research datasets**.

## Computational workflow

1. **WSI patch extraction and habitat mapping** — UNI feature extraction, eight-class tissue classification (background, bronchi, inflammatory/immune, normal lung, necrosis, stroma, tumor, vessel), GeoJSON contour processing, and habitat map reconstruction.
2. **WSI map preprocessing** — patient-level tissue packing, seven-channel one-hot habitat patch extraction (64 × 64, stride 32).
3. **Survival prediction** — four pretrained CNN architectures (R18, R34, R50, RX50) for patch-level risks, GMM/Fisher vector aggregation, and a locked random survival forest for patient-level continuous risk.
4. **Multimodal integration** — a composite RSF model integrating Path-IO risk, precomputed CT-derived risk, and clinical predictors.
5. **TCGA inference tutorial** — notebook for TCGA-LUAD/LUSC image download, clinical retrieval, inference, and survival evaluation (C-index, continuous-risk Cox HR, 95% CI, and P-value). No Kaplan–Meier plots.

## Repository structure

```text
Path-IO-Framework/
├── README.md
├── requirements.txt
├── .gitignore
├── 1.Patch_extraction_WSI_habitat_map_generation/
│   ├── patch_extraction/
│   ├── tissue_classification/
│   ├── inference/
│   └── habitat_map_generation/
├── 2.WSI_map_preprocessing/
│   ├── tissue_packing.py
│   └── generate_patches.py
├── 3.Survival_prediction/
│   ├── Macro_networks_new.py
│   ├── models.py
│   ├── common.py
│   ├── train_survival_models.py
│   ├── extract_patch_risks.py
│   ├── fisher_rsf.py
│   ├── train_fisher_rsf.py
│   ├── predict_fisher_rsf.py
│   └── weights/README.md
├── 4.Multimodal_integration/
│   ├── train_composite_model.py
│   ├── predict_composite_risk.py
│   └── weights/README.md
└── 5.TCGA_inference_tutorial/
    ├── PathIO_TCGA_LUAD_LUSC_End_to_End_Inference.ipynb
    ├── prepare_tcga_slides.py
    ├── infer_released_cnns.py
    └── checkpoints.example.csv
```

## Setup

```bash
conda create -n pathio_env python=3.10 -y
conda activate pathio_env
# Install a compatible PyTorch build for your system first.
pip install -r requirements.txt
```

Open the individual module READMEs for inputs, processing details, and commands. The TCGA tutorial provides an example workflow. Native OpenSlide libraries and adequate disk space are required for large WSIs.

## Pretrained models — controlled access

**Model weights are not publicly distributed.** This includes the trained UNI tissue classifier, the survival CNN checkpoints, the locked Fisher-RSF artifact, and the composite survival model. Researchers may request access from the corresponding authors, providing their institutional affiliation and a brief description of intended research use. Requests are considered in accordance with applicable institutional policies and third-party model licenses.

The code and tutorial require compatible pretrained models to perform inference. The four CNN outputs must match the locked Fisher-RSF feature order and preprocessing metadata. The repository does not contain the original clinical training datasets or CT prediction model.

## Interpretation and evaluation

Path-IO produces continuous survival risk scores. C-index and Cox hazard ratios, confidence intervals, and P-values require observed survival times and event indicators. TCGA-LUAD/LUSC are not equivalent to the ICI-treated metastatic NSCLC cohorts; tutorial results should not be described as validated immunotherapy treatment-benefit predictions.

## Research use and issues

This implementation is for research, not clinical decision-making. Do not upload protected patient data or restricted models to issues or pull requests. Report reproducible software issues at https://github.com/WuLabMDA/Path-IO-Framework/issues.

## Citation and license

Please cite the associated Path-IO manuscript when available. A repository license should be added after confirmation of institutional distribution terms. Third-party code, models, and datasets retain their own terms.
