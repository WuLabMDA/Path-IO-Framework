# Module 5 — TCGA-LUAD/LUSC inference tutorial

The notebook `PathIO_TCGA_LUAD_LUSC_End_to_End_Inference.ipynb` demonstrates the Path-IO workflow using TCGA-LUAD or TCGA-LUSC H&E slides and clinical records from the [NCI GDC Data Portal](https://portal.gdc.cancer.gov/).

## Workflow
1. Download public diagnostic H&E WSIs and TCGA clinical records.
2. Prepare slide-to-patient identifiers and extract WSI patches (Module 1).
3. Obtain UNI embeddings and predict tissue classes; reconstruct habitat maps (Module 1).
4. Pack patient-level maps and extract seven-channel 64 × 64 habitat patches (Module 2).
5. Load four pretrained survival CNNs and generate patch-level risks (Module 3).
6. Apply the corresponding locked RSF pipeline to generate continuous patient-level risk scores.
7. Match patient predictions to clinical survival outcomes and compute C-index, continuous-risk Cox HR, 95% CI, and Cox P-value. 

## Model access
Pretrained UNI, tissue classifier, CNN checkpoints, and Fisher-RSF model files are **not included** and are available only through approved controlled access. 

## Important limitations
TCGA-LUAD/LUSC are not the same population as the metastatic NSCLC ICI-treated development cohorts. The tutorial demonstrates computational inference and exploratory survival evaluation, not validation of immunotherapy treatment benefit. The composite model in Module 4 also requires compatible CT-derived risks and clinical predictors, which TCGA H&E alone does not provide.

