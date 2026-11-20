# Module 4 — Multimodal integration

This module trains and applies a composite Random Survival Forest model using **CT-derived risk scores, Path-IO pathology risk scores, and clinical predictors**. CT risk scores must already be computed; raw CT-image inference is not included.

## Files
- `train_composite_model.py`: fit the composite RSF model from CT and pathology/clinical tables.
- `predict_composite_risk.py`: load a fitted model and predict composite risk.
- `requirements.txt`: dependencies.
- `weights/README.md`: controlled-access information.

## Training example
```bash
python train_composite_model.py --ct CT_data.csv --he combined_HE_OS_PFS_risk.csv --output weights
```

The training data are not included. For prediction, request the corresponding fitted model through controlled access and use the feature schema required by the model. Outcome-specific variable names in scripts are technical identifiers; the public README focuses on the general survival prediction framework.
