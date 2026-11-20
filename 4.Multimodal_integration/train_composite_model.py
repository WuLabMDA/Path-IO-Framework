"""Train and save a CT + pathology + clinical composite survival model.

Input files:
  CT CSV: MRN, CT_OS_risk
  Pathology CSV: MRN, HE_OS_risk, OS, OS_Status, Cohort, clinical columns
The supplied implementation fits the OS endpoint; endpoint details are kept in
technical metadata rather than generalized in the public-facing README.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

CATEGORICAL = ["Gender", "PDL1", "IO_Strategy", "Pathology", "Tobacco",
               "StageAtICI", "LiverMet", "AdrenalMet", "BoneMet", "ECOG", "NMO"]
NUMERIC = ["Age", "Line_of_IO", "BrainMet", "Radiation", "Surgery"]
FEATURES = ["CT_OS_risk", "HE_OS_risk"] + NUMERIC + CATEGORICAL
PARAMS = dict(n_estimators=100, min_samples_split=10, min_samples_leaf=5,
              max_features="log2", max_depth=None, random_state=42,
              n_jobs=-1, low_memory=False)


def clean_data(data):
    data = data.copy()
    for col in CATEGORICAL:
        data[col] = (data[col].astype('string').str.strip().str.lower()
                     .replace(['unknown', 'nan', 'none', 'na', 'n/a', ''], np.nan)
                     .astype(object))
        data[col] = data[col].where(pd.notna(data[col]), np.nan)
    for col in NUMERIC + ['CT_OS_risk', 'HE_OS_risk', 'OS', 'OS_Status']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


def build_model():
    prep = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler())
        ]), ['CT_OS_risk', 'HE_OS_risk'] + NUMERIC),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL),
    ])
    return Pipeline([('prep', prep), ('rsf', RandomSurvivalForest(**PARAMS))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct', required=True, help='CT risk CSV with MRN and CT_OS_risk')
    p.add_argument('--he', required=True, help='Path-IO risk, clinical and outcome CSV')
    p.add_argument('--output', default='weights', help='Directory for fitted model')
    p.add_argument('--discovery-label', default='Discovery')
    args = p.parse_args()

    ct = pd.read_csv(args.ct, dtype={'MRN': str})
    he = pd.read_csv(args.he, dtype={'MRN': str})
    for df in (ct, he):
        df.columns = df.columns.str.strip()
    required_ct = {'MRN', 'CT_OS_risk'}
    required_he = {'MRN', 'HE_OS_risk', 'OS', 'OS_Status', 'Cohort', *NUMERIC, *CATEGORICAL}
    for name, df, req in [('CT', ct, required_ct), ('pathology', he, required_he)]:
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f'{name} CSV missing columns: {sorted(missing)}')
        df['MRN'] = df['MRN'].astype('string').str.strip().replace('', pd.NA)
        df.dropna(subset=['MRN'], inplace=True)
        if df['MRN'].duplicated().any():
            raise ValueError(f'{name} CSV has duplicate MRN identifiers')

    data = he.merge(ct[['MRN', 'CT_OS_risk']], on='MRN', how='inner', validate='one_to_one')
    data = clean_data(data)
    data = data[data['Cohort'].astype(str).str.strip().eq(args.discovery_label)].copy()
    data = data.dropna(subset=FEATURES + ['OS', 'OS_Status'])
    data = data[(data['OS'] > 0) & data['OS_Status'].isin([0, 1])]
    if len(data) < 2 or not data['OS_Status'].astype(bool).any():
        raise ValueError('Insufficient training observations or no observed events')

    model = build_model()
    target = Surv.from_arrays(data['OS_Status'].astype(bool).to_numpy(),
                              data['OS'].astype(float).to_numpy())
    model.fit(data[FEATURES], target)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'composite_survival_model.joblib'
    joblib.dump(model, path, compress=3)
    (out / 'model_metadata.json').write_text(json.dumps({
        'endpoint': 'OS', 'time_column': 'OS', 'event_column': 'OS_Status',
        'features': FEATURES, 'training_rows': len(data),
        'training_events': int(data['OS_Status'].sum()),
        'rsf_parameters': {k: v for k, v in PARAMS.items() if k != 'n_jobs'}
    }, indent=2))
    print(f'Saved model: {path} | training patients: {len(data)}')


if __name__ == '__main__':
    main()
