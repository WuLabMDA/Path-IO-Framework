"""Apply the fitted composite model to CT, Path-IO, and clinical risk data."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train_composite_model import CATEGORICAL, NUMERIC, FEATURES, clean_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='CSV with CT, Path-IO and clinical features')
    p.add_argument('--model', default='weights/composite_survival_model.joblib')
    p.add_argument('--output', default='composite_risk_predictions.csv')
    args = p.parse_args()
    df = pd.read_csv(args.input, dtype={'MRN': str})
    df.columns = df.columns.str.strip()
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    df = clean_data(df)
    valid = df[['CT_OS_risk', 'HE_OS_risk']].notna().all(axis=1)
    result = pd.DataFrame({'composite_risk': np.nan}, index=df.index)
    if 'MRN' in df:
        result.insert(0, 'MRN', df['MRN'])
    if valid.any():
        model = joblib.load(args.model)  # Load only trusted model files.
        result.loc[valid, 'composite_risk'] = model.predict(df.loc[valid, FEATURES])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f'Saved {valid.sum()} predictions to {path}')


if __name__ == '__main__':
    main()
