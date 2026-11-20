"""Predict continuous patient-level Path-IO risk from four-CNN patch-risk NPZ files."""
import argparse
from pathlib import Path
import pandas as pd
from fisher_rsf import load_pipeline, read_risks, predict_patient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline', required=True, help='Trusted fitted .joblib artifact')
    p.add_argument('--risk-dir', required=True, help='Directory of four-CNN NPZ files')
    p.add_argument('--output', required=True, help='Output CSV')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f'{out} exists; use --overwrite to replace')
    pipe = load_pipeline(args.pipeline)
    files = sorted(Path(args.risk_dir).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No NPZ files in {args.risk_dir}')
    rows = []
    for path in files:
        x = read_risks(path, list(pipe['columns']))
        risk, risk_z = predict_patient(x, pipe)
        rows.append({'patient_file':path.stem, 'n_patches':len(x),
                     'risk':risk, 'risk_z':risk_z})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Saved {len(rows)} patient predictions to {out}')

if __name__ == '__main__':
    main()
