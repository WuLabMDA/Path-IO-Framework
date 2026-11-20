"""Fit four-CNN Fisher-RSF on training patients only; evaluate other cohorts without refitting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from fisher_rsf import MODELS, fisher_vector, read_risks, predict_patient


def clean_ids(s):
    return s.astype('string').str.replace(r'\.0$', '', regex=True).str.strip()


def cohort_data(csv, cohort, split_col, endpoint, risk_dir):
    d = pd.read_csv(csv)
    time_col, event_col = endpoint, f'{endpoint}_Status'
    required = {'case_ID', time_col, event_col}
    if cohort in ('train','valid','development','internal'):
        required.add(split_col)
    if not required.issubset(d.columns):
        raise ValueError(f'{csv}: missing columns {required - set(d.columns)}')
    d['case_ID'] = clean_ids(d['case_ID'])
    if cohort in ('train','valid','development','internal'):
        group = d[split_col].astype(str).str.strip().str.lower()
        allowed = {'train': ['train'], 'valid': ['valid'],
                   'development': ['train', 'valid'], 'internal': ['test']}[cohort]
        d = d.loc[group.isin(allowed)]
    d = d[['case_ID', time_col, event_col]].dropna().copy()
    d[time_col] = pd.to_numeric(d[time_col], errors='coerce')
    d[event_col] = pd.to_numeric(d[event_col], errors='coerce')
    d = d.loc[(d[time_col] > 0) & d[event_col].isin([0, 1])].copy()
    if d['case_ID'].duplicated().any():
        raise ValueError(f'{csv}: duplicate case_ID in {cohort}; resolve before training')
    source = 'mda' if cohort in ('train','valid','development','internal') else cohort
    d['risk_file'] = d['case_ID'].map(lambda c: risk_dir / f'{source}_{c}.npz')
    missing = (~d['risk_file'].map(Path.is_file)).sum()
    if missing:
        print(f'{cohort}: {missing} patients without patch risks excluded')
    return d.loc[d['risk_file'].map(Path.is_file)].reset_index(drop=True)


def report(cohort, d, risk, mu, sd, endpoint, output):
    t, e = endpoint, f'{endpoint}_Status'
    out = d[['case_ID', t, e]].copy()
    out['risk'] = risk
    out['risk_z'] = (risk - mu) / (sd + 1e-12)
    out.to_csv(output / f'{endpoint}_{cohort}_continuous_RSF_risk.csv', index=False)
    row = {'cohort': cohort, 'N': len(out), 'events': int(out[e].sum()),
           'C_index': np.nan, 'HR_per_train_SD': np.nan,
           'CI_low': np.nan, 'CI_high': np.nan, 'Cox_P': np.nan}
    if len(out) >= 2 and out[e].sum() > 0:
        try:
            row['C_index'] = float(concordance_index(out[t], -risk, out[e]))
        except (ValueError, ZeroDivisionError):
            pass
    if out[e].sum() > 0 and out['risk_z'].std() > 0:
        try:
            fit = CoxPHFitter().fit(out[[t, e, 'risk_z']], duration_col=t, event_col=e)
            s = fit.summary.loc['risk_z']
            row.update(HR_per_train_SD=float(np.exp(s['coef'])),
                       CI_low=float(np.exp(s['coef lower 95%'])),
                       CI_high=float(np.exp(s['coef upper 95%'])), Cox_P=float(s['p']))
        except (ValueError, ArithmeticError) as exc:
            print(f'{cohort}: Cox fit unavailable: {exc}')
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--risk-dir', required=True, help='Directory of OS_4D or PFS_4D NPZ files')
    p.add_argument('--mda-csv', required=True, help='CSV with case_ID, Group, endpoint, endpoint_Status')
    p.add_argument('--mayo-csv'); p.add_argument('--cimac-csv'); p.add_argument('--roussy-csv')
    p.add_argument('--output', required=True)
    p.add_argument('--endpoint', choices=['OS','PFS'], default='OS')
    p.add_argument('--split-col', default='Group')
    p.add_argument('--components', type=int, default=25)
    args = p.parse_args()
    risk_dir = Path(args.risk_dir)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    cohorts = {'train': args.mda_csv, 'valid': args.mda_csv,
               'development': args.mda_csv, 'internal': args.mda_csv,
               'mayo': args.mayo_csv, 'cimac': args.cimac_csv, 'roussy': args.roussy_csv}
    cohorts = {k:v for k,v in cohorts.items() if v}
    data = {c:cohort_data(path,c,args.split_col,args.endpoint,risk_dir) for c,path in cohorts.items()}
    if data['train'].empty:
        raise ValueError('No training patients with matching patch-risk files')
    if not data['train'][f'{args.endpoint}_Status'].astype(bool).any():
        raise ValueError('Training data must contain observed events')

    # Read column identities from training data; never assume array position.
    first = data['train'].iloc[0]['risk_file']
    with np.load(first, allow_pickle=False) as z:
        names = [str(c) for c in z['columns']]
    columns = []
    for arch in MODELS:
        matches = [c for c in names if c.startswith(f'{args.endpoint}_{arch}_S')]
        if len(matches) != 1:
            raise ValueError(f'{first}: expected one {arch} column, found {matches}')
        columns.append(matches[0])
    raw = {c:[read_risks(f,columns) for f in d['risk_file']] for c,d in data.items()}
    train_patches = np.vstack(raw['train'])
    mu = train_patches.mean(axis=0)
    sd = train_patches.std(axis=0)
    sd[sd < 1e-8] = 1
    normalized = {c:[((x-mu)/sd).astype(np.float32) for x in xs] for c,xs in raw.items()}
    gmm = GaussianMixture(n_components=args.components, covariance_type='diag',
                          reg_covar=1e-5, n_init=3, max_iter=500, random_state=42)
    gmm.fit(np.vstack(normalized['train']))
    fisher = {c:np.vstack([fisher_vector(x,gmm) for x in xs]) for c,xs in normalized.items() if xs}
    scaler = StandardScaler().fit(fisher['train'])
    features = {c:scaler.transform(x).astype(np.float32) for c,x in fisher.items()}
    event = f'{args.endpoint}_Status'
    y = Surv.from_arrays(data['train'][event].astype(bool).to_numpy(),
                         data['train'][args.endpoint].astype(float).to_numpy())
    rsf = RandomSurvivalForest(n_estimators=300, max_depth=2, min_samples_split=20,
                               min_samples_leaf=10, max_features='log2', bootstrap=False,
                               max_samples=None, low_memory=True, n_jobs=-1,
                               random_state=42).fit(features['train'], y)
    train_risk = rsf.predict(features['train'])
    rmu = float(train_risk.mean()); rsd = float(train_risk.std(ddof=0))
    pipe = {'patch_mu':mu, 'patch_sd':sd, 'gmm':gmm, 'scaler':scaler,
            'rsf':rsf, 'risk_mean':rmu, 'risk_sd':rsd,
            'models':list(MODELS), 'columns':columns, 'K':args.components,
            'feature_dim':features['train'].shape[1]}
    path = output / f'{args.endpoint}_locked_Fisher_RSF_pipeline.joblib'
    joblib.dump(pipe, path, compress=3)
    print('Saved:', path)
    rows = []
    for cohort, x in features.items():
        risk = rsf.predict(x)
        rows.append(report(cohort, data[cohort], risk, rmu, rsd, args.endpoint, output))
    results = pd.DataFrame(rows)
    results.to_csv(output / f'{args.endpoint}_continuous_RSF_Cox_results.csv', index=False)
    print(results.to_string(index=False))

if __name__ == '__main__':
    main()
