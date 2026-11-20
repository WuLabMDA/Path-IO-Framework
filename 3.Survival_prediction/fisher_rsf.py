"""Shared Fisher encoding and locked pipeline inference for four CNN risk columns."""
from pathlib import Path
import joblib
import numpy as np

MODELS = ('R18', 'R34', 'R50', 'RX50')


def fisher_vector(x, gmm):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != gmm.means_.shape[1]:
        raise ValueError('Expected nonempty [patches, CNN models] array')
    q = gmm.predict_proba(x)
    n = len(x)
    d = (x[:, None, :] - gmm.means_[None]) / np.sqrt(gmm.covariances_[None] + 1e-10)
    a = (q[:, :, None] * d).sum(0) / (n * np.sqrt(gmm.weights_[:, None] + 1e-10))
    b = (q[:, :, None] * (d*d - 1)).sum(0) / (n * np.sqrt(2*gmm.weights_[:, None] + 1e-10))
    f = np.c_[a, b].ravel()
    f = np.sign(f) * np.sqrt(np.abs(f))
    return (f / (np.linalg.norm(f) + 1e-10)).astype(np.float32)


def read_risks(path, columns):
    with np.load(path, allow_pickle=False) as z:
        risk = np.asarray(z['risk'], dtype=np.float32)
        names = [str(v) for v in z['columns']]
    if risk.ndim != 2 or risk.shape[1] != len(names) or len(risk) == 0:
        raise ValueError(f'{path}: invalid patch-risk shape')
    if len(set(names)) != len(names):
        raise ValueError(f'{path}: duplicate risk column names')
    if any(c not in names for c in columns):
        raise ValueError(f'{path}: missing required columns {set(columns) - set(names)}')
    x = risk[:, [names.index(c) for c in columns]]
    if not np.isfinite(x).all():
        raise ValueError(f'{path}: nonfinite patch risks')
    return x


def transform_patient(x, pipe):
    x = np.asarray(x, dtype=np.float32)
    x = ((x - pipe['patch_mu']) / pipe['patch_sd']).astype(np.float32)
    fv = fisher_vector(x, pipe['gmm'])[None, :]
    return pipe['scaler'].transform(fv).astype(np.float32)


def predict_patient(x, pipe):
    features = transform_patient(x, pipe)
    risk = float(pipe['rsf'].predict(features)[0])
    z = (risk - float(pipe['risk_mean'])) / (float(pipe['risk_sd']) + 1e-12)
    return risk, z


def load_pipeline(path):
    # joblib files are pickle-based: only load artifacts from trusted sources.
    pipe = joblib.load(Path(path))
    required = {'patch_mu','patch_sd','gmm','scaler','rsf','risk_mean','risk_sd','models','columns','K','feature_dim'}
    if not required.issubset(pipe):
        raise ValueError(f'Missing pipeline keys: {required - set(pipe)}')
    if len(pipe['columns']) != 4 or list(pipe['models']) != list(MODELS):
        raise ValueError('Pipeline must use R18, R34, R50, RX50 in that order')
    return pipe
