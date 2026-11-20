"""Shared data loading and deterministic evaluation helpers."""
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index

ARCHS = ('R18', 'R34', 'R50', 'RX50')

def clean_ids(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

def load_table(path, endpoint, group=None):
    q = pd.read_csv(path)
    event = f'{endpoint}_Status'
    required = {'case_ID', endpoint, event}
    if group is not None:
        required.add('Group')
    if not required.issubset(q.columns):
        raise ValueError(f'{path}: missing columns {required - set(q.columns)}')
    if group is not None:
        q = q[q.Group.astype(str).str.lower().str.strip() == group.lower()]
    q = q[['case_ID', endpoint, event]].copy()
    q['case_ID'] = clean_ids(q.case_ID)
    q[endpoint] = pd.to_numeric(q[endpoint], errors='coerce')
    q[event] = pd.to_numeric(q[event], errors='coerce')
    q = q.dropna().query(f'`{endpoint}` > 0')
    if not q[event].isin([0, 1]).all():
        raise ValueError(f'{path}: {event} must be 0/1, where 1 is event')
    if q.case_ID.duplicated().any():
        raise ValueError(f'{path}: duplicate case_ID values')
    return q.reset_index(drop=True)

def attach_cache(q, cache_dir, prefix):
    q = q.copy()
    q['cache'] = [str(Path(cache_dir) / f'{prefix}_{v}.pt') for v in q.case_ID]
    missing = q[~q.cache.map(lambda p: Path(p).is_file())]
    if len(missing):
        print(f'WARNING: {len(missing)} patients without cache in {prefix}; excluded')
    return q[q.cache.map(lambda p: Path(p).is_file())].reset_index(drop=True)

def load_patches(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    patches = data['patches'] if isinstance(data, dict) else data
    if not isinstance(patches, torch.Tensor) or patches.ndim != 4 or patches.shape[1:] != (7, 64, 64):
        raise ValueError(f'{path}: expected [N,7,64,64] tensor')
    if not len(patches) or torch.all(patches == 0):
        raise ValueError(f'{path}: no nonzero patches')
    return patches.float()

def sample_patches(x, max_patches, augment=False):
    if len(x) > max_patches:
        dominant = x.mean((2, 3)).argmax(1)
        habitats = torch.unique(dominant)
        k = max(1, max_patches // len(habitats))
        chosen = []
        for h in habitats:
            z = torch.where(dominant == h)[0]
            if augment:
                z = z[torch.randperm(len(z))]
            chosen.extend(z[:k].tolist())
        chosen = list(dict.fromkeys(chosen))
        remaining = sorted(set(range(len(x))) - set(chosen))
        if augment:
            random.shuffle(remaining)
        x = x[chosen + remaining[:max(0, max_patches - len(chosen))]]
    if augment:
        if random.random() < .5:
            x = torch.flip(x, [3])
        if random.random() < .5:
            x = torch.flip(x, [2])
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, [2, 3])
    return x.float()

class PatientDataset(Dataset):
    def __init__(self, table, endpoint, max_patches=160, augment=False):
        self.q = table.reset_index(drop=True)
        self.endpoint = endpoint
        self.max_patches = max_patches
        self.augment = augment
    def __len__(self):
        return len(self.q)
    def __getitem__(self, i):
        row = self.q.iloc[i]
        x = sample_patches(load_patches(row.cache), self.max_patches, self.augment)
        return x, np.float32(row[self.endpoint]), np.float32(row[f'{self.endpoint}_Status']), str(row.case_ID)

def patient_loader(table, endpoint, max_patches=160, augment=False, shuffle=False, workers=0):
    return DataLoader(PatientDataset(table, endpoint, max_patches, augment),
                      batch_size=1, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)

def cindex(table, risk, endpoint):
    if not len(table):
        return float('nan')
    try:
        return float(concordance_index(table[endpoint], -np.asarray(risk), table[f'{endpoint}_Status']))
    except ZeroDivisionError:
        return float('nan')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
