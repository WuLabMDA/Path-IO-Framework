#!/usr/bin/env python3
"""Extract seven-channel one-hot patches from patient-level RGB habitat maps."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

RGB = np.array([
    [54,54,54], [251,217,204], [41,150,191], [1,0,128],
    [142,142,142], [71,169,45], [229,76,37], [187,47,154]
], dtype=np.uint8)
BGR = RGB[:, ::-1]


def load_map(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f'Cannot read {path}')
    m = np.full(im.shape[:2], 255, np.uint8)
    for k, color in enumerate(BGR):
        m[np.all(im == color, axis=2)] = k
    if (m == 255).any():
        raise RuntimeError(f'{path}: unknown colors')
    return m


def extract(m, patch_size=64, stride=32, min_tissue=.10):
    h, w = m.shape
    valid = np.ones((h, w), bool)
    ph, pw = max(0, patch_size-h), max(0, patch_size-w)
    if ph or pw:
        m = np.pad(m, ((0, ph), (0, pw)), constant_values=0)
        valid = np.pad(valid, ((0, ph), (0, pw)), constant_values=False)
    H, W = m.shape
    patches, coords = [], []
    best = None
    for y in range(0, H-patch_size+1, stride):
        for x in range(0, W-patch_size+1, stride):
            q = m[y:y+patch_size, x:x+patch_size]
            v = valid[y:y+patch_size, x:x+patch_size]
            denom = int(v.sum())
            tissue = float(((q > 0) & v).sum()/denom) if denom else 0.0
            if tissue >= min_tissue or best is None or tissue > best[0]:
                z = np.stack([(q == k).astype(np.float32) for k in range(1, 8)])
                if best is None or tissue > best[0]:
                    best = (tissue, z, x, y)
                if tissue >= min_tissue:
                    patches.append(z)
                    coords.append([x, y])
    if not patches:
        if best is None or best[0] == 0:
            raise RuntimeError('No non-background habitat')
        _, z, x, y = best
        patches, coords = [z], [[x, y]]
    P = torch.from_numpy(np.stack(patches)).float()
    C = torch.tensor(coords, dtype=torch.int32)
    if torch.all(P == 0):
        raise RuntimeError('All-zero patient')
    return P, C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', action='append', required=True, metavar='COHORT=FOLDER',
                    help='Repeat for multiple cohorts; folders contain patient PNG maps')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--patch-size', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--min-tissue', type=float, default=.10)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.patch_size <= 0 or args.stride <= 0 or not 0 <= args.min_tissue <= 1:
        ap.error('Invalid patch size, stride, or minimum tissue fraction')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.input:
        if '=' not in spec:
            ap.error('--input must have format COHORT=FOLDER')
        cohort, folder = spec.split('=', 1)
        folder = Path(folder)
        if not cohort or not folder.is_dir():
            ap.error(f'Invalid cohort or missing folder: {spec}')
        files = sorted(folder.glob('*.png'))
        total = one = errors = 0
        print(f'{cohort}: {len(files)} patient maps')
        for f in tqdm(files, desc=cohort):
            out = args.output/f'{cohort.lower()}_{f.stem}.pt'
            if out.exists() and not args.overwrite:
                rows.append(dict(cohort=cohort, patient_id=f.stem, status='EXISTS',
                                 n_patches=np.nan, output=str(out)))
                continue
            try:
                P, C = extract(load_map(f), args.patch_size, args.stride, args.min_tissue)
                torch.save({'patches': P, 'coords': C}, out)
                total += len(P)
                one += len(P) == 1
                rows.append(dict(cohort=cohort, patient_id=f.stem, status='DONE',
                                 n_patches=len(P), output=str(out)))
            except Exception as exc:
                errors += 1
                print(f'ERROR {cohort} {f.stem}: {exc}')
                rows.append(dict(cohort=cohort, patient_id=f.stem,
                                 status=f'ERROR: {exc}', n_patches=np.nan, output=''))
        print(f'{cohort}: patches={total:,} | one_patch={one} | errors={errors}')
    pd.DataFrame(rows, columns=['cohort','patient_id','status','n_patches','output']).to_csv(
        args.output/'patch_generation_summary.csv', index=False)
    n = zp = zpat = 0
    for f in args.output.glob('*.pt'):
        d = torch.load(f, map_location='cpu', weights_only=True)
        P = d['patches']
        zero = (P.reshape(len(P), -1) == 0).all(1)
        n += len(P)
        zp += int(zero.sum())
        zpat += int(zero.all())
    print(f'CACHE QC | total patches: {n} | zero patches: {zp} | zero patients: {zpat}')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
