#!/usr/bin/env python3
"""Pack slide-level RGB habitat maps into patient-level maps without resizing."""
import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

BG = (54, 54, 54)


def natural(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', p.stem)]


def mask(a):
    return np.any(a != np.asarray(BG, dtype=np.uint8), axis=2)


def crop(a):
    y, x = np.where(mask(a))
    return None if not len(x) else a[y.min():y.max()+1, x.min():x.max()+1]


def keep_idx(active, gap):
    active = np.asarray(active, bool)
    keep = np.zeros(len(active), bool)
    i = 0
    while i < len(active):
        if active[i]:
            keep[i] = True
            i += 1
            continue
        s = i
        while i < len(active) and not active[i]:
            i += 1
        e = i
        if s == 0 or e == len(active):
            continue
        n = min(gap, e-s)
        if n:
            q = s + (e-s-n)//2
            keep[q:q+n] = True
    return np.where(keep)[0]


def prepare(path, inner_gap, rotate):
    with Image.open(path) as im:
        a = np.asarray(im.convert('RGB')).copy()
    a = crop(a)
    if a is None:
        return None
    r = keep_idx(np.any(mask(a), axis=1), inner_gap)
    if len(r):
        a = a[r]
    c = keep_idx(np.any(mask(a), axis=0), inner_gap)
    if len(c):
        a = a[:, c]
    a = crop(a)
    if a is None:
        return None
    m = mask(a)
    z = np.full(a.shape, BG, dtype=np.uint8)
    z[m] = a[m]
    if rotate and z.shape[0] > z.shape[1]:
        z = np.rot90(z).copy()
    return z


def layout(blocks, width, gap):
    positions = []
    x = y = rowh = used = 0
    for a in blocks:
        h, w = a.shape[:2]
        if x and x+w > width:
            y += rowh+gap
            x = rowh = 0
        positions.append((x, y))
        x += w+gap
        rowh = max(rowh, h)
        used = max(used, x-gap)
    return positions, used, y+rowh


def compact(blocks, gap, target_ar):
    if len(blocks) == 1:
        h, w = blocks[0].shape[:2]
        return [(0, 0)], w, h
    ws = [a.shape[1] for a in blocks]
    area = sum(a.shape[0]*a.shape[1] for a in blocks)
    lo = max(ws)
    hi = sum(ws)+gap*(len(ws)-1)
    ideal = math.sqrt(area*target_ar)
    candidates = {lo, hi}
    candidates.update(int(np.clip(ideal*x, lo, hi)) for x in np.linspace(.6, 3, 40))
    run = 0
    for i, w in enumerate(ws):
        run += w+(gap if i else 0)
        candidates.add(run)
    best = None
    for width in candidates:
        pos, w, h = layout(blocks, width, gap)
        ar = w/max(h, 1)
        score = w*h*(1+.15*abs(math.log(max(ar, 1e-8)/target_ar)))
        if best is None or score < best[0]:
            best = (score, pos, w, h)
    return best[1:]


def pack(blocks, gap, margin, target_ar):
    positions, w, h = compact(blocks, gap, target_ar)
    out = np.full((h+2*margin, w+2*margin, 3), BG, dtype=np.uint8)
    for a, (x, y) in zip(blocks, positions):
        hh, ww = a.shape[:2]
        out[y+margin:y+margin+hh, x+margin:x+margin+ww] = a
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', action='append', required=True, metavar='COHORT=FOLDER',
                    help='Repeat for multiple cohorts; e.g. --input MDA=/maps/MDA')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inner-gap', type=int, default=2)
    ap.add_argument('--sample-gap', type=int, default=4)
    ap.add_argument('--margin', type=int, default=4)
    ap.add_argument('--target-ar', type=float, default=1.5)
    ap.add_argument('--no-rotate', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--patient-id-regex', default=r'^([^-]+)',
                    help='Regex with group 1 capturing patient ID from filename stem; default takes text before first hyphen')
    args = ap.parse_args()
    if min(args.inner_gap, args.sample_gap, args.margin) < 0 or args.target_ar <= 0:
        ap.error('Gaps and margin must be nonnegative, and target-ar must be positive')
    pattern = re.compile(args.patient_id_regex)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.input:
        if '=' not in spec:
            ap.error('--input must have format COHORT=FOLDER')
        cohort, folder = spec.split('=', 1)
        folder = Path(folder)
        if not cohort or not folder.is_dir():
            ap.error(f'Invalid cohort or missing folder: {spec}')
        outdir = args.output/cohort
        outdir.mkdir(parents=True, exist_ok=True)
        files = sorted(folder.glob('*.png'), key=natural)
        patients = {}
        for f in files:
            match = pattern.search(f.stem)
            if not match:
                warnings.warn(f'Skipping filename without patient ID: {f.name}')
                continue
            patients.setdefault(match.group(1), []).append(f)
        print(f'{cohort}: {len(files)} maps, {len(patients)} patients')
        for pid, fs in tqdm(patients.items(), desc=cohort):
            outfile = outdir/f'{pid}.png'
            rec = dict(cohort=cohort, patient_id=pid, n_maps=0, status='',
                       output=str(outfile), width=np.nan, height=np.nan, input_maps='')
            if outfile.exists() and not args.overwrite:
                rec['status'] = 'EXISTS'
                rec['n_maps'] = len(fs)
                rows.append(rec)
                continue
            try:
                blocks, used = [], []
                for f in sorted(fs, key=natural):
                    a = prepare(f, args.inner_gap, not args.no_rotate)
                    if a is not None:
                        blocks.append(a)
                        used.append(f.name)
                rec['n_maps'] = len(blocks)
                rec['input_maps'] = '; '.join(used)
                if not blocks:
                    rec['status'] = 'NO_TISSUE'
                else:
                    canvas = pack(blocks, args.sample_gap, args.margin, args.target_ar)
                    Image.fromarray(canvas).save(outfile, compress_level=3)
                    rec.update(status='DONE', width=canvas.shape[1], height=canvas.shape[0])
            except Exception as exc:
                rec['status'] = f'ERROR: {exc}'
                warnings.warn(f'{cohort} | {pid}: {exc}')
            rows.append(rec)
    summary = args.output/'patient_level_map_summary.csv'
    pd.DataFrame(rows, columns=['cohort','patient_id','n_maps','status','output',
                                'width','height','input_maps']).to_csv(summary, index=False)
    print(f'Saved summary: {summary}')


if __name__ == '__main__':
    main()
