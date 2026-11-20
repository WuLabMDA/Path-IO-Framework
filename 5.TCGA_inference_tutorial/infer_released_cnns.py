"""Infer four-CNN patch risks using already-released checkpoints, no training CSV needed.

Import model code from Module 3. The checkpoint manifest specifies exact CNN
column names; those must match the fitted Fisher-RSF pipeline.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ARCHS = ('R18', 'R34', 'R50', 'RX50')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--module3', required=True, type=Path, help='Directory containing models.py')
    ap.add_argument('--cache', required=True, type=Path, help='Patient patch .pt folder')
    ap.add_argument('--checkpoints', required=True, type=Path, help='CSV: model,checkpoint,column')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('--batch-size must be positive')
    sys.path.insert(0, str(args.module3.resolve()))
    from models import Net

    df = pd.read_csv(args.checkpoints, dtype=str).fillna('')
    if not {'model', 'checkpoint', 'column'}.issubset(df.columns):
        ap.error('Manifest must contain model,checkpoint,column columns')
    if df.model.tolist() != list(ARCHS):
        ap.error('Manifest rows must be ordered R18,R34,R50,RX50')
    if df.column.duplicated().any() or not all(df.column.str.len()):
        ap.error('Manifest column identifiers must be nonempty and unique')
    files = sorted(args.cache.glob('*.pt'))
    if not files:
        ap.error('No patient .pt caches found')
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but not available')

    # Load one CNN at a time to limit GPU memory use; write temporary per-model arrays.
    temp = args.output / '_temporary_cnn_risks'
    temp.mkdir(exist_ok=True)
    for j, row in df.iterrows():
        checkpoint = Path(row.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = args.checkpoints.parent / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = Net(row.model).to(device).eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if isinstance(state, dict):
            for k in ('model_state_dict', 'state_dict', 'model'):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f'CNN {j+1}/4: {row.model} ({row.column})')
        with torch.inference_mode():
            for path in files:
                target = temp / f'{path.stem}__{j}.npy'
                if target.exists() and not args.overwrite:
                    continue
                data = torch.load(path, map_location='cpu', weights_only=True)
                x = data['patches'] if isinstance(data, dict) else data
                x = torch.as_tensor(x, dtype=torch.float32)
                if x.ndim != 4 or x.shape[1:] != (7,64,64) or len(x) == 0:
                    raise ValueError(f'{path}: expected [N,7,64,64]')
                result = []
                for start in range(0, len(x), args.batch_size):
                    y = model.patch_risk(x[start:start+args.batch_size].to(device))
                    result.append(y.float().cpu().numpy())
                risk = np.concatenate(result).astype(np.float32)
                if not np.isfinite(risk).all():
                    raise ValueError(f'{path}: nonfinite risk')
                np.save(target, risk)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    for path in files:
        dst = args.output / f'{path.stem}.npz'
        if dst.exists() and not args.overwrite:
            continue
        data = torch.load(path, map_location='cpu', weights_only=True)
        x = data['patches'] if isinstance(data, dict) else data
        coords = data.get('coords') if isinstance(data, dict) else None
        if coords is None:
            coords = np.arange(len(x), dtype=np.int32)[:, None]
        coords = np.asarray(coords)
        arrays = [np.load(temp / f'{path.stem}__{j}.npy') for j in range(4)]
        if any(len(a) != len(x) for a in arrays):
            raise ValueError(f'{path}: patch count mismatch')
        np.savez_compressed(dst, risk=np.stack(arrays, axis=1), coords=coords,
                            columns=df.column.to_numpy(dtype=str))
    print(f'Wrote {len(files)} patient risk NPZ files to {args.output}')
    print(f'Temporary arrays remain at {temp}; remove after checking outputs.')

if __name__ == '__main__':
    main()
