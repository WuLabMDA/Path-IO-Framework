"""Select each model's best VALIDATION checkpoint, infer every patch, save [N,4] risk.
No test/external cohort metrics are used in checkpoint selection.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from models import Net
from common import ARCHS, load_patches

def select_checkpoints(root, endpoint, models, seeds, min_epoch):
    selected = []
    for arch in models:
        candidates = []
        for seed in seeds:
            folder = Path(root) / f'{arch}_{endpoint}_new_clinical_split_S{seed}'
            csv = folder / 'all_results.csv'
            if not csv.exists():
                raise FileNotFoundError(csv)
            df = pd.read_csv(csv)
            if not {'epoch', 'valid_C'}.issubset(df.columns):
                raise ValueError(f'{csv}: expected epoch and valid_C')
            for _, row in df.iterrows():
                ep = int(row.epoch)
                if ep < min_epoch or not np.isfinite(row.valid_C):
                    continue
                ck = folder / f'S{seed}_E{ep:03d}.pt'
                if ck.exists():
                    candidates.append((float(row.valid_C), ep, seed, ck))
        if not candidates:
            raise RuntimeError(f'No valid checkpoints for {arch} {endpoint}')
        # Highest validation C-index, earliest epoch, lowest seed for ties.
        score, epoch, seed, ck = sorted(candidates, key=lambda v: (-v[0], v[1], v[2]))[0]
        selected.append({'endpoint': endpoint, 'model': arch, 'seed': seed,
                         'epoch': epoch, 'valid_C': score, 'checkpoint': str(ck)})
    return pd.DataFrame(selected)

def load_model(row, device, dropout):
    model = Net(row['model'], dropout=dropout).to(device)
    state = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def infer(model, patches, batch_size, device):
    results = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            result = model.patch_risk(patches[start:start+batch_size].to(device))
            results.append(result.float().cpu().numpy())
    return np.concatenate(results).astype(np.float32)

def read_cache(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    patches = load_patches(path)
    coords = data.get('coords') if isinstance(data, dict) else None
    if coords is None:
        coords = torch.arange(len(patches), dtype=torch.int32)[:, None]
    coords = torch.as_tensor(coords).cpu().numpy()
    if len(coords) != len(patches):
        raise ValueError(f'{path}: patch/coordinate count mismatch')
    return patches, coords

def run_endpoint(args, endpoint, files, device):
    out = Path(args.output) / f'{endpoint}_4D'
    out.mkdir(parents=True, exist_ok=True)
    selected = select_checkpoints(args.training_root, endpoint, args.models,
                                  args.seeds, args.min_epoch)
    selected.to_csv(Path(args.output) / f'{endpoint}_selected_checkpoints.csv', index=False)
    columns = np.array([f'{endpoint}_{r.model}_S{r.seed}' for r in selected.itertuples()])
    print(selected[['model', 'seed', 'epoch', 'valid_C']].to_string(index=False))
    # Process one patient at a time to avoid temporary risk files and reloading caches.
    models = [load_model(row, device, args.dropout) for _, row in selected.iterrows()]
    try:
        for path in tqdm(files, desc=f'{endpoint} patch risks'):
            destination = out / f'{path.stem}.npz'
            if destination.exists() and not args.overwrite:
                continue
            patches, coords = read_cache(path)
            risks = np.stack([infer(model, patches, args.batch_size, device)
                              for model in models], axis=1)
            if risks.shape != (len(patches), len(args.models)) or not np.isfinite(risks).all():
                raise RuntimeError(f'{path}: invalid risk array')
            np.savez_compressed(destination, risk=risks, coords=coords, columns=columns)
    finally:
        del models
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f'Saved {endpoint} patch risks to {out}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True, help='Patient patch cache directory containing .pt files')
    p.add_argument('--training-root', required=True, help='Directory containing training run folders')
    p.add_argument('--output', required=True)
    p.add_argument('--endpoint', choices=['OS', 'PFS', 'both'], default='OS')
    p.add_argument('--models', nargs='+', choices=ARCHS, default=list(ARCHS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--min-epoch', type=int, default=3)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    files = sorted(Path(args.cache).glob('*.pt'))
    if not files:
        raise ValueError(f'No .pt files found in {args.cache}')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for endpoint in (['OS', 'PFS'] if args.endpoint == 'both' else [args.endpoint]):
        run_endpoint(args, endpoint, files, torch.device(args.device))

if __name__ == '__main__':
    main()
