"""Train four checkpoint-compatible Path-IO CNNs using patient-level Cox loss.
Validation-only checkpoint selection; external cohorts are never used for selection.
"""
import argparse
import contextlib
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models import Net
from common import (ARCHS, load_table, attach_cache, patient_loader,
                    cindex, set_seed)

def cox_loss(risk, time, event):
    # Breslow partial likelihood, matching original training loss.
    idx = torch.argsort(time, descending=True)
    risk, event = risk[idx], event[idx]
    return -(event * (risk - torch.logcumsumexp(risk, 0))).sum() / event.sum().clamp_min(1e-8)

@torch.no_grad()
def predict(model, table, endpoint, device, max_patches):
    model.eval()
    risks = []
    for x, _, _, _ in patient_loader(table, endpoint, max_patches):
        risk = model(x.to(device)).float()
        if not torch.isfinite(risk):
            raise ValueError('Nonfinite patient risk')
        risks.append(risk.item())
    return np.asarray(risks)

def train_one(args, architecture, seed, data, device):
    out = Path(args.output) / f'{architecture}_{args.endpoint}_new_clinical_split_S{seed}'
    out.mkdir(parents=True, exist_ok=True)
    results = out / 'all_results.csv'
    set_seed(seed)
    model = Net(architecture, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start = 1
    rows = []
    latest = out / 'latest_training_state.pt'
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start = int(state['epoch']) + 1
        if results.exists():
            rows = pd.read_csv(results).to_dict('records')
        print(f'Resuming {architecture} seed={seed} at epoch {start}')
    train_loader = patient_loader(data['train'], args.endpoint, args.max_patches,
                                  augment=True, shuffle=True, workers=args.workers)
    for epoch in range(start, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risks, times, events, losses = [], [], [], []
        for i, (x, t, e, _) in enumerate(train_loader, 1):
            # Keep computation graph for a batch of patients.
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                r = model(x.to(device))
            risks.append(r.reshape(1))
            times.append(t.to(device).reshape(1))
            events.append(e.to(device).reshape(1))
            if len(risks) == args.cox_batch or i == len(train_loader):
                rr = torch.cat(risks).float()
                tt = torch.cat(times).float()
                ee = torch.cat(events).float()
                if ee.sum().item() > 0:
                    loss = cox_loss(rr, tt, ee)
                    if torch.isfinite(loss):
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        scaler.step(optimizer)
                        scaler.update()
                        losses.append(float(loss.detach().cpu()))
                optimizer.zero_grad(set_to_none=True)
                risks, times, events = [], [], []
        metrics = {'seed': seed, 'epoch': epoch,
                   'loss': float(np.mean(losses)) if losses else float('nan')}
        for name, table in data.items():
            pred = predict(model, table, args.endpoint, device, args.max_patches)
            metrics[f'{name}_C'] = cindex(table, pred, args.endpoint)
            pd.DataFrame({'case_ID': table.case_ID, 'risk': pred}).to_csv(
                out / f'S{seed}_E{epoch:03d}_{name}.csv', index=False)
        ck = out / f'S{seed}_E{epoch:03d}.pt'
        torch.save(model.state_dict(), ck)
        rows = [r for r in rows if int(r['epoch']) != epoch]
        rows.append(metrics)
        pd.DataFrame(rows).sort_values('epoch').to_csv(results, index=False)
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()}, latest)
        print(f'{architecture} seed={seed} epoch={epoch} loss={metrics["loss"]:.4f} '
              f'train_C={metrics["train_C"]:.4f} valid_C={metrics["valid_C"]:.4f}')
    table = pd.read_csv(results)
    eligible = table[(table.epoch >= args.min_epoch) & np.isfinite(table.valid_C)]
    if eligible.empty:
        raise RuntimeError(f'No valid validation C-index for {architecture} seed {seed}')
    best = eligible.sort_values(['valid_C', 'epoch'], ascending=[False, True]).iloc[0]
    print(f'BEST validation {architecture} seed={seed}: epoch={int(best.epoch)}, C={best.valid_C:.4f}')
    del model, optimizer, scaler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clinical', required=True, help='CSV with case_ID, Group, OS/PFS, OS_Status/PFS_Status')
    p.add_argument('--cache', required=True, help='Folder with cohort_caseID.pt caches')
    p.add_argument('--output', required=True)
    p.add_argument('--endpoint', choices=['OS', 'PFS'], default='OS')
    p.add_argument('--prefix', default='mda', help='Cache filename prefix for primary cohort')
    p.add_argument('--external', action='append', default=[], metavar='NAME:CSV:PREFIX',
                   help='Optional evaluation cohort; never used for model selection')
    p.add_argument('--models', nargs='+', choices=ARCHS, default=list(ARCHS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--min-epoch', type=int, default=3)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--max-patches', type=int, default=160)
    p.add_argument('--cox-batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    device = torch.device(args.device)
    data = {group: attach_cache(load_table(args.clinical, args.endpoint, group),
                                args.cache, args.prefix) for group in ('train', 'valid', 'test')}
    if not len(data['train']) or not len(data['valid']):
        raise ValueError('Training and validation groups must be nonempty')
    data['development'] = pd.concat([data['train'], data['valid']], ignore_index=True)
    data['internal'] = data.pop('test')
    for entry in args.external:
        name, csv, prefix = entry.split(':', 2)
        if name in data:
            raise ValueError(f'Duplicate cohort {name}')
        data[name] = attach_cache(load_table(csv, args.endpoint), args.cache, prefix)
    print('Patients:', {name: len(table) for name, table in data.items()})
    for arch in args.models:
        for seed in args.seeds:
            train_one(args, arch, seed, data, device)

if __name__ == '__main__':
    main()
