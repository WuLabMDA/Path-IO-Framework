"""Collect GDC-downloaded .svs slides into a flat folder without modifying originals."""
import argparse
import csv
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gdc-root', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--project', choices=['TCGA-LUAD', 'TCGA-LUSC'], required=True)
    ap.add_argument('--mode', choices=['copy', 'symlink'], default='copy')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    slides = sorted(p for p in args.gdc_root.rglob('*') if p.is_file() and p.suffix.lower() == '.svs')
    if not slides:
        raise SystemExit('No .svs files found; check download format and directory.')
    rows = []
    for src in slides:
        name = src.name
        patient = name[:12].upper()
        if not patient.startswith('TCGA-') or len(patient) != 12:
            print(f'WARNING: Nonstandard TCGA barcode: {name}')
        dst = args.output / name
        if dst.exists():
            if dst.resolve() != src.resolve() and dst.stat().st_size != src.stat().st_size:
                raise FileExistsError(f'Conflicting slide filename: {dst}')
            status = 'exists'
        else:
            if args.mode == 'symlink':
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            status = 'created'
        rows.append({'project': args.project, 'patient_id': patient,
                     'slide': name, 'source': str(src), 'destination': str(dst), 'status': status})
    with (args.output / 'slide_inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'{len(rows)} .svs files indexed in {args.output}')

if __name__ == '__main__':
    main()
