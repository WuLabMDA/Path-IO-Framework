#!/usr/bin/env python3
"""UNI embedding extraction, optional linear-head training, and tissue inference.

Input for extract/train/evaluate: class-labelled directories (one subdirectory per
class). Input for predict: arbitrary image directory, optionally recursively.
The original patient-disjoint classifier can be loaded for evaluate/predict.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

CLASSES = ["Background", "Bronchi", "Inflammatory", "Lung", "Necrosis", "Stroma", "Tumor", "Vessel"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def collect_images(root, labelled):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    rows = []
    if labelled:
        for label, name in enumerate(CLASSES):
            folder = root / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected class directory: {folder}")
            rows.extend((str(p), label) for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        rows = [(str(p), -1) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if not rows:
        raise ValueError(f"No supported images found in {root}")
    return pd.DataFrame(rows, columns=['file', 'label'])


class PatchDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        record = self.manifest.iloc[index]
        try:
            with Image.open(record['file']) as img:
                tensor = TRANSFORM(img.convert('RGB'))
            return tensor, int(record['label']), str(record['file']), True
        except (OSError, ValueError) as exc:
            print(f"WARNING: cannot read {record['file']}: {exc}")
            return torch.zeros(3, 224, 224), int(record['label']), str(record['file']), False


def load_uni(checkpoint, device):
    model = timm.create_model('vit_large_patch16_224', img_size=224,
                              patch_size=16, init_values=1e-5,
                              num_classes=0, dynamic_img_size=True)
    try:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except TypeError as exc:
        raise RuntimeError('A recent PyTorch version supporting weights_only=True is required') from exc
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_embeddings(model, manifest, device, batch_size, workers):
    loader = DataLoader(PatchDataset(manifest), batch_size=batch_size,
                        shuffle=False, num_workers=workers,
                        pin_memory=device.type == 'cuda')
    features, records = [], []
    with torch.inference_mode():
        for batch, labels, paths, valid in loader:
            valid = valid.bool()
            if not valid.any():
                continue
            output = model(batch[valid].to(device, non_blocking=True))
            features.append(output.detach().cpu().numpy().astype(np.float32))
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
            records.extend((paths[i], int(labels[i])) for i in valid_indices)
    if not features:
        raise ValueError('All images failed to load')
    return np.concatenate(features), pd.DataFrame(records, columns=['file', 'label'])


def save_features(out_dir, name, features, manifest):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{name}_features.npy', features)
    manifest.to_csv(out_dir / f'{name}_manifest.csv', index=False)


def load_features(out_dir, name):
    features = np.load(out_dir / f'{name}_features.npy', allow_pickle=False)
    manifest = pd.read_csv(out_dir / f'{name}_manifest.csv')
    if len(features) != len(manifest):
        raise ValueError('Feature and manifest row counts differ')
    return features, manifest


def check_classifier(clf, features):
    if not hasattr(clf, 'predict'):
        raise TypeError('Classifier does not support predict()')
    if hasattr(clf, 'n_features_in_') and features.shape[1] != clf.n_features_in_:
        raise ValueError(f'Expected {clf.n_features_in_} UNI features, got {features.shape[1]}')


def write_predictions(clf, features, manifest, output):
    check_classifier(clf, features)
    labels = clf.predict(features).astype(int)
    if not np.isin(labels, np.arange(len(CLASSES))).all():
        raise ValueError('Classifier returned labels outside 0..7')
    result = manifest.copy()
    result['pred_label'] = labels
    result['pred_class'] = [CLASSES[i] for i in labels]
    if (result['label'] >= 0).all():
        result['true_class'] = [CLASSES[int(i)] for i in result['label']]
    result.to_csv(output, index=False)
    return result


def evaluate(predictions, out_dir):
    y = predictions['label'].to_numpy(dtype=int)
    pred = predictions['pred_label'].to_numpy(dtype=int)
    if not np.isin(y, np.arange(8)).all():
        raise ValueError('Evaluation requires labels 0..7')
    metrics = {'accuracy': accuracy_score(y, pred),
               'balanced_accuracy': balanced_accuracy_score(y, pred),
               'macro_f1': f1_score(y, pred, labels=list(range(8)), average='macro', zero_division=0)}
    pd.DataFrame([metrics]).to_csv(out_dir / 'test_metrics.csv', index=False)
    report = classification_report(y, pred, labels=list(range(8)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / 'test_classification_report.csv')
    cm = confusion_matrix(y, pred, labels=list(range(8)))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / 'test_confusion_matrix.csv')
    print(metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['extract', 'train', 'evaluate', 'predict'], required=True)
    p.add_argument('--input', help='Patch directory; class subfolders for extract/evaluate')
    p.add_argument('--split-name', default='train', help='Feature cache prefix for extract/train')
    p.add_argument('--features-dir', default='features')
    p.add_argument('--output-dir', default='outputs')
    p.add_argument('--uni-checkpoint', help='Locally downloaded authorized UNI pytorch_model.bin')
    p.add_argument('--classifier', help='Saved fitted UNI 8-class classifier joblib')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(args.features_dir)
    if args.mode == 'train':
        X, manifest = load_features(feature_dir, args.split_name)
        y = manifest['label'].to_numpy(dtype=int)
        if not np.isin(np.arange(8), np.unique(y)).all():
            raise ValueError('Training data must contain all eight classes')
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=3000, class_weight='balanced', n_jobs=-1))
        clf.fit(X, y)
        target = out / 'UNI_8class_classifier.joblib'
        joblib.dump(clf, target)
        print(f'Saved classifier: {target}')
        return
    if not args.input or not args.uni_checkpoint:
        p.error('--input and --uni-checkpoint are required for extract/evaluate/predict')
    labelled = args.mode in {'extract', 'evaluate'}
    manifest = collect_images(args.input, labelled=labelled)
    device = torch.device(args.device)
    model = load_uni(args.uni_checkpoint, device)
    X, manifest = extract_embeddings(model, manifest, device, args.batch_size, args.workers)
    if args.mode == 'extract':
        save_features(feature_dir, args.split_name, X, manifest)
        print(f'Saved {len(manifest)} embeddings to {feature_dir}')
        return
    if not args.classifier:
        p.error('--classifier is required for evaluate/predict')
    # Only load joblib files from trusted sources; loading can execute code.
    clf = joblib.load(args.classifier)
    predictions = write_predictions(clf, X, manifest, out / 'tissue_predictions.csv')
    if args.mode == 'evaluate':
        evaluate(predictions, out)
    print(f'Saved {len(predictions)} predictions to {out / "tissue_predictions.csv"}')


if __name__ == '__main__':
    main()
