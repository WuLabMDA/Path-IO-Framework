"""UNI eight-class tissue inference (without stain normalization).

Input: per-slide directories of image patches named using svs2tile.py convention.
Output: per-slide CSV predictions, RGB habitat maps, corrupt-file lists, and QC.
"""
import argparse
import os
import re
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

CLASSES = ["Background", "Bronchi", "Inflammatory", "Lung", "Necrosis", "Stroma", "Tumor", "Vessel"]
COLORS = np.array([(54,54,54), (251,217,204), (41,150,191), (1,0,128),
                   (142,142,142), (71,169,45), (229,76,37), (187,47,154)], dtype=np.uint8)
EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PATTERN = re.compile(r"^(.*?)_(\d+)_x(-?\d+)_(\d+)_(\d+)_y(-?\d+)_(\d+)_(\d+)$")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class PatchDataset(Dataset):
    def __init__(self, files, background_threshold):
        self.files = files
        self.background_threshold = background_threshold

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            with Image.open(self.files[index]) as image:
                image = image.convert("RGB")
                a = np.asarray(image, dtype=np.uint8)
                maximum, minimum = a.max(axis=2), a.min(axis=2)
                blank_score = float(((maximum >= 220) & ((maximum - minimum) <= 20)).mean())
                if blank_score >= self.background_threshold:
                    return torch.zeros(3, 224, 224), blank_score, True
                return TRANSFORM(image), blank_score, True
        except Exception:
            return torch.zeros(3, 224, 224), float("nan"), False


def parse_filename(filename):
    match = PATTERN.match(Path(filename).stem)
    if match is None:
        raise ValueError(f"Unexpected patch filename: {Path(filename).name}")
    ids, size, x, x_idx, max_x, y, y_idx, max_y = match.groups()
    return [ids, int(size), int(x), int(y), int(x_idx), int(y_idx), int(max_x), int(max_y)]


def save_map(df, filename):
    # Missing/corrupt patches are represented using the background color.
    height, width = int(df.y_idx.max()) + 1, int(df.x_idx.max()) + 1
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = COLORS[0]
    valid = df.readable == 1
    image[df.loc[valid, "y_idx"].to_numpy(dtype=int),
          df.loc[valid, "x_idx"].to_numpy(dtype=int)] = COLORS[df.loc[valid, "pred"].to_numpy(dtype=int)]
    io.imsave(str(filename), image, check_contrast=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", required=True, help="Directory containing one subfolder per WSI")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--uni-checkpoint", required=True, help="UNI pytorch_model.bin")
    parser.add_argument("--classifier", required=True, help="Fitted UNI_8class_classifier.joblib")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--background-threshold", type=float, default=0.997)
    parser.add_argument("--save-probs", action="store_true", help="Include eight-class probability arrays in CSV")
    args = parser.parse_args()

    root, output = Path(args.patch_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                              init_values=1e-5, num_classes=0, dynamic_img_size=True)
    state = torch.load(args.uni_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = joblib.load(args.classifier)  # Load only trusted model files.
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("Classifier must implement predict_proba")
    if not np.array_equal(classifier.classes_, np.arange(8)):
        raise ValueError("Classifier classes must be integer labels 0 through 7")

    qc_path = output / "UNI_inference_QC.csv"
    if qc_path.exists():
        try:
            qc_records = pd.read_csv(qc_path).to_dict("records")
        except Exception:
            qc_records = []
    else:
        qc_records = []

    slide_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for slide_dir in slide_dirs:
        slide = slide_dir.name
        destination = output / slide
        csv_path = destination / f"{slide}_prediction_256.csv"
        map_path = destination / f"{slide}_map_256.png"
        if csv_path.exists() and map_path.exists():
            print(f"Skipping completed slide: {slide}")
            continue

        files = sorted(path for path in slide_dir.iterdir() if path.suffix.lower() in EXTENSIONS)
        if not files:
            print(f"Skipping slide without patches: {slide}")
            continue

        # Validate filename metadata before time-consuming inference.
        metadata = pd.DataFrame([parse_filename(path.name) for path in files],
                                columns=["ids", "patch_size", "x", "y", "x_idx", "y_idx", "max_x", "max_y"])
        if (metadata[["x_idx", "y_idx"]] < 0).any().any():
            raise ValueError(f"Negative patch-grid coordinates in {slide}")
        metadata.insert(0, "fname", [path.name for path in files])
        dataset = PatchDataset(files, args.background_threshold)
        loader_options = dict(batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
        if args.workers > 0:
            loader_options.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(dataset, **loader_options)

        count = len(files)
        predictions = np.zeros(count, dtype=np.uint8)
        probabilities = np.zeros((count, 8), dtype=np.float32)
        blank_scores = np.full(count, np.nan, dtype=np.float32)
        readable = np.zeros(count, dtype=bool)
        position = 0

        print(f"Processing {slide}: {count} patches")
        with torch.inference_mode():
            for images, scores, ok in tqdm(loader, desc=slide, unit="batch"):
                n = len(scores)
                indices = np.arange(position, position + n)
                position += n
                scores_np = scores.numpy()
                ok_np = ok.numpy().astype(bool)
                blank_scores[indices] = scores_np
                readable[indices] = ok_np
                background = ok_np & (scores_np >= args.background_threshold)
                probabilities[indices[background], 0] = 1.0
                tissue = ok_np & ~background
                if not tissue.any():
                    continue

                selected = indices[tissue]
                x = images[tissue].to(device, non_blocking=True)
                amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                with amp:
                    features = model(x).float().cpu().numpy()
                p = classifier.predict_proba(features)
                probabilities[selected] = p
                predictions[selected] = np.argmax(p, axis=1).astype(np.uint8)

        metadata["blank_score"] = blank_scores
        metadata["optical_background"] = (readable & (blank_scores >= args.background_threshold)).astype(np.uint8)
        metadata["readable"] = readable.astype(np.uint8)
        metadata["prob_max"] = probabilities.max(axis=1)
        metadata["pred"] = predictions
        metadata["pred_class"] = [CLASSES[i] for i in predictions]
        if args.save_probs:
            metadata["probs"] = [p.tolist() for p in probabilities]

        destination.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(csv_path, index=False)
        save_map(metadata, map_path)

        corrupt = [files[i].name for i in np.flatnonzero(~readable)]
        if corrupt:
            pd.DataFrame({"file": corrupt}).to_csv(destination / f"{slide}_corrupt.csv", index=False)
        qc_records = [row for row in qc_records if str(row.get("slide")) != slide]
        n_readable = int(readable.sum())
        n_background = int((readable & (blank_scores >= args.background_threshold)).sum())
        qc_records.append({"slide": slide, "total_patches": count, "readable": n_readable,
                           "corrupt": len(corrupt), "background": n_background,
                           "background_pct": 100 * n_background / max(n_readable, 1),
                           "tissue": n_readable - n_background})
        pd.DataFrame(qc_records).to_csv(qc_path, index=False)
        print(f"Saved {csv_path} and {map_path}")

    print(f"Completed. Outputs: {output}")


if __name__ == "__main__":
    main()
