#!/usr/bin/env python3
"""Postprocess UNI tissue predictions with optional GeoJSON contours and export maps.

Requires Shapely >= 2.0. Input CSVs are produced by predict_tissue_classes.py.
"""
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage import io
from tqdm import tqdm

CLASSES = np.array([
    "Background", "Bronchi", "Inflammatory", "Lung",
    "Necrosis", "Stroma", "Tumor", "Vessel",
])
COLORS = np.array([
    (54, 54, 54), (251, 217, 204), (41, 150, 191), (1, 0, 128),
    (142, 142, 142), (71, 169, 45), (229, 76, 37), (187, 47, 154),
], dtype=np.uint8)
SUFFIX = "_prediction_256.csv"


def slide_id(csv_path):
    filename = Path(csv_path).name
    if not filename.endswith(SUFFIX):
        raise ValueError(f"Unexpected prediction filename: {filename}")
    return filename[:-len(SUFFIX)]


def output_paths(slide, post_root, final_root):
    return (
        post_root / f"{slide}_prediction_geo_01pct.csv",
        post_root / f"{slide}_map_geo_01pct.png",
        final_root / f"{slide}.png",
    )


def process_one(task):
    csv_path, geo_root, post_root, final_root, threshold, chunk_size, overwrite = task
    csv_path = Path(csv_path)
    post_root, final_root = Path(post_root), Path(final_root)
    slide = slide_id(csv_path)
    csv_out, map_out, final_out = output_paths(slide, post_root, final_root)
    try:
        if not overwrite and csv_out.exists() and map_out.exists() and final_out.exists():
            return [slide, np.nan, np.nan, np.nan, False, "already_done", np.nan]

        d = pd.read_csv(csv_path)
        required = {"x_idx", "y_idx", "x", "y", "patch_size", "pred"}
        missing = required.difference(d.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if d.empty:
            raise ValueError("Empty prediction CSV")
        if d[list(required)].isna().any().any():
            raise ValueError("Missing values in required coordinate/prediction columns")

        x = d.x_idx.to_numpy(dtype=int)
        y = d.y_idx.to_numpy(dtype=int)
        if np.any(x < 0) or np.any(y < 0):
            raise ValueError("Negative patch grid indices")
        px = d.x.to_numpy(dtype=float)
        py = d.y.to_numpy(dtype=float)
        sz = d.patch_size.to_numpy(dtype=float)
        if np.any(sz <= 0):
            raise ValueError("patch_size must be positive")
        orig = d.pred.to_numpy(dtype=int)
        if np.any((orig < 0) | (orig >= len(CLASSES))):
            raise ValueError("pred must be integer tissue class indices from 0 to 7")

        tissue = orig != 0
        ids = np.flatnonzero(tissue)
        overlap = np.ones(len(d), dtype=np.float32)
        geo_used, coverage, reason = False, np.nan, "geojson_missing"
        geo_path = Path(geo_root) / f"{slide}.geojson" if geo_root else None

        if geo_path is not None and geo_path.exists():
            with geo_path.open(encoding="utf-8") as f:
                geojson = json.load(f)
            polygons = []
            for feature in geojson.get("features", []):
                try:
                    geometry = shape(feature["geometry"])
                    if not geometry.is_valid:
                        geometry = geometry.buffer(0)
                    if not geometry.is_empty:
                        polygons.append(geometry)
                except (KeyError, TypeError, ValueError):
                    continue
            if polygons:
                region = unary_union(polygons)
                if not region.is_valid:
                    region = region.buffer(0)
                if not region.is_empty:
                    overlap = np.zeros(len(d), dtype=np.float32)
                    for start in range(0, len(ids), chunk_size):
                        ii = ids[start:start + chunk_size]
                        boxes = shapely.box(px[ii], py[ii], px[ii] + sz[ii], py[ii] + sz[ii])
                        inside = shapely.contains(region, boxes)
                        intersects = shapely.intersects(boxes, region)
                        overlap[ii[inside]] = 1.0
                        edge = intersects & ~inside
                        if edge.any():
                            jj = ii[edge]
                            intersection = shapely.intersection(boxes[edge], region)
                            overlap[jj] = shapely.area(intersection) / (sz[jj] ** 2)
                    coverage = float((overlap[tissue] > 0).mean()) if tissue.any() else 1.0
                    geo_used, reason = True, "geojson_used"
                else:
                    coverage, reason = 0.0, "geojson_empty"
            else:
                coverage, reason = 0.0, "geojson_empty"
        elif geo_root is None:
            reason = "geojson_not_provided"

        pred = orig.copy()
        if geo_used:
            pred[overlap < threshold] = 0

        # One pixel per patch-grid cell, matching the original implementation.
        grid = np.zeros((int(y.max()) + 1, int(x.max()) + 1), dtype=np.uint8)
        grid[y, x] = pred.astype(np.uint8)

        d["geo_overlap"] = overlap
        d["pred_post"] = pred
        d["pred_class_post"] = CLASSES[pred]
        d["geo_used"] = geo_used
        d["geo_coverage"] = coverage
        d["postprocess_reason"] = reason

        post_root.mkdir(parents=True, exist_ok=True)
        final_root.mkdir(parents=True, exist_ok=True)
        d.to_csv(csv_out, index=False)
        io.imsave(str(map_out), COLORS[grid], check_contrast=False)
        shutil.copy2(map_out, final_out)

        removed = int(((orig != 0) & (pred == 0)).sum())
        return [slide, len(d), int(tissue.sum()), coverage, geo_used, reason, removed]
    except Exception as exc:
        return [slide, np.nan, np.nan, np.nan, False, f"ERROR: {exc}", np.nan]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", required=True, type=Path,
                        help="Root containing recursive *_prediction_256.csv files")
    parser.add_argument("--geo-root", type=Path, default=None,
                        help="Optional directory containing slide-name.geojson contours")
    parser.add_argument("--output", required=True, type=Path,
                        help="Postprocessed CSVs, maps and summary directory")
    parser.add_argument("--final-maps", required=True, type=Path,
                        help="Directory for exported slide-name.png maps")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Minimum fraction of patch overlapping tissue (default 0.01)")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.threshold <= 1):
        parser.error("--threshold must be between 0 and 1")
    if args.chunk_size < 1 or args.workers < 1:
        parser.error("--chunk-size and --workers must be positive")
    if not args.csv_root.is_dir():
        parser.error(f"CSV root does not exist: {args.csv_root}")
    if args.geo_root is not None and not args.geo_root.is_dir():
        parser.error(f"GeoJSON root does not exist: {args.geo_root}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.final_maps.mkdir(parents=True, exist_ok=True)

    all_files = sorted(args.csv_root.rglob(f"*{SUFFIX}"))
    tasks = []
    skipped = 0
    for file in all_files:
        slide = slide_id(file)
        paths = output_paths(slide, args.output, args.final_maps)
        if not args.overwrite and all(p.exists() for p in paths):
            skipped += 1
            continue
        tasks.append((str(file), str(args.geo_root) if args.geo_root else None,
                      str(args.output), str(args.final_maps), args.threshold,
                      args.chunk_size, args.overwrite))
    print(f"Total slides: {len(all_files)} | Already done: {skipped} | Remaining: {len(tasks)}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Slides", unit="slide"):
            result = future.result()
            rows.append(result)
            if result[5].startswith("ERROR:"):
                print(f"ERROR {result[0]}: {result[5]}")

    summary = args.output / "geojson_postprocessing_summary.csv"
    if rows:
        new = pd.DataFrame(rows, columns=[
            "slide", "patches", "original_tissue", "geo_coverage",
            "geo_used", "reason", "removed",
        ])
        if summary.exists():
            old = pd.read_csv(summary)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates("slide", keep="last")
        new.to_csv(summary, index=False)
    print(f"Done. New slides attempted: {len(rows)}")
    print(f"Postprocessed outputs: {args.output}")
    print(f"Final maps: {args.final_maps}")


if __name__ == "__main__":
    main()
