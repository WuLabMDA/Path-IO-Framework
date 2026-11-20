# WSI Patch Extraction

This module extracts non-overlapping RGB image patches from Aperio `.svs` whole-slide images for subsequent UNI-based tissue classification and habitat map generation.

## Requirements

- Python 3.10 or a compatible version
- OpenSlide and its native library (`openslide-python`)
- Pillow
- NumPy (imported by the original script)

Install the Python packages with `pip install openslide-python Pillow numpy`. The OpenSlide native library may require separate installation depending on your operating system.

## Input and output

Set `slide_rootpath` and `out_rootpath` in `svs2tile.py` to your own input and output directories. The script searches for `.svs` files directly within the input directory and creates one subdirectory per slide.

| Slide property `aperio.AppMag` | Extraction tile | Saved patch |
|---|---|---|
| `20` | 256 × 256 pixels at level 0 | 256 × 256 PNG |
| `40` | 512 × 512 pixels at level 0 | Resized to 256 × 256 PNG |

The script uses non-overlapping tiles and excludes incomplete tiles along the right and bottom edges. It does not apply tissue/background filtering. Slides without `aperio.AppMag` equal to `20` or `40` are skipped.

Patch filenames retain the original level-0 coordinates and tile-grid indices, for example:

`slideA_256_x0_0_100_y0_0_80.png`

## Usage

```bash
python svs2tile.py 0 20
```

The first argument is the inclusive starting index and the second is the exclusive ending index in the sorted list of `.svs` files. The script uses multiprocessing for tile extraction.

**Notes:** The original script is preserved without functional changes. Existing slide output folders are skipped, even if a prior extraction was incomplete. The script also reads a full-slide image at the selected level when determining dimensions; this can require substantial memory for very large WSIs. No patient images or intermediate patches are included in this repository.
