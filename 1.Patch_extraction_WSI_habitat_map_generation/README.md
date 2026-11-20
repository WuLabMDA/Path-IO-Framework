# Module 1 — Patch extraction and WSI habitat map generation

This module extracts H&E WSI tiles, obtains UNI features, classifies patches into eight tissue compartments, and reconstructs slide-level habitat maps. Tissue contour GeoJSON files may be generated as part of the tissue-processing workflow.

## Contents
- `patch_extraction/svs2tile.py`: WSI patch extraction
- `tissue_classification/uni_tissue_classifier.py`: UNI embedding extraction and eight-class classifier workflow
- `inference/predict_tissue_classes.py`: tissue classification inference
- `habitat_map_generation/generate_habitat_maps.py`: habitat map reconstruction

**Model access:** UNI backbone and the trained eight-class classifier are **not included** in this public repository. Request access from the corresponding authors, subject to applicable model licenses and institutional policies. Follow the module-specific README and script `--help` for exact command options. Never upload patient data or model weights to the public repository.
