# Module 2 — WSI habitat map preprocessing

- `tissue_packing.py`: combine habitat maps belonging to the same patient into a compact patient-level representation.
- `generate_patches.py`: extract 64 × 64 seven-channel one-hot habitat patches with stride 32, preserving the non-background tissue class order.

Inputs are tissue habitat maps from Module 1 and patient/slide identifiers. The output is a patient-level `.pt` cache containing `patches` and `coords`. **These `.pt` outputs are patient data, not model weights, and should be stored outside the public repository.**

Run each script's `--help` for available options. No trained model weights are distributed with this module.
