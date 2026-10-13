# Measurement Baseline (B1 Evaluation Pipeline)

## Overview
This branch (`measurement-baseline`) introduces a non-invasive geometric and runtime evaluation suite for RopStitch without modifying model architecture, search dynamics, or inference outputs.

## Verified Results (1105 UDIS-D test pairs)

| Metric | Official Frozen B0 | Measurement Baseline | Difference |
|---|---:|---:|---:|
| **PSNR Easy** | 29.921793 | 29.921793 | 0.000000 |
| **PSNR Moderate** | 24.955190 | 24.955190 | 0.000000 |
| **PSNR Hard** | 20.595434 | 20.595434 | 0.000000 |
| **PSNR Average** | 24.699020 | 24.699020 | 0.000000 |
| **SSIM Easy** | 0.92638874 | 0.92638874 | 0.000000 |
| **SSIM Moderate** | 0.84496814 | 0.84496814 | 0.000000 |
| **SSIM Hard** | 0.67172980 | 0.67172980 | 0.000000 |
| **SSIM Average** | 0.80006220 | 0.80006220 | 0.000000 |

## Geometric and Runtime Baselines
- **Folding ratio**: 0.000%
- **Degenerate cells ratio**: 0.000%
- **Average stretch**: Ref 0.0357, Tgt 0.0354
- **Maximum stretch**: Ref 0.1202, Tgt 0.1176
- **Scale variation (log-area std)**: Ref 0.0731, Tgt 0.0721
- **Bending degree (discrete curvature)**: Ref 0.0160, Tgt 0.0159
- **Effective coverage**: 93.07%
- **Single-forward inference time**: 0.119 s
- **Full ternary-search latency**: 1.169 s

## Artifacts
- `summary.json`: Machine-readable summary statistics.
- `per_sample_measurements.csv`: 1106 lines (header + 1105 sample records with 30 fields).
- `per_sample_metrics.csv`: Standard metric CSV.
- `test.log`: Complete console output log.
- `meshes/`: Saved mesh files (.npz) containing final `mesh_ref`, `mesh_tgt`, `rigid_mesh`, and `Cdec` for each sample.
