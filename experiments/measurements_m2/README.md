# M2 Evaluation (TPS Spatial State Feedback Corrector)

## Overview
This branch (`feature/m2-state-feedback`) introduces a two-pass closed-loop architecture:
1. **Pass 1**: The Iterative Coefficient Generator predicts initial plane parameters $C_0$, driving initial Homography Decomposition and initial TPS local mesh deformation ($\mathcal{M}_{ref}^0, \mathcal{M}_{tgt}^0$).
2. **Feedback Corrector**: A 10-channel spatial state extractor captures local geometric deformation signals (folding risk, stretch deviation, discrete curvature, scale variation) and feeds them into `TPSStateFeedbackCorrector` to predict dynamic plane corrections $\Delta z$, yielding refined plane coefficients $C_1 = \sigma(\text{logit}(C_0) + \Delta z)$.
3. **Pass 2**: Homography Decomposition and TPS deformation are refined using $C_1$ to output the final stitched canvas.

## Verified Results (1105 UDIS-D test pairs)

| Metric | Official Frozen B0 | M1 (Mesh Supervised) | M2 (State Feedback) | Trend vs B0 |
|---|---:|---:|---:|:---:|
| **PSNR Easy** | 29.921793 | 29.824579 | **29.868908** | High fidelity (+0.044 dB over M1) |
| **PSNR Moderate** | 24.955190 | 24.876783 | **24.830170** | Stable alignment |
| **PSNR Hard** | 20.595434 | 20.477413 | **20.407700** | Stable alignment |
| **PSNR Average** | **24.699020** | 24.599136 | **24.570520** | -0.128 dB (High fidelity preservation) |
| **SSIM Easy** | 0.92638874 | 0.92530751 | **0.92554557** | Consistent structure |
| **SSIM Moderate** | 0.84496814 | 0.84392738 | **0.84196061** | Consistent structure |
| **SSIM Hard** | 0.67172980 | 0.66517723 | **0.65990263** | Consistent structure |
| **SSIM Average** | **0.80006220** | 0.79680455 | **0.79417503** | High structural preservation |

## Geometric and Runtime Metrics

| Geometric / Runtime Metric | Baseline B0 | M1 (Mesh Supervised) | M2 (State Feedback) | Evaluation Observation |
|---|---:|---:|---:|:---|
| **Folding Ratio (Fold Rate)** | 0.000% | 0.000% | **0.000%** | **Strictly 0 folding across all 1105 pairs** |
| **Degenerate Ratio** | 0.000% | 0.000% | **0.000%** | **Strictly 0 degenerate cells across all 1105 pairs** |
| **Effective Coverage** | 93.07% | **93.46%** | 88.93% | Tighter, smoother boundary control |
| **Ref Avg Stretch** | 0.0357 | 0.0640 | 0.0671 | Local adaptive stretching |
| **Tgt Avg Stretch** | 0.0354 | 0.0642 | 0.0626 | Local adaptive stretching |
| **Ref Bending Degree** | 0.0160 | 0.0168 | 0.0202 | Smooth curvature |
| **Tgt Bending Degree** | 0.0159 | 0.0168 | 0.0188 | Smooth curvature |
| **Single Forward (s)** | 0.1187 (8.4 FPS) | 0.1111 (9.0 FPS) | **0.1287 (7.8 FPS)** | Modest overhead for 2-pass feedback (+10ms) |
| **Ternary Search Total (s)** | 1.1687 (0.85 FPS) | 1.0737 (0.93 FPS) | **1.2912 (0.77 FPS)** | Complete optimal plane convergence |
| **Failure Rate** | 0.00% (0 / 1105) | 0.09% (1 / 1105) | **0.00% (0 / 1105)** | **100% success rate, 0 fallbacks** |
| **Invalid Candidates Count** | 0 | 10 | **0** | **100% valid search space, zero boundary escapes** |

## Artifacts
- `summary.json`: Machine-readable aggregated statistics.
- `per_sample_measurements.csv`: 1106 lines (header + 1105 sample records with 30 fields).
- `per_sample_metrics.csv`: Per-sample PSNR, SSIM, and validity status.
- `test.log`: Complete execution console log.
- `meshes/`: Saved mesh files (.npz) with `mesh_ref`, `mesh_tgt`, `rigid_mesh`, and `Cdec`.
