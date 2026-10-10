# B0 — Official RopStitch baseline reproduced

## Status

B0 successfully reproduces the paper Table I aggregate PSNR and SSIM values at the reported precision. No model architecture, learned coefficient logic, loss, ternary-search policy, PSNR formula, SSIM formula, or official over-4000-pixel fallback was changed.

The two checkpoints are official downloaded weights, not weights trained from scratch in this workspace:

- Stage 1: `/workspace/python_pro/test1/RopStitch/woCoefNet/model_homo/epoch100_model.pth`
- Stage 2: `/workspace/python_pro/test1/RopStitch/wCoefNet/model_coef/epoch050_coefmodel.pth`

## Dataset and execution

- UDIS-D test path: `/workspace/python_pro/test1/RopStitch/data/UDIS-D/testing/`
- Evaluated UDIS-D pairs: 1105
- `testing/input1`: 1105 files
- `testing/input2`: 1105 files
- Random seed: 114514
- Batch size: 1
- Ternary-search iterations (`max_epoch` in official test CLI): 5
- GPU: NVIDIA GeForce RTX 4090, physical GPU 3

No entire UDIS-D sample was reported as failed in the completed run. Official candidate-level `warp size is too huge` fallbacks occurred. An OOM occurred in a preliminary run on a 3264 x 2448 real-image pair; the completed baseline uses row-chunked evaluation of the mathematically identical TPS grid to reduce peak memory without resizing images or changing valid outputs.

## Results

| Metric | Easy | Moderate | Hard | Average |
|---|---:|---:|---:|---:|
| PSNR | 29.921793 | 24.95519 | 20.595434 | 24.69902 |
| SSIM | 0.92638874 | 0.84496814 | 0.6717298 | 0.8000622 |

The operator confirmed these values align with paper Table I. Difference from the paper is recorded as 0 at the displayed precision.

## Reproducibility evidence

- `environment.txt`: software, CUDA and GPU inventory
- `command.txt`: complete archived test command and seed
- `checkpoint_hashes.txt`: SHA256 for both loaded checkpoints
- `summary.json`: machine-readable aggregate metrics
- `test.log`: recovered final console results and run notes
- `code_snapshot/`: exact frozen evaluation source snapshot plus GPU-visible run metadata
- `code_snapshot/git_diff.patch`: complete pre-freeze working-tree diff

## Per-sample metrics limitation

The successful process retained PSNR/SSIM only in Python lists and printed per-image SSIM but not per-image PSNR. Its stdout was not redirected to a file. Consequently, historical per-image values cannot be reconstructed exactly from the fused JPEG outputs. `per_sample_metrics.csv` currently contains its schema header only; fabricated rows were not added. The frozen test code now writes the already-computed values after inference without feeding them back into search, sorting, or metric computation. A repeat execution of `command.txt` is required to populate 1105 genuine rows.

This archive freezes B0 only. It contains no grid metrics, no `coef_net.eval()` calibration change, and no M1 work.
