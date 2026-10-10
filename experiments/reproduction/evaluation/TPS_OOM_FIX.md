# Exact-resolution TPS OOM fix

## Failure

`wCoefNet/Codes/test.py` failed on `data/stitch_real` pair 122. Both source images are 3264 x 2448. The original TPS implementation materialized the complete `[batch, 169, height*width]` radial-basis grid and attempted an additional 8.94 GiB CUDA allocation.

## Change

`wCoefNet/Codes/utils/torch_tps_transform.py` now partitions large output grids into complete row blocks. Each block uses the same normalized coordinates, TPS radial basis, transformation matrix, and bilinear interpolation formula as the official implementation. Results are concatenated along the output-height dimension.

The change does not resize or skip images and does not modify model weights, meshes, coefficients, metrics, or the TPS formula. Outputs with at most 2,000,000 pixels continue to use the untouched official path. Larger outputs use 128-row chunks by default.

Optional environment controls:

```bash
ROPSTITCH_TPS_CHUNK_ROWS=64 python test.py ...
ROPSTITCH_TPS_CHUNK_THRESHOLD=0 python test.py ...  # force chunking for every output
```

Smaller row blocks use less peak memory but take longer. Setting the threshold to a very large integer restores the original full-grid execution for comparison.

## Equivalence check

With seed 2026, a 6-channel random input, a 13 x 13 control grid, an output of 73 x 91, and forced 17-row chunks:

```text
reference_shape = (1, 6, 73, 91)
chunked_shape   = (1, 6, 73, 91)
all_finite      = True
max_abs_error   = 0.0
mean_abs_error  = 0.0
allclose_1e-5   = True
```

The module also passed `python -m py_compile` and `git diff --check`.

## Official fallback preserved

The official behavior for an output canvas exceeding 4000 pixels is preserved exactly: `build_output_model` returns `(None, False)`, and `test_once` substitutes the unwarped input pair before computing PSNR/SSIM. The briefly evaluated alternative of excluding these candidates was reverted because it changes official results.
