# Stage-1 Training Recorder

Run the official training command through the recorder from `woCoefNet/Codes`:

```bash
cd /workspace/python_pro/test1/RopStitch/woCoefNet/Codes
bash ../../experiments/reproduction/record_stage1_training.sh -- \
  python train.py \
  --gpu 0 \
  --batch_size 4 \
  --max_epoch 100 \
  --train_path /workspace/python_pro/test1/RopStitch/data/UDIS-D/training/ \
  --test_path /workspace/python_pro/test1/RopStitch/data/UDIS-D/testing/
```

The wrapper does not modify training. Every invocation creates a timestamped directory under `experiments/reproduction/stage1/runs/` containing:

- `train.log`: combined stdout and stderr;
- `metadata.txt`: command, Git commit/status/diff, environment and start/end GPU information;
- `gpu.csv`: GPU memory, utilization, temperature and power sampled every 10 seconds;
- `gpu_monitor_errors.log`: any sampling failures;
- `exit_status.txt`: exit code and timestamps;
- checkpoint listings from before and after training.

To sample every five seconds:

```bash
SAMPLE_SECONDS=5 bash experiments/reproduction/record_stage1_training.sh -- python ...
```

Important: the current official script incorrectly derives its checkpoint location from the process working directory rather than the source-file path. Starting it from `woCoefNet/Codes` makes the official `MODEL_DIR` resolve to `woCoefNet/model_homo`. The recorder warns about other working directories, inventories `woCoefNet/model_homo`, and does not redirect or move model files.

The repository's `TestDataset` sorts each side independently and pairs images by list index. Because only `testing/input2/000001.jpg` exists, merely ignoring the count mismatch would shift every subsequent validation pair. Move that unmatched file to a timestamped backup outside `testing/input2` before training, or restore its matching `testing/input1/000001.jpg`.

This recorder captures whatever losses and validation SSIM the official script prints. It cannot reconstruct values that the training process never prints. Per-image PSNR, SSIM, meshes, fold rate, stretch and CUDA-event inference time should be collected by a separate post-training evaluation script, because those are evaluation results rather than training-process telemetry.
