# RopStitch Stage 1 Reproduction Report

## Outcome

Stage 1 training was correctly stopped before the smoke test. The test set is incomplete: `testing/input1` has 1105 images while `testing/input2` has 1106, with `000001.jpg` missing from `testing/input1`. The active container session also cannot access the NVIDIA driver. These are explicit stop conditions in the reproduction request.

## Repository

- Absolute path: `/workspace/python_pro/test1/RopStitch`
- Commit: `c47a5aef88ec31abe1d1dc0f470b42dbacc904c1`
- Initial branch: `main`
- Reproduction branch: `reproduce-baseline`
- A pre-existing modification to `woCoefNet/Codes/train.py` changes the default GPU and local data paths. It was preserved and not overwritten.
- Pre-existing untracked content consists of the local `data/` tree and Python bytecode caches.

Full initial status and modification purpose are recorded in `experiments/reproduction/source_initial_state.txt`.

## Environment

- OS: Linux 5.15.0-139-generic, x86_64, glibc 2.31
- Python: 3.9.25 (`ropstitch` Conda environment)
- PyTorch: 2.8.0+cu128
- torchvision: 0.23.0+cu128
- CUDA build: 12.8
- cuDNN: 91002
- scikit-image: 0.24.0
- OpenCV: 4.13.0
- NumPy: 2.0.1
- CUDA available: no
- Visible GPUs: 0
- GPU model and memory: unavailable because `nvidia-smi` could not communicate with the NVIDIA driver

## Dataset

- Train path: `/workspace/python_pro/test1/RopStitch/data/UDIS-D/training/`
- Test path: `/workspace/python_pro/test1/RopStitch/data/UDIS-D/testing/`
- `training/input1`: 10440
- `training/input2`: 10440
- `testing/input1`: 1105
- `testing/input2`: 1106
- Missing test counterpart: `testing/input1/000001.jpg`

## Training Configuration

- Intended seed: 2026
- Intended batch size: 4
- Intended epochs: 100
- Optimizer: AdamW, learning rate 4e-5, betas (0.9, 0.999), eps 1e-8, weight decay 1e-4
- Scheduler: ExponentialLR, gamma 0.97
- Grid: 12 x 12 cells
- Training input: 512 x 512
- Intended checkpoint directory: `/workspace/python_pro/test1/RopStitch/experiments/reproduction/stage1/checkpoints/`

The intended command is preserved in `experiments/reproduction/commands.md`; it was not executed.

## From-scratch Protection

The official script automatically resumes from the lexicographically last `.pth` in `MODEL_DIR`, restoring model, optimizer, epoch, and global iteration. No `.pth` files were found under `woCoefNet/model_homo`, so no backup move was necessary. No pretrained weight was used.

## Results

- Smoke test: not run due to failed dataset gate
- One-epoch result: not available
- 100 epochs completed: no
- Initial/final loss: not available
- Best validation SSIM: not available
- Final checkpoint: not available
- Final checkpoint reload: not performed
- Ten-pair inference: not performed
- Validation images: none
- Total training time: 0

## Files Changed by This Reproduction Attempt

- `experiments/reproduction/environment.txt`
- `experiments/reproduction/dataset_check.txt`
- `experiments/reproduction/source_initial_state.txt`
- `experiments/reproduction/commands.md`
- `experiments/reproduction/stage1/config.json`
- `experiments/reproduction/stage1/smoke_test.log`
- `experiments/reproduction/stage1/train.log`
- `experiments/reproduction/stage1/REPORT.md`
- Required empty output directories were created.

No model structure, loss, SSIM/PSNR formula, augmentation, or training algorithm was changed. No stage-2 code was run.

## Problems and Readiness

1. Restore the missing `testing/input1/000001.jpg` from the correct UDIS-D source so both test sides contain 1106 matching images.
2. Restore NVIDIA driver/GPU passthrough until both `nvidia-smi` and `torch.cuda.is_available()` succeed.

The run does **not** meet the conditions to enter stage 2 because stage 1 has not trained or passed validation.
