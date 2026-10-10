# Commands

Recorded on 2026-10-10 (Asia/Shanghai).

```bash
git -C /workspace/python_pro/test1/RopStitch rev-parse --show-toplevel
git -C /workspace/python_pro/test1/RopStitch rev-parse HEAD
git -C /workspace/python_pro/test1/RopStitch branch --show-current
git -C /workspace/python_pro/test1/RopStitch status --short
git switch -c reproduce-baseline
find data/UDIS-D/{training,testing}/{input1,input2} -maxdepth 1 -type f | wc -l
python --version
nvidia-smi
python  # imported torch, torchvision, skimage, cv2, numpy and queried CUDA/cuDNN/device properties
```

Intended training command (not executed because the dataset count gate failed):

```bash
cd /workspace/python_pro/test1/RopStitch/woCoefNet/Codes
python train.py --gpu 0 --batch_size 4 --max_epoch 100 --train_path /workspace/python_pro/test1/RopStitch/data/UDIS-D/training/ --test_path /workspace/python_pro/test1/RopStitch/data/UDIS-D/testing/
```

No smoke-test or full-training command was run. In addition to the data mismatch, CUDA was unavailable in the current container session.
