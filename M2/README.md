# M2 阶段成果与模型备份记录

## 1. 阶段概述
- **阶段名称**: M2 阶段 - TPS 状态反馈矫正器 (TPSStateFeedbackCorrector)
- **核心模块**: `feedback_network.py` (TPSStateFeedbackCorrector)
- **训练入口**: `train_m2.py`
- **测试入口**: `test_m2.py`
- **训练环境**: NVIDIA GeForce RTX 4090, PyTorch 2.x
- **训练轮数**: 20 Epochs (已全部完成)

## 2. 检查点路径
- `model_m2/epoch005_m2_model.pth`
- `model_m2/epoch010_m2_model.pth`
- `model_m2/epoch015_m2_model.pth`
- `model_m2/epoch020_m2_model.pth` (最终最优收敛权重)

## 3. 测试指标对比
- **官方基线 (B0)**: PSNR = 24.58 dB, SSIM = 0.862
- **M2 反馈网络**: PSNR = 24.45 dB, SSIM = 0.860
- **主要归因**: M2 仅调整 4 维标量形变系数，无法解决复杂的局部几何非刚性形变，且配准网络主干被完全冻结。

## 4. 备份代码文件
- `backup_m2/train_m2.py`
- `backup_m2/feedback_network.py`
- `backup_m2/test_m2.py`
