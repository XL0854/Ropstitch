# M3 阶段：拓扑感知残差微调方案 (TAPR + 可微 SSIM 损失)

## 1. 方案概述
- **阶段名称**: M3 阶段 - 拓扑感知网格残差精修网络 (Topology-Aware Residual Refiner, TAPR)
- **核心模块**:
  - `tapr_refiner.py`: 拓扑感知网格残差预测网络（多尺度 ResBlock + 零初始化兜底）
  - `loss_m3.py`: 可微 Masked SSIM 结构损失 + L1 光度对齐损失 + 拓扑正则化约束
- **训练入口**: `train_m3.py`
- **测试入口**: `test_m3.py`
- **训练环境**: NVIDIA GeForce RTX 4090, 15 Epochs 全量训练完成

## 2. 核心架构亮点
1. **2D 局部空间感知**: 摒弃原版全连接（Linear）层破坏几何空间的缺陷，直接在 2D 特征图与重叠残差差分图上预测网格位移。
2. **绝对零初始化兜底（Zero-Init）**: 初始残差位移严格为 0，保证模型初始状态严格对齐官方基线 B0，绝不倒退。
3. **可微 SSIM 导向**: 让梯度直接朝着提升人眼结构相似度（SSIM）的方向优化。

## 3. 文件组织
- `tapr_refiner.py`: 残差精修网络实现
- `loss_m3.py`: 复合损失函数（L1 + SSIM + 平滑度）
- `train_m3.py`: 训练流水线
- `test_m3.py`: 推断与三分搜索评测工具
