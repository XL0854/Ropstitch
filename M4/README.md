# 方案 2 (组合 B): 跨图交叉注意力 + 卷积式拓扑网格头 (Combo B)

## 1. 架构动机与根本性突破
前两阶段（M2, M3）的实验明确揭示了原版 RopStitch 的核心天花板：
* **原版缺陷**: TPS 回归头将 `[B, 128, 4, 4]` 特征图用 `Flatten` 拍成 2048 维一维向量，再用全连接层（Dense Linear）硬预测 338 个网格点坐标，**彻底摧毁了 2D 图像局部拓扑与几何邻域信息**，且两图缺乏深层上下文交互。
* **组合 B 的重塑突破**:
  1. **跨图交叉注意力 (Cross-Attention Matcher)**: 在特征层实现双向 Query-Key-Value 交互，让图 1 主动感知图 2 的对应特征，解决弱纹理与大视差匹配难题。
  2. **全卷积网格预测头 (ConvMeshHead)**: 彻底铲除 `Flatten + Linear`，采用纯 2D 卷积与自适应空间池化，直接映射局部感受野到对应网格点，保留完整的二维空间拓扑结构。
  3. **解冻主干端到端微调**: 解冻配准主干网络的 `active_resnet_backbone`，采用分层学习率联合优化。

## 2. 代码组织
- `modules_b.py`: 核心模块实现（`CrossAttentionMatcher` 与 `ConvMeshHead`）
- `train_b.py`: 端到端联合微调训练流水线
- `test_b.py`: 推断与三分搜索评测工具
- `model_b/`: 训练权重保存目录
- `summary_b/`: TensorBoard 可视化目录
