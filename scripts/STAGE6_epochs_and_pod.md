# 阶段6：batch/epoch 说明

## 一、batch 是什么、有什么用

- **batch** = 每次梯度更新时使用的样本数（`batch_size`）
- **作用**：把 `n_train × n_points` 个监督对分成多批，每批做一次前向+反向，更新参数
- **批次数/epoch** = `ceil((n_train × n_points) / batch_size)`

例如 Beltrami 默认：`120 × 5000 = 600,000` 监督对，`batch_size=24` → 约 25,000 批/epoch。

---

## 二、epoch 80–120 与 PINN 几千上万的区别

| 框架 | 1 epoch 含义 | 梯度更新次数/epoch | 10000 epochs ≈ |
|------|-------------|-------------------|-----------------|
| **PINN** (N-S_equations) | 1 次 optimizer.step，通常整批 collocation 点 | 1 | 1 万次更新 |
| **DeepONet** (本仓库) | 1 轮完整遍历所有 (样本,点) 对 | 批次数（如 25,000） | 2.5 亿次更新 |

所以：
- PINN 的 10000 epochs ≈ 1 万次梯度更新
- DeepONet 的 80 epochs ≈ 80 × 25,000 = **200 万次**梯度更新

**结论**：DeepONet 的 80–120 epochs 在总更新次数上远多于 PINN 的几千 epoch，因此是足够的。

---

## 三、POD

POD（PODTrunk、build_ns_pod_basis）已移至**阶段 7** 规划。
