# Checkpoint 与恢复训练

## 功能

- **按 epoch 保存**：`checkpoint_every: 10` 表示每 10 个 epoch 保存一次
- **恢复训练**：`--resume experiments/xxx/checkpoints/latest.pt` 从最新 checkpoint 继续
- **随机种子**：config 中 `data.seed` 会写入 checkpoint，恢复时还原 RNG 状态
- **训练历史**：每 epoch 的 loss、rel_l2、test_mse 自动写入 `training_history.json`

## main.py 使用示例

```bash
# 首次训练，每 10 epoch 保存
uv run python main.py --config configs/paper_diffusion_reaction.yaml

# 若 30 epoch 后 loss 仍下降，想续训到 80 epoch：
# 1. 修改 config 中 epochs: 80
# 2. 从 latest.pt 恢复
uv run python main.py --config configs/paper_diffusion_reaction.yaml --resume experiments/diffusion_reaction/checkpoints/latest.pt
```

## 实验脚本使用示例

Stage 7 / Stage 8 / compare_4_combos / compare_ablation 均支持 `--checkpoint-every` 和 `--resume`：

```bash
# Stage 8：每 10 epoch 保存，便于续训
uv run python scripts/run_stage8_experiments.py --checkpoint-every 10 --seed 42

# 续训 trunk_fnn（需在 config 中把 epochs 改为 80 或 100）
uv run python scripts/run_stage8_experiments.py --resume experiments/stage8/trunk_fnn_transformer_bayes/checkpoints/latest.pt

# Stage 3 四组合
uv run python scripts/compare_4_combos.py --epochs 50 --checkpoint-every 10 --seed 42
```

训练结束后，`experiments/<run>/training_history.json` 会保存每 epoch 的 loss、rel_l2、test_mse，便于画曲线或判断是否需续训。

## 早停验证

先跑少量 epoch（如 5），看 loss/MSE 是否正常下降；若异常可调参后重跑。checkpoint 便于中断后快速续训。
