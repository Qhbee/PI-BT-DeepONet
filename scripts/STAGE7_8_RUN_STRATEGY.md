# Stage 7、8 实验运行策略

## 渐进式验证（避免浪费 30 小时）

1. **ULTRA 30 epochs**（约 30–40 min 全阶段）
   ```bash
   $env:STAGE7_ULTRA="1"; uv run python scripts/run_stage7_experiments.py --epochs 30 --checkpoint-every 5 --seed 42
   $env:STAGE8_ULTRA="1"; uv run python scripts/run_stage8_experiments.py --epochs 30 --checkpoint-every 5 --seed 42
   ```
   - 数据极小（12 样本），快速看 RelL2 能否下降
   - 若 RelL2 仍 >0.8，说明需更多数据

2. **FASTER 5 epochs**（约 30 min/阶段）
   ```bash
   $env:STAGE7_ULTRA="0"; uv run python scripts/run_stage7_experiments.py --epochs 5 --faster --checkpoint-every 5 --seed 42
   $env:STAGE8_ULTRA="0"; uv run python scripts/run_stage8_experiments.py --epochs 5 --faster --checkpoint-every 5 --seed 42
   ```
   - nx=15, nt=16，数据量适中
   - 5 epoch 后看 RelL2，若可接受则 resume 续训

3. **续训 5 epoch**（从 checkpoint 恢复）
   ```bash
   uv run python scripts/run_stage7_experiments.py --epochs 10 --faster --resume experiments/stage7/standard_pi_transformer_bayes/checkpoints/latest.pt --seed 42
   ```
   - 需在 config 或脚本中确保 epochs 大于 checkpoint 的 epoch

## 配置对比

| 模式 | n_train | nx×nt | total_samples | 5 epoch 耗时 | 30 epoch 耗时 |
|------|---------|-------|---------------|--------------|---------------|
| ULTRA | 12 | 12×11 | 1,584 | ~5 min | ~30 min |
| FASTER | 300 | 15×16 | 72,000 | ~30 min | ~3 h |
| FAST | 300 | 30×31 | 279,000 | ~2 h | ~6 h |
| 标准 | 300 | 100×101 | 3,030,000 | ~20 h | ~120 h |
