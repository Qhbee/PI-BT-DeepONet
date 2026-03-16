# 论文实验脚本

正式用于论文的实验脚本，参数写在代码内，无命令行参数，方便直接修改后运行。

## exp1_baseline_comparison.py

**实验 1**：Antiderivative 算例下，PI-BT-DeepONet 与基准模型对比。

| 模型                   | PI | 贝叶斯            | Branch      |
|----------------------|----|----------------|-------------|
| vanilla_deeponet     | 无  | 无              | FNN         |
| pi_deeponet          | 有  | 无              | FNN         |
| b_deeponet           | 无  | alpha-VI (α=1) | FNN         |
| transformer_deeponet | 无  | 无              | Transformer |
| pi_bt_deeponet       | 有  | alpha-VI (α=1) | Transformer |

**运行**：直接执行 `python exp1_baseline_comparison.py`，修改脚本顶部 `CONFIG` 调整参数。

**输出**：
- `experiments/paper/exp1_baseline_comparison/config.json` - 实验配置
- `experiments/paper/exp1_baseline_comparison/<model>/` - 各模型 log、checkpoints、training_history.json、result.json
- `experiments/paper/exp1_baseline_comparison/exp1_summary_epochs*.csv` - 汇总表
