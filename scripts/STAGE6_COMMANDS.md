# 阶段 6 正式实验启动命令

## 一、一键脚本（推荐）

```bash
# 全部实验：路线 A + 路线 B，hard-truncation + multi-CLS，15 epochs
python scripts/run_stage6_experiments.py --route all --branch both --epochs 15

# 快速模式（约 10x 加速）：n_train=50, nx=ny=32, n_collocation=64
python scripts/run_stage6_experiments.py --route all --branch both --epochs 15 --fast

# 极速模式（每实验约 30s）：n_train=20, nx=ny=16, n_collocation=16
python scripts/run_stage6_experiments.py --route all --branch both --epochs 5 --ultra

# 或使用批处理（Windows）
scripts\run_stage6.bat 15
```

## 二、分步命令

### 路线 A 先跑（参数化输入）

```bash
# Kovasznay + Beltrami，各跑 hard 和 cls
python scripts/run_stage6_experiments.py --route A --branch both --epochs 15
```

### 路线 B 再跑（边界/初值序列输入）

```bash
python scripts/run_stage6_experiments.py --route B --branch both --epochs 15
```

### 单独对比 hard vs multi-CLS

```bash
# 仅 hard-truncation
python scripts/run_stage6_experiments.py --route all --branch hard --epochs 15

# 仅 multi-CLS
python scripts/run_stage6_experiments.py --route all --branch cls --epochs 15
```

## 三、实验矩阵（最小）

| 路线 | Case | Branch | 说明 |
|------|------|--------|------|
| A | ns_kovasznay_parametric | hard | Re 参数 → (u,v,p) |
| A | ns_kovasznay_parametric | cls | 同上，Multi-CLS |
| A | ns_beltrami_parametric | hard | (Re,a,d) → (u,v,p) |
| A | ns_beltrami_parametric | cls | 同上，Multi-CLS |
| B | ns_kovasznay_bc2field | hard | 入口序列 → (u,v,p) |
| B | ns_kovasznay_bc2field | cls | 同上，Multi-CLS |
| B | ns_beltrami_ic2field | hard | 初值序列 → (u,v,w,p) |
| B | ns_beltrami_ic2field | cls | 同上，Multi-CLS |

共 **8 组** 实验，每组 15 epochs。

## 四、输出表格示例

脚本运行结束后会打印并保存到 `experiments/stage6/stage6_summary_epochs15.csv`：

| Case | Branch | Params | Time(s) | Loss | RelL2 | TestMSE |
|------|--------|--------|---------|------|-------|---------|
| ns_kovasznay_parametric | hard | xxx | xxx | xxx | xxx | xxx |
| ns_kovasznay_parametric | cls | xxx | xxx | xxx | xxx | xxx |
| ... | ... | ... | ... | ... | ... | ... |
