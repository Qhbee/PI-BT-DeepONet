"""
论文实验 1：Antiderivative 算例下，PI-BT-DeepONet 与基准模型对比。

模型：
  1. Vanilla DeepONet    - FNN branch + FNN trunk, 无 PI, 无贝叶斯
  2. PI-DeepONet         - FNN branch + FNN trunk, 有 PI, 无贝叶斯
  3. B-DeepONet          - FNN branch + FNN trunk, 无 PI, alpha-VI (alpha=1)
  4. Transformer-DeepONet - Transformer branch (mean pooling) + FNN trunk, 无 PI, 无贝叶斯
  5. PI-BT-DeepONet      - Transformer branch + FNN trunk, 有 PI, alpha-VI (alpha=1)

参数写在下方 CONFIG 中，无命令行参数，方便直接修改。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

# =============================================================================
# 配置参数（直接修改此处）
# =============================================================================
CONFIG = {
    # 数据
    "n_train": 300,
    "n_test": 100,
    "n_sensors": 50,
    "n_points_per_sample": 10,
    "length_scale": 0.5,
    "seed": 42,
    # 模型（公平对比：FNN 与 Transformer 容量相近）
    "output_dim": 20,
    "branch_hidden": [20, 20],
    "trunk_hidden": [20, 20],
    "num_sensors": 50,
    "coord_dim": 1,
    "transformer_d_model": 32,
    "transformer_nhead": 4,
    "transformer_num_layers": 2,
    "transformer_dropout": 0.1,
    "prior_sigma": 1.0,
    # 训练
    "epochs": 30,
    "batch_size": 64,
    "lr": 0.001,
    "pi_weight": 0.1,
    "bc_weight": 1.0,
    "ic_weight": 1.0,
    "n_collocation": 128,
    # 贝叶斯 (alpha-VI, alpha=1)
    "alpha": 1.0,
    "mc_samples": 3,
    "eval_mc_samples": 20,
    # 输出
    "experiment_dir": "experiments/paper/exp1_baseline_comparison",
}

# 模型定义：(name, pi, bayes, branch_type)
MODELS = [
    ("vanilla_deeponet", False, False, "fnn"),
    ("pi_deeponet", True, False, "fnn"),
    ("b_deeponet", False, True, "fnn"),
    ("transformer_deeponet", False, False, "transformer"),
    ("pi_bt_deeponet", True, True, "transformer"),
]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    import sys
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))

    from src.models import (
        BayesianDeepONet,
        BayesianFNNBranch,
        BayesianFNNTrunk,
        BayesianTransformerBranch,
        DeepONet,
        FNNBranch,
        FNNTrunk,
        TransformerBranch,
    )
    from src.data.generators.antiderivative import generate_antiderivative_data
    from src.training.trainer import train_antiderivative

    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成数据
    data = generate_antiderivative_data(
        n_train=cfg["n_train"],
        n_test=cfg["n_test"],
        n_sensors=cfg["n_sensors"],
        n_points_per_sample=cfg["n_points_per_sample"],
        length_scale=cfg["length_scale"],
        seed=cfg["seed"],
    )

    exp_root = Path(cfg["experiment_dir"])
    exp_root.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_path = exp_root / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[Config] {config_path}")

    results = []

    for name, use_pi, use_bayes, branch_type in MODELS:
        print(f"\n{'='*60}\n{name}\n{'='*60}")

        if branch_type == "fnn":
            if use_bayes:
                branch = BayesianFNNBranch(
                    cfg["num_sensors"],
                    cfg["branch_hidden"],
                    cfg["output_dim"],
                    prior_sigma=cfg["prior_sigma"],
                )
                trunk = BayesianFNNTrunk(
                    cfg["coord_dim"],
                    cfg["trunk_hidden"],
                    cfg["output_dim"],
                    prior_sigma=cfg["prior_sigma"],
                )
                model = BayesianDeepONet(branch, trunk, bias=True, min_noise=1e-3)
            else:
                branch = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
                trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                model = DeepONet(branch, trunk, cfg["output_dim"], bias=True)
        else:
            if use_bayes:
                branch = BayesianTransformerBranch(
                    cfg["num_sensors"],
                    cfg["output_dim"],
                    d_model=cfg["transformer_d_model"],
                    nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"],
                    dropout=cfg["transformer_dropout"],
                    prior_sigma=cfg["prior_sigma"],
                )
                trunk = BayesianFNNTrunk(
                    cfg["coord_dim"],
                    cfg["trunk_hidden"],
                    cfg["output_dim"],
                    prior_sigma=cfg["prior_sigma"],
                )
                model = BayesianDeepONet(branch, trunk, bias=True, min_noise=1e-3)
            else:
                branch = TransformerBranch(
                    cfg["num_sensors"],
                    cfg["output_dim"],
                    d_model=cfg["transformer_d_model"],
                    nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"],
                    dropout=cfg["transformer_dropout"],
                )
                trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                model = DeepONet(branch, trunk, cfg["output_dim"], bias=True)

        n_params = count_params(model)
        log_dir = exp_root / name
        log_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        _, metrics = train_antiderivative(
            model,
            data,
            lr=cfg["lr"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            log_dir=str(log_dir),
            device=device,
            bayes_method="alpha_vi" if use_bayes else "deterministic",
            alpha=cfg["alpha"],
            mc_samples=cfg["mc_samples"],
            eval_mc_samples=cfg["eval_mc_samples"],
            pi_constraint="antiderivative" if use_pi else "none",
            pi_weight=cfg["pi_weight"] if use_pi else 0.0,
            bc_weight=cfg["bc_weight"] if use_pi else 0.0,
            ic_weight=cfg["ic_weight"] if use_pi else 0.0,
            n_collocation=cfg["n_collocation"] if use_pi else 0,
            seed=cfg["seed"],
            checkpoint_every=5,
            checkpoint_dir=str(log_dir / "checkpoints"),
        )
        elapsed = time.perf_counter() - t0

        # 保存训练历史
        hist = metrics.get("history", [])
        if hist:
            hist_path = log_dir / "training_history.json"
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
            print(f"  [History] {len(hist)} epochs -> {hist_path}")

        # 保存单模型结果
        model_result = {
            "name": name,
            "params": n_params,
            "time_s": elapsed,
            "loss": metrics["loss"],
            "rel_l2": metrics["rel_l2"],
            "test_mse": metrics["test_mse"],
        }
        with open(log_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(model_result, f, indent=2)
        results.append(model_result)

    # 保存汇总 CSV
    csv_path = exp_root / f"exp1_summary_epochs{cfg['epochs']}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("name,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['name']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[CSV] {csv_path}")

    # 打印表格
    print("\n" + "=" * 95)
    print(f"论文实验 1: Antiderivative 基准对比 (epochs={cfg['epochs']})")
    print("=" * 95)
    header = f"{'模型':<25} {'参数量':>12} {'时间(s)':>10} {'时间(min)':>10} {'loss':>12} {'rel_l2':>10} {'test_mse':>12}"
    print(header)
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<25} {r['params']:>12,} {r['time_s']:>10.1f} {r['time_s']/60:>10.2f} {r['loss']:>12.6f} {r['rel_l2']:>10.6f} {r['test_mse']:>12.6f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
