"""
论文实验 1：Antiderivative 算例下，PI-BT-DeepONet 与基准模型对比。

探究版：有噪声 N(0,0.02²)、贝叶斯每10 epoch 评估、Transformer 参数量缩减、结果带时间戳、每模型配图。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
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
    # 噪声（探究版）
    "noise_std": 0.02,
    # 模型
    "output_dim": 20,
    "branch_hidden": [20, 20],
    "trunk_hidden": [20, 20],
    "num_sensors": 50,
    "coord_dim": 1,
    # Transformer 缩减：d_model 32→16，参数量约减半
    "transformer_d_model": 16,
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
    # 贝叶斯
    "alpha": 1.0,
    "mc_samples": 3,
    "eval_mc_samples": 20,
    "eval_every_bayes": 10,
    "eval_every_det": 5,
    # 输出
    "experiment_dir": "experiments/paper/exp1_baseline_comparison",
}

MODELS = [
    ("vanilla_deeponet", False, False, "fnn"),
    ("pi_deeponet", True, False, "fnn"),
    ("b_deeponet", False, True, "fnn"),
    ("transformer_deeponet", False, False, "transformer"),
    ("pi_bt_deeponet", True, True, "transformer"),
]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _add_noise(data: dict, noise_std: float, seed: int | None) -> None:
    """In-place add N(0, noise_std²) to s_train and s_test."""
    if noise_std <= 0:
        return
    rng = np.random.default_rng(seed)
    data["s_train"] = (data["s_train"].astype(np.float64) + rng.normal(0, noise_std, data["s_train"].shape)).astype(np.float32)
    data["s_test"] = (data["s_test"].astype(np.float64) + rng.normal(0, noise_std, data["s_test"].shape)).astype(np.float32)


def _plot_model_curves(hist: list[dict], out_path: Path, model_name: str) -> None:
    """Plot loss, test_mse, rel_l2 vs epoch (3 subplots horizontal)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [Skip] matplotlib not found, no plot for {model_name}")
        return

    epochs = [h["epoch"] for h in hist]
    loss = [h.get("loss") for h in hist]
    # forward-fill rel_l2 and test_mse when not evaluated
    rel_l2, test_mse = [], []
    last_r, last_m = None, None
    for h in hist:
        last_r = h.get("rel_l2") if h.get("rel_l2") is not None else last_r
        last_m = h.get("test_mse") if h.get("test_mse") is not None else last_m
        rel_l2.append(last_r)
        test_mse.append(last_m)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(model_name, fontsize=12)

    axes[0].plot(epochs, loss, "b-", lw=1)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    if any(m is not None for m in test_mse):
        axes[1].plot(epochs, test_mse, "g-", lw=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test MSE")
    axes[1].set_title("Test MSE")
    axes[1].grid(True, alpha=0.3)

    if any(r is not None for r in rel_l2):
        axes[2].plot(epochs, rel_l2, "r-", lw=1)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Rel L2")
    axes[2].set_title("Rel L2")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {out_path}")


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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = generate_antiderivative_data(
        n_train=cfg["n_train"],
        n_test=cfg["n_test"],
        n_sensors=cfg["n_sensors"],
        n_points_per_sample=cfg["n_points_per_sample"],
        length_scale=cfg["length_scale"],
        seed=cfg["seed"],
    )
    _add_noise(data, cfg.get("noise_std", 0), cfg["seed"] + 1)

    exp_root = Path(cfg["experiment_dir"]) / f"run_{ts}"
    exp_root.mkdir(parents=True, exist_ok=True)
    figures_dir = exp_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

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

        eval_every = cfg["eval_every_bayes"] if use_bayes else cfg["eval_every_det"]

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
            eval_every=eval_every,
        )
        elapsed = time.perf_counter() - t0

        hist = metrics.get("history", [])
        hist_path = log_dir / "training_history.json"
        if hist:
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
            print(f"  [History] {len(hist)} epochs -> {hist_path}")
            _plot_model_curves(hist, figures_dir / f"{name}.png", name)

        model_result = {
            "name": name,
            "params": n_params,
            "time_s": elapsed,
            "loss": metrics["loss"],
            "rel_l2": metrics["rel_l2"],
            "test_mse": metrics["test_mse"],
        }
        result_path = log_dir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(model_result, f, indent=2)
        results.append(model_result)

    csv_path = exp_root / f"exp1_summary_epochs{cfg['epochs']}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("name,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['name']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[CSV] {csv_path}")

    print("\n" + "=" * 95)
    print(f"论文实验 1 (noise_std={cfg.get('noise_std',0)}, run={ts})")
    print("=" * 95)
    header = f"{'模型':<25} {'参数量':>12} {'时间(s)':>10} {'时间(min)':>10} {'loss':>12} {'rel_l2':>10} {'test_mse':>12}"
    print(header)
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<25} {r['params']:>12,} {r['time_s']:>10.1f} {r['time_s']/60:>10.2f} {r['loss']:>12.6f} {r['rel_l2']:>10.6f} {r['test_mse']:>12.6f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
