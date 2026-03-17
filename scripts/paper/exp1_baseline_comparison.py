"""
论文实验 1：Antiderivative 算例下，PI-BT-DeepONet 与基准模型对比。

探究版：有噪声 N(0,0.02²)、贝叶斯每10 epoch 评估、Transformer 参数量缩减、结果带时间戳、每模型配图。
"""

from __future__ import annotations

import json
import shutil
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
    "epochs": 60,
    "batch_size": 64,
    "lr": 0.001,
    "pi_weight": 0.1,
    "bc_weight": 1.0,
    "ic_weight": 1.0,
    "n_collocation": 128,
    # 贝叶斯
    "alpha": 1.0,
    "mc_samples": 3,
    "b_deeponet_pretrain": True,  # b_deeponet 是否先用 vanilla 预训练再贝叶斯微调
    "b_deeponet_pretrain_ratio": 2 / 3,  # 预训练占比；2/3 → 40 预训练 + 20 贝叶斯
    "pi_bt_deeponet_pretrain": True,  # pi_bt_deeponet 是否先用 transformer 预训练
    "pi_bt_deeponet_pretrain_ratio": 5 / 6,  # 5/6 → 25 预训练 + 5 贝叶斯
    "prior_sigma_pretrained": 0.1,  # 预训练后先验 N(μ_pretrained, σ)，σ 小则 KL 更强保留预训练解
    "eval_mc_samples": 20,
    "eval_every_bayes": 10,
    "eval_every_det": 5,
    # 输出
    "experiment_dir": "experiments/paper/exp1_baseline_comparison",
}

MODELS = [
    ("vanilla_deeponet", False, False, "fnn"),
    # ("pi_deeponet", True, False, "fnn"),
    ("b_deeponet", False, True, "fnn"),
    # ("transformer_deeponet", False, False, "transformer"),
    # ("pi_bt_deeponet", True, True, "transformer"),
]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _get_latest_prev_run(exp_dir: Path, current_run: Path) -> Path | None:
    """返回最近一次（时间戳最大）的 run 目录，排除当前 run。"""
    if not exp_dir.exists():
        return None
    runs = sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    runs = [r for r in runs if r != current_run]
    return runs[-1] if runs else None


def _load_pretrain_hist_from_tb(pretrain_dir: Path, max_epochs: int) -> list[dict]:
    """从 tensorboard events 读取预训练历史（当 training_history.json 不存在时备选）。"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return []
    if not pretrain_dir.exists():
        return []
    try:
        ea = event_accumulator.EventAccumulator(str(pretrain_dir))
        ea.Reload()
    except Exception:
        return []
    tags = ea.Tags().get("scalars", [])
    loss_data = {e.step: e.value for e in ea.Scalars("loss/train")} if "loss/train" in tags else {}
    rel_data = {e.step: e.value for e in ea.Scalars("metric/rel_l2")} if "metric/rel_l2" in tags else {}
    mse_data = {e.step: e.value for e in ea.Scalars("loss/test_mse")} if "loss/test_mse" in tags else {}
    if not loss_data:
        return []
    steps = sorted(s for s in loss_data if s < max_epochs)
    return [
        {
            "epoch": s + 1,
            "loss": loss_data[s],
            "rel_l2": rel_data.get(s),
            "test_mse": mse_data.get(s),
        }
        for s in steps
    ]


def _get_latest_run_with_file(exp_dir: Path, current_run: Path, rel_path: str) -> Path | None:
    """返回最近一次包含 rel_path 的 run 目录（从新到旧找第一个有的）。"""
    if not exp_dir.exists():
        return None
    runs = sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    runs = [r for r in runs if r != current_run]
    for r in reversed(runs):
        if (r / rel_path).exists():
            return r
    return None


def _try_copy_deterministic_from_prev_run(
    exp_dir: Path, exp_root: Path, name: str, figures_dir: Path, target_epochs: int
) -> tuple[bool, Path | None]:
    """
    确定性模型：拷贝上次 run 的结果到本次。
    返回 (skipped, resume_from):
    - skipped=True: 已有完整结果（epoch_{target_epochs}.pt），跳过训练
    - resume_from 非空: 有部分结果，从该 checkpoint 继续训练到 target_epochs
    - 否则: 无可用结果，需从头训练
    """
    dst = exp_root / name
    ckpt_dir = dst / "checkpoints"
    target_ckpt = ckpt_dir / f"epoch_{target_epochs}.pt"

    if target_ckpt.exists():
        return True, None

    prev_run = _get_latest_run_with_file(exp_dir, exp_root, f"{name}/checkpoints")
    if prev_run is None:
        return False, None

    src = prev_run / name
    if not src.is_dir():
        return False, None

    shutil.copytree(src, dst, dirs_exist_ok=True)
    prev_fig = prev_run / "figures" / f"{name}.png"
    if prev_fig.exists():
        shutil.copy2(prev_fig, figures_dir / f"{name}.png")

    if target_ckpt.exists():
        print(f"  [Reuse] 拷贝上次 run 的 {name} 结果到本次目录，跳过训练")
        return True, None

    max_epoch = 0
    for f in ckpt_dir.glob("epoch_*.pt"):
        try:
            e = int(f.stem.split("_")[1])
            max_epoch = max(max_epoch, e)
        except (ValueError, IndexError):
            pass
    if max_epoch > 0 and max_epoch < target_epochs:
        resume_ckpt = ckpt_dir / f"epoch_{max_epoch}.pt"
        if resume_ckpt.exists():
            print(f"  [Reuse] 拷贝上次 run 的 {name}，从 epoch {max_epoch} 继续训练到 {target_epochs}")
            return False, resume_ckpt

    return False, None


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
        init_bayesian_fnn_from_deterministic,
        init_bayesian_transformer_from_deterministic,
        set_bayesian_prior_from_weights,
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

    exp_dir = Path(cfg["experiment_dir"])
    for name, use_pi, use_bayes, branch_type in MODELS:
        print(f"\n{'='*60}\n{name}\n{'='*60}")

        # 确定性模型：拷贝上次结果，完整则跳过；部分则从 checkpoint 继续训练
        # 贝叶斯模型：不拷贝完整结果，只复用确定性预训练部分
        resume_from: Path | None = None
        resume_prev_hist: list[dict] = []
        if not use_bayes:
            skipped, resume_from = _try_copy_deterministic_from_prev_run(
                exp_dir, exp_root, name, figures_dir, cfg["epochs"]
            )
            if skipped:
                with open(exp_root / name / "result.json", encoding="utf-8") as f:
                    model_result = json.load(f)
                results.append(model_result)
                continue
            if resume_from is not None:
                prev_hist_path = exp_root / name / "training_history.json"
                if prev_hist_path.exists():
                    with open(prev_hist_path, encoding="utf-8") as f:
                        resume_prev_hist = json.load(f)

        pretrain_model = None
        pretrain_hist: list[dict] = []
        pretrain_epochs = 0
        bayes_epochs = cfg["epochs"]  # 无预训练时用满 epochs
        # b_deeponet: 1) 优先复用上次 run 的 b_deeponet 预训练；2) 否则用本次 vanilla 的 epoch 2/3；3) 否则自训
        if branch_type == "fnn" and use_bayes and cfg.get("b_deeponet_pretrain", False):
            ratio = cfg.get("b_deeponet_pretrain_ratio", 0.5)
            pretrain_epochs = int(cfg["epochs"] * ratio)
            bayes_epochs = cfg["epochs"] - pretrain_epochs
            pretrain_ckpt = None
            from_prev_run = False
            # 1) 最近一次有 b_deeponet 预训练的 run
            prev_b = _get_latest_run_with_file(exp_dir, exp_root, f"{name}/_pretrain/final_model.pt")
            if prev_b is not None:
                pretrain_ckpt = prev_b / name / "_pretrain" / "final_model.pt"
                from_prev_run = True
            # 2) 本次 vanilla_deeponet 的 checkpoint（vanilla 需先跑）
            if pretrain_ckpt is None:
                vanilla_path = exp_root / "vanilla_deeponet" / "checkpoints" / f"epoch_{pretrain_epochs}.pt"
                if vanilla_path.exists():
                    pretrain_ckpt = vanilla_path
            # 3) 最近一次有 vanilla_deeponet 的 run（本次未训练 vanilla 时可用）
            if pretrain_ckpt is None:
                prev_v = _get_latest_run_with_file(
                    exp_dir, exp_root, f"vanilla_deeponet/checkpoints/epoch_{pretrain_epochs}.pt"
                )
                if prev_v is not None:
                    pretrain_ckpt = prev_v / "vanilla_deeponet" / "checkpoints" / f"epoch_{pretrain_epochs}.pt"
                    from_prev_run = True
            if pretrain_ckpt is not None:
                src = "上次" if from_prev_run else "本次"
                print(f"  [Pretrain] 复用{src} run 的 vanilla 预训练 (epoch_{pretrain_epochs})...")
                vanilla_branch = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
                vanilla_trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                pretrain_model = DeepONet(vanilla_branch, vanilla_trunk, cfg["output_dim"], bias=True)
                ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
                sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                pretrain_model.load_state_dict(sd, strict=True)
                pretrain_hist = []
                if from_prev_run and prev_b is not None:
                    prev_hist = prev_b / name / "_pretrain" / "training_history.json"
                    prev_tb = prev_b / name / "_pretrain"
                    if prev_hist.exists():
                        with open(prev_hist, encoding="utf-8") as f:
                            pretrain_hist = [h for h in json.load(f) if h.get("epoch", 0) <= pretrain_epochs]
                    elif prev_tb.exists():
                        pretrain_hist = _load_pretrain_hist_from_tb(prev_tb, pretrain_epochs)
                else:
                    vanilla_hist = exp_root / "vanilla_deeponet" / "training_history.json"
                    if vanilla_hist.exists():
                        with open(vanilla_hist, encoding="utf-8") as f:
                            pretrain_hist = [h for h in json.load(f) if h.get("epoch", 0) <= pretrain_epochs]
            else:
                print(f"  [Pretrain] vanilla 预训练 {pretrain_epochs} epochs，贝叶斯微调 {bayes_epochs} epochs（总 {cfg['epochs']}）...")
                vanilla_branch = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
                vanilla_trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                pretrain_model = DeepONet(vanilla_branch, vanilla_trunk, cfg["output_dim"], bias=True)
                pretrain_log = exp_root / name / "_pretrain"
                pretrain_log.mkdir(parents=True, exist_ok=True)
                _, pretrain_metrics = train_antiderivative(
                    pretrain_model,
                    data,
                    lr=cfg["lr"],
                    epochs=pretrain_epochs,
                    batch_size=cfg["batch_size"],
                    log_dir=str(pretrain_log),
                    device=device,
                    bayes_method="deterministic",
                    pi_constraint="none",
                    pi_weight=0.0,
                    bc_weight=0.0,
                    ic_weight=0.0,
                    n_collocation=0,
                    seed=cfg["seed"],
                    checkpoint_every=0,
                    eval_every=cfg["eval_every_det"],
                )
                torch.save(pretrain_model.state_dict(), pretrain_log / "final_model.pt")
                pretrain_hist = pretrain_metrics.get("history", [])
                if pretrain_hist:
                    with open(pretrain_log / "training_history.json", "w", encoding="utf-8") as f:
                        json.dump(pretrain_hist, f, indent=2)
            print("  [Pretrain] 完成，初始化 b_deeponet...")
        # pi_bt_deeponet: 用 transformer+PI 预训练（带 PI 约束），或复用上次 run 的预训练
        elif branch_type == "transformer" and use_bayes and cfg.get("pi_bt_deeponet_pretrain", False):
            ratio = cfg.get("pi_bt_deeponet_pretrain_ratio", 5 / 6)
            pretrain_epochs = int(cfg["epochs"] * ratio)
            bayes_epochs = cfg["epochs"] - pretrain_epochs
            pretrain_ckpt = exp_root / name / "_pretrain" / "final_model.pt"
            from_prev_run = False
            if not pretrain_ckpt.exists():
                prev_run = _get_latest_run_with_file(exp_dir, exp_root, f"{name}/_pretrain/final_model.pt")
                if prev_run is not None:
                    pretrain_ckpt = prev_run / name / "_pretrain" / "final_model.pt"
                    from_prev_run = True
            if pretrain_ckpt.exists():
                src = "上次" if from_prev_run else "本次"
                print(f"  [Pretrain] 复用{src} run 的 transformer+PI 预训练...")
                trans_branch = TransformerBranch(
                    cfg["num_sensors"],
                    cfg["output_dim"],
                    d_model=cfg["transformer_d_model"],
                    nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"],
                    dropout=cfg["transformer_dropout"],
                )
                trans_trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                pretrain_model = DeepONet(trans_branch, trans_trunk, cfg["output_dim"], bias=True)
                ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
                pretrain_model.load_state_dict(ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt.get("model_state_dict", ckpt), strict=True)
                pretrain_hist = []
                if from_prev_run and prev_run is not None:
                    prev_hist = prev_run / name / "_pretrain" / "training_history.json"
                    prev_tb = prev_run / name / "_pretrain"
                    if prev_hist.exists():
                        with open(prev_hist, encoding="utf-8") as f:
                            pretrain_hist = [h for h in json.load(f) if h.get("epoch", 0) <= pretrain_epochs]
                    elif prev_tb.exists():
                        pretrain_hist = _load_pretrain_hist_from_tb(prev_tb, pretrain_epochs)
            else:
                print(f"  [Pretrain] transformer+PI 预训练 {pretrain_epochs} epochs，贝叶斯微调 {bayes_epochs} epochs（总 {cfg['epochs']}）...")
                trans_branch = TransformerBranch(
                    cfg["num_sensors"],
                    cfg["output_dim"],
                    d_model=cfg["transformer_d_model"],
                    nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"],
                    dropout=cfg["transformer_dropout"],
                )
                trans_trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                pretrain_model = DeepONet(trans_branch, trans_trunk, cfg["output_dim"], bias=True)
                pretrain_log = exp_root / name / "_pretrain"
                pretrain_log.mkdir(parents=True, exist_ok=True)
                _, pretrain_metrics = train_antiderivative(
                    pretrain_model,
                    data,
                    lr=cfg["lr"],
                    epochs=pretrain_epochs,
                    batch_size=cfg["batch_size"],
                    log_dir=str(pretrain_log),
                    device=device,
                    bayes_method="deterministic",
                    pi_constraint="antiderivative",
                    pi_weight=cfg["pi_weight"],
                    bc_weight=cfg["bc_weight"],
                    ic_weight=cfg["ic_weight"],
                    n_collocation=cfg["n_collocation"],
                    seed=cfg["seed"],
                    checkpoint_every=0,
                    eval_every=cfg["eval_every_det"],
                )
                torch.save(pretrain_model.state_dict(), pretrain_log / "final_model.pt")  # 供下次 run 复用
                pretrain_hist = pretrain_metrics.get("history", [])
                if pretrain_hist:
                    with open(pretrain_log / "training_history.json", "w", encoding="utf-8") as f:
                        json.dump(pretrain_hist, f, indent=2)
            print("  [Pretrain] 完成，初始化 pi_bt_deeponet...")

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
                if pretrain_model is not None:
                    model.to(device)
                    pretrain_model.to(device)
                    init_bayesian_fnn_from_deterministic(model, pretrain_model, rho_init=-5.0)
                    set_bayesian_prior_from_weights(model, cfg.get("prior_sigma_pretrained"))
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
                if pretrain_model is not None:
                    model.to(device)
                    pretrain_model.to(device)
                    init_bayesian_transformer_from_deterministic(model, pretrain_model, rho_init=-5.0)
                    set_bayesian_prior_from_weights(model, cfg.get("prior_sigma_pretrained"))
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
            epochs=bayes_epochs if (use_bayes and pretrain_model is not None) else cfg["epochs"],
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
            resume_from=str(resume_from) if resume_from is not None else None,
        )
        elapsed = time.perf_counter() - t0

        hist = metrics.get("history", [])
        if use_bayes and pretrain_hist and pretrain_epochs > 0:
            hist = pretrain_hist + [{**h, "epoch": h["epoch"] + pretrain_epochs} for h in hist]
        elif resume_prev_hist:
            hist = resume_prev_hist + hist
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
