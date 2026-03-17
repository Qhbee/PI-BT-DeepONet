"""
论文实验 2：2D Poisson 算例下，PI-BT-DeepONet 与基准模型对比。

-∇²p = f on [0,1]², p=0 on boundary
支持 query_sampling: "uniform"（均匀网格）或 "random"（随机采样）
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
# 参数说明：
# ---------- 数据 ----------
#   n_train: 训练样本数。每个样本 = 一个不同的右端项 f(x,y)，即算子实例数。
#   n_test: 测试样本数。
#   nx, ny: 源项 f 的传感器网格分辨率。num_sensors = nx*ny，branch 输入维度。
#   n_points_per_sample: 每个样本的 query 点数。uniform 时取 sqrt 取整（如 64→8×8）。
#   max_mode: Fourier 最大模数。f=Σ a_mn sin(mπx)sin(nπy)，m,n∈[1,max_mode]，总模数=max_mode²。
#             越大问题越难（高频多），越小越简单。建议 2~8。
#   length_scale: Fourier 系数 GRF 采样的长度尺度。
#   seed: 随机种子。
#   query_sampling: "uniform"=规则网格；"random"=在 [0,1]² 随机采样。
# ---------- 噪声 ----------
#   noise_std: 噪声强度。0=无噪声。
#   noise_relative: True 时 σ=noise_std*std(s)，如 0.02 表示 2% 信噪比；False 时 σ=noise_std 绝对。
#   只加在 s_train，s_test 保持干净用于评估。
# ---------- 模型（FNN） ----------
#   output_dim: DeepONet 输出秩。需与 max_mode² 匹配，越大表达能力越强。
#   branch_hidden: branch 隐藏层维度。
#   trunk_hidden: trunk 隐藏层维度。
#   coord_dim: 坐标维度，2D Poisson 为 2 (x,y)。
# ---------- 模型（Transformer） ----------
#   transformer_d_model, nhead, num_layers, dropout: Transformer 结构参数。
#   prior_sigma: 贝叶斯先验 σ。
# ---------- 训练 ----------
#   epochs: 轮数上限。早停会提前结束。
#   early_stop: 是否早停（基于 test rel_l2）。
#   early_stop_patience: 连续 N epoch 无提升则停止（按 eval_every 间隔累计）。
#   batch_size, lr: 常规训练参数。
#   pi_weight: PDE 残差 -∇²p-f=0 的权重。
#   bc_weight: 边界条件 p=0 的权重。
#   ic_weight: Poisson 无初值，保持 0。
#   n_collocation: PI 配点数。
# ---------- 贝叶斯 ----------
#   alpha: α-divergence 的 α，1 对应 KL。
#   mc_samples: 每步蒙特卡洛采样数。
#   b_deeponet_pretrain: b_deeponet 是否先用 vanilla 预训练再贝叶斯微调。
#   b_deeponet_pretrain_ratio: 预训练占比。
#   pi_bt_deeponet_pretrain: pi_bt_deeponet 是否先用 transformer+PI 预训练。
#   pi_bt_deeponet_pretrain_ratio: 同上。
#   prior_sigma_pretrained: 预训练后先验 σ。
#   eval_mc_samples: 评估时 MC 采样数。
#   eval_every_bayes / eval_every_det: 评估频率。
# ---------- 输出 ----------
#   experiment_dir: 实验输出目录。
#   reuse_prev_run: 改 max_mode/output_dim/nx/ny 等后必须 False。
# =============================================================================
CONFIG = {
    # 数据
    "n_train": 800,
    "n_test": 200,
    "nx": 10,
    "ny": 10,
    "n_points_per_sample": 25,
    "max_mode": 2,
    "length_scale": 2.0,
    "seed": 42,
    "query_sampling": "uniform",
    # 噪声（只加在训练集）
    "noise_std": 0.02,
    "noise_relative": True,  # True=相对噪声(推荐)，False=绝对噪声
    # 模型
    "output_dim": 20,
    "branch_hidden": [64, 64],
    "trunk_hidden": [64, 64],
    "coord_dim": 2,
    # Transformer
    "transformer_d_model": 32,
    "transformer_nhead": 8,
    "transformer_num_layers": 2,
    "transformer_dropout": 0.1,
    "prior_sigma": 1.0,
    # 训练
    "epochs": 200,  # 轮数上限，早停会提前结束
    "early_stop": True,
    "early_stop_patience": 30,  # 连续 30 epoch 无提升则停（按 eval_every 间隔累计）
    "batch_size": 64,
    "lr": 0.001,
    "pi_weight": 0.1,
    "bc_weight": 1.0,
    "ic_weight": 0.0,
    "n_collocation": 64,
    # 贝叶斯
    "alpha": 1.0,
    "mc_samples": 3,
    "b_deeponet_pretrain": True,
    "b_deeponet_pretrain_ratio": 2 / 3,
    "pi_bt_deeponet_pretrain": True,
    "pi_bt_deeponet_pretrain_ratio": 5 / 6,
    "prior_sigma_pretrained": 0.1,
    "eval_mc_samples": 20,
    "eval_every_bayes": 10,
    "eval_every_det": 5,
    # 输出
    "experiment_dir": "experiments/paper/exp2_poisson_2d",
    # 复用开关：改 max_mode/output_dim/nx/ny 等系数后必须设为 False，否则复用旧结果会出错
    "reuse_prev_run": False,
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
    if not exp_dir.exists():
        return None
    runs = sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    runs = [r for r in runs if r != current_run]
    return runs[-1] if runs else None


def _load_pretrain_hist_from_tb(pretrain_dir: Path, max_epochs: int) -> list[dict]:
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
        {"epoch": s + 1, "loss": loss_data[s], "rel_l2": rel_data.get(s), "test_mse": mse_data.get(s)}
        for s in steps
    ]


def _get_latest_run_with_file(exp_dir: Path, current_run: Path, rel_path: str) -> Path | None:
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


def _add_noise(
    data: dict,
    noise_std: float,
    seed: int | None,
    train_only: bool = True,
    relative: bool = False,
) -> None:
    """对目标添加 N(0, σ²) 噪声。
    train_only=True：只加在 s_train，s_test 保持干净用于评估（推荐）。
    relative=True：σ = noise_std * std(s_train)，即噪声为信号标准差的倍数。
    relative=False：σ = noise_std，绝对噪声。2D Poisson 的 p 通常较小，0.02 可能过大。
    """
    if noise_std <= 0:
        return
    rng = np.random.default_rng(seed)
    s = data["s_train"].astype(np.float64)
    sigma = (noise_std * np.std(s)) if relative else noise_std
    data["s_train"] = (s + rng.normal(0, sigma, s.shape)).astype(np.float32)
    if not train_only:
        data["s_test"] = (data["s_test"].astype(np.float64) + rng.normal(0, sigma, data["s_test"].shape)).astype(np.float32)


def _plot_model_curves(hist: list[dict], out_path: Path, model_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [Skip] matplotlib not found, no plot for {model_name}")
        return
    epochs = [h["epoch"] for h in hist]
    loss = [h.get("loss") for h in hist]
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
    from src.data.generators.poisson_2d import generate_poisson_2d_data
    from src.training.trainer import train_poisson_2d

    cfg = CONFIG
    num_sensors = cfg["nx"] * cfg["ny"]
    cfg["num_sensors"] = num_sensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = generate_poisson_2d_data(
        n_train=cfg["n_train"],
        n_test=cfg["n_test"],
        nx=cfg["nx"],
        ny=cfg["ny"],
        n_points_per_sample=cfg["n_points_per_sample"],
        max_mode=cfg["max_mode"],
        length_scale=cfg["length_scale"],
        seed=cfg["seed"],
        query_sampling=cfg["query_sampling"],
    )
    _add_noise(data, cfg.get("noise_std", 0), cfg["seed"] + 1, relative=cfg.get("noise_relative", True))

    exp_root = Path(cfg["experiment_dir"]) / f"run_{ts}"
    exp_root.mkdir(parents=True, exist_ok=True)
    figures_dir = exp_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_root / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[Config] {config_path} (query_sampling={cfg['query_sampling']})")

    results = []
    exp_dir = Path(cfg["experiment_dir"])

    for name, use_pi, use_bayes, branch_type in MODELS:
        print(f"\n{'='*60}\n{name}\n{'='*60}")

        resume_from: Path | None = None
        resume_prev_hist: list[dict] = []
        if not use_bayes:
            if cfg.get("reuse_prev_run", False):
                skipped, resume_from = _try_copy_deterministic_from_prev_run(
                    exp_dir, exp_root, name, figures_dir, cfg["epochs"]
                )
            else:
                skipped, resume_from = False, None
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
        bayes_epochs = cfg["epochs"]

        if branch_type == "fnn" and use_bayes and cfg.get("b_deeponet_pretrain", False):
            ratio = cfg.get("b_deeponet_pretrain_ratio", 0.5)
            pretrain_epochs = int(cfg["epochs"] * ratio)
            bayes_epochs = cfg["epochs"] - pretrain_epochs
            pretrain_ckpt = None
            from_prev_run = False
            if cfg.get("reuse_prev_run", False):
                prev_b = _get_latest_run_with_file(exp_dir, exp_root, f"{name}/_pretrain/final_model.pt")
            else:
                prev_b = None
            if prev_b is not None:
                pretrain_ckpt = prev_b / name / "_pretrain" / "final_model.pt"
                from_prev_run = True
            if pretrain_ckpt is None:
                vanilla_path = exp_root / "vanilla_deeponet" / "checkpoints" / f"epoch_{pretrain_epochs}.pt"
                if vanilla_path.exists():
                    pretrain_ckpt = vanilla_path
            if pretrain_ckpt is None and cfg.get("reuse_prev_run", False):
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
                _, pretrain_metrics = train_poisson_2d(
                    pretrain_model, data,
                    lr=cfg["lr"], epochs=pretrain_epochs, batch_size=cfg["batch_size"],
                    log_dir=str(pretrain_log), device=device, bayes_method="deterministic",
                    pi_constraint="none", pi_weight=0.0, bc_weight=0.0, ic_weight=0.0,
                    n_collocation=0, seed=cfg["seed"], checkpoint_every=0, eval_every=cfg["eval_every_det"],
                )
                torch.save(pretrain_model.state_dict(), pretrain_log / "final_model.pt")
                pretrain_hist = pretrain_metrics.get("history", [])
                if pretrain_hist:
                    with open(pretrain_log / "training_history.json", "w", encoding="utf-8") as f:
                        json.dump(pretrain_hist, f, indent=2)
            print("  [Pretrain] 完成，初始化 b_deeponet...")

        elif branch_type == "transformer" and use_bayes and cfg.get("pi_bt_deeponet_pretrain", False):
            ratio = cfg.get("pi_bt_deeponet_pretrain_ratio", 5 / 6)
            pretrain_epochs = int(cfg["epochs"] * ratio)
            bayes_epochs = cfg["epochs"] - pretrain_epochs
            pretrain_ckpt = exp_root / name / "_pretrain" / "final_model.pt"
            from_prev_run = False
            prev_run = None
            if not pretrain_ckpt.exists() and cfg.get("reuse_prev_run", False):
                prev_run = _get_latest_run_with_file(exp_dir, exp_root, f"{name}/_pretrain/final_model.pt")
                if prev_run is not None:
                    pretrain_ckpt = prev_run / name / "_pretrain" / "final_model.pt"
                    from_prev_run = True
            if pretrain_ckpt.exists():
                src = "上次" if from_prev_run else "本次"
                print(f"  [Pretrain] 复用{src} run 的 transformer+PI 预训练...")
                trans_branch = TransformerBranch(
                    cfg["num_sensors"], cfg["output_dim"],
                    d_model=cfg["transformer_d_model"], nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"], dropout=cfg["transformer_dropout"],
                )
                trans_trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                pretrain_model = DeepONet(trans_branch, trans_trunk, cfg["output_dim"], bias=True)
                ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
                pretrain_model.load_state_dict(
                    ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt
                    else ckpt.get("model_state_dict", ckpt), strict=True
                )
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
                    cfg["num_sensors"], cfg["output_dim"],
                    d_model=cfg["transformer_d_model"], nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"], dropout=cfg["transformer_dropout"],
                )
                trans_trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                pretrain_model = DeepONet(trans_branch, trans_trunk, cfg["output_dim"], bias=True)
                pretrain_log = exp_root / name / "_pretrain"
                pretrain_log.mkdir(parents=True, exist_ok=True)
                _, pretrain_metrics = train_poisson_2d(
                    pretrain_model, data,
                    lr=cfg["lr"], epochs=pretrain_epochs, batch_size=cfg["batch_size"],
                    log_dir=str(pretrain_log), device=device, bayes_method="deterministic",
                    pi_constraint="poisson_2d", pi_weight=cfg["pi_weight"],
                    bc_weight=cfg["bc_weight"], ic_weight=cfg["ic_weight"],
                    n_collocation=cfg["n_collocation"], seed=cfg["seed"],
                    checkpoint_every=0, eval_every=cfg["eval_every_det"],
                )
                torch.save(pretrain_model.state_dict(), pretrain_log / "final_model.pt")
                pretrain_hist = pretrain_metrics.get("history", [])
                if pretrain_hist:
                    with open(pretrain_log / "training_history.json", "w", encoding="utf-8") as f:
                        json.dump(pretrain_hist, f, indent=2)
            print("  [Pretrain] 完成，初始化 pi_bt_deeponet...")

        if branch_type == "fnn":
            if use_bayes:
                branch = BayesianFNNBranch(
                    cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"],
                    prior_sigma=cfg["prior_sigma"],
                )
                trunk = BayesianFNNTrunk(
                    cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"],
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
                    cfg["num_sensors"], cfg["output_dim"],
                    d_model=cfg["transformer_d_model"], nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"], dropout=cfg["transformer_dropout"],
                    prior_sigma=cfg["prior_sigma"],
                )
                trunk = BayesianFNNTrunk(
                    cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"],
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
                    cfg["num_sensors"], cfg["output_dim"],
                    d_model=cfg["transformer_d_model"], nhead=cfg["transformer_nhead"],
                    num_layers=cfg["transformer_num_layers"], dropout=cfg["transformer_dropout"],
                )
                trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
                model = DeepONet(branch, trunk, cfg["output_dim"], bias=True)

        n_params = count_params(model)
        log_dir = exp_root / name
        log_dir.mkdir(parents=True, exist_ok=True)
        eval_every = cfg["eval_every_bayes"] if use_bayes else cfg["eval_every_det"]

        t0 = time.perf_counter()
        _, metrics = train_poisson_2d(
            model, data,
            lr=cfg["lr"],
            epochs=bayes_epochs if (use_bayes and pretrain_model is not None) else cfg["epochs"],
            batch_size=cfg["batch_size"],
            log_dir=str(log_dir),
            device=device,
            bayes_method="alpha_vi" if use_bayes else "deterministic",
            alpha=cfg["alpha"],
            mc_samples=cfg["mc_samples"],
            eval_mc_samples=cfg["eval_mc_samples"],
            pi_constraint="poisson_2d" if use_pi else "none",
            pi_weight=cfg["pi_weight"] if use_pi else 0.0,
            bc_weight=cfg["bc_weight"] if use_pi else 0.0,
            ic_weight=cfg["ic_weight"],
            n_collocation=cfg["n_collocation"] if use_pi else 0,
            seed=cfg["seed"],
            checkpoint_every=5,
            checkpoint_dir=str(log_dir / "checkpoints"),
            eval_every=eval_every,
            resume_from=str(resume_from) if resume_from is not None else None,
            early_stop=cfg.get("early_stop", False),
            early_stop_patience=cfg.get("early_stop_patience", 30),
            early_stop_metric=cfg.get("early_stop_metric", "rel_l2"),
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

    csv_path = exp_root / f"exp2_summary_epochs{cfg['epochs']}_query_{cfg['query_sampling']}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("name,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['name']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[CSV] {csv_path}")

    print("\n" + "=" * 95)
    print(f"论文实验 2 - 2D Poisson (query_sampling={cfg['query_sampling']}, run={ts})")
    print("=" * 95)
    header = f"{'模型':<25} {'参数量':>12} {'时间(s)':>10} {'时间(min)':>10} {'loss':>12} {'rel_l2':>10} {'test_mse':>12}"
    print(header)
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<25} {r['params']:>12,} {r['time_s']:>10.1f} {r['time_s']/60:>10.2f} {r['loss']:>12.6f} {r['rel_l2']:>10.6f} {r['test_mse']:>12.6f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
