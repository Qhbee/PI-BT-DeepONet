"""一次性：读取已有确定性 PI-DeepONet 权重为预训练，再 α-VI 训练 + 经典 f 曲线图（画图脚本含 98\%/99\% MC 可信带）。

默认：`checkpoints/best.pt` → 初始化贝叶斯 → 仅训练 `pi_b_deeponet`；画图脚本默认贝叶斯 + 可信带。
无参运行：``uv run python scripts/paper/_smoke_run_plot_antiderivative.py``
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# --- 默认与 plot_exp1_antiderivative_classic_curves 对齐 ---
DEFAULT_RUN_DIR = ROOT / "experiments/paper/exp1_antiderivative_smoke"
DEFAULT_DET_SUBDIR = "pi_deeponet"
DEFAULT_BAYES_MODEL_NAME = "pi_b_deeponet"
PRETRAIN_CKPT_PREFERRED_NAME = "best.pt"
PRETRAIN_FALLBACK_EPOCH = 40
# α-VI 上限轮数；早停（rel_l2，patience=25 按 eval 间隔计）可大幅提前结束。"up to" 指硬上限而非必跑满。
# 全程在 CPU 上可能达数小时～十多小时（见 train_operator：每 batch 多次 VI 前向 + PI 配点 + 定期 200 次 MC 整测试集）。
DEFAULT_EPOCHS_BAYES = 100
# 须 ≥ eval_every：train_operator 里无提升时每次 eval 会 += eval_every（见 trainer），过小会导致「一次验证没变好」就停。
DEFAULT_EARLY_STOP_PATIENCE = 25
DEFAULT_PRETRAIN_PRIOR_SIGMA = 0.1
# train_operator：α-VI 每个 batch 内对权重采样次数，用于 ELBO，与下面「验证/画图」抽样无关。
DEFAULT_MC_SAMPLES_TRAIN = 3
# train_operator：验证集 rel_l2 时的预测平均抽样数；写入 config.json 的 eval_mc_samples（仅训练用，与 plot 脚本 DEFAULT_EVAL_MC_SAMPLES 无关）。
DEFAULT_TRAIN_EVAL_MC_SAMPLES = 200

from src.data.generators.antiderivative import generate_antiderivative_data
from src.models import (
    BayesianDeepONet,
    BayesianFNNBranch,
    BayesianFNNTrunk,
    DeepONet,
    FNNBranch,
    FNNTrunk,
    init_bayesian_fnn_from_deterministic,
    set_bayesian_prior_from_weights,
)
from src.training.trainer import train_antiderivative


def _resolve_pretrain_ckpt(
    run_dir: Path,
    *,
    pretrain_ckpt: Path | None,
    prefer_best: bool,
    pretrain_epoch: int | None,
    det_subdir: str,
) -> Path:
    if pretrain_ckpt is not None:
        p = pretrain_ckpt.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"pretrain checkpoint not found: {p}")
        return p
    cdir = run_dir / det_subdir / "checkpoints"
    if prefer_best:
        best = cdir / PRETRAIN_CKPT_PREFERRED_NAME
        if best.is_file():
            return best
        print(f"[smoke] 未找到 {best}，按回退顺序继续查找预训练权重", flush=True)
    epochs_try: list[int] = []
    if pretrain_epoch == -1:
        pass
    elif pretrain_epoch is not None and pretrain_epoch >= 0:
        epochs_try.append(pretrain_epoch)
        if pretrain_epoch != PRETRAIN_FALLBACK_EPOCH:
            epochs_try.append(PRETRAIN_FALLBACK_EPOCH)
    else:
        epochs_try.append(PRETRAIN_FALLBACK_EPOCH)
    for e in epochs_try:
        ep = cdir / f"epoch_{e}.pt"
        if ep.is_file():
            return ep
    latest = cdir / "latest.pt"
    if latest.is_file():
        print(f"[smoke] 使用 {latest}（建议写入 {PRETRAIN_CKPT_PREFERRED_NAME} 以便默认识别最优）", flush=True)
        return latest
    raise FileNotFoundError(
        f"未找到确定性预训练权重：期望 {cdir} 下有 {PRETRAIN_CKPT_PREFERRED_NAME}、epoch_*.pt 或 latest.pt；"
        f"或传入 --pretrain_ckpt。"
    )


def _load_det_state(ckpt_path: Path, device: str) -> dict:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    return blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob


def main() -> None:
    ap = argparse.ArgumentParser(description="反导数 smoke：best.pt（或回退）→ α-VI → 经典曲线图")
    ap.add_argument(
        "--run_dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"实验根目录（默认: {DEFAULT_RUN_DIR}）",
    )
    ap.add_argument(
        "--det_subdir",
        type=str,
        default=DEFAULT_DET_SUBDIR,
        help=f"确定性模型子目录（默认: {DEFAULT_DET_SUBDIR}）",
    )
    ap.add_argument(
        "--no_best_pretrain",
        action="store_true",
        help=f"不优先使用 checkpoints/{PRETRAIN_CKPT_PREFERRED_NAME}，直接在缺失时按 epoch / latest 查找",
    )
    ap.add_argument(
        "--pretrain_epoch",
        type=int,
        default=None,
        help=f"best 缺失时尝试 epoch_<n>，再尝试 epoch_{PRETRAIN_FALLBACK_EPOCH}；-1 表示不尝试任何 epoch_*，直接用 latest",
    )
    ap.add_argument(
        "--pretrain_ckpt",
        type=Path,
        default=None,
        help="显式指定确定性 .pt（最高优先级）",
    )
    ap.add_argument("--epochs_bayes", type=int, default=DEFAULT_EPOCHS_BAYES, help="α-VI 训练轮数上限")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    pre_path = _resolve_pretrain_ckpt(
        run_dir,
        pretrain_ckpt=args.pretrain_ckpt,
        prefer_best=not args.no_best_pretrain,
        pretrain_epoch=args.pretrain_epoch,
        det_subdir=args.det_subdir,
    )
    print(f"[smoke] 确定性预训练权重: {pre_path}", flush=True)

    cfg = {
        "n_train": 800,
        "n_test": 200,
        "n_sensors": 50,
        "n_points_per_sample": 50,
        "length_scale": 0.5,
        "seed": 42,
        "output_dim": 20,
        "branch_hidden": [32, 32],
        "trunk_hidden": [32, 32],
        "num_sensors": 50,
        "coord_dim": 1,
        "transformer_d_model": 32,
        "transformer_nhead": 4,
        "transformer_num_layers": 2,
        "transformer_dropout": 0.0,
        "prior_sigma": 1.0,
        "prior_sigma_pretrained": DEFAULT_PRETRAIN_PRIOR_SIGMA,
        "eval_mc_samples": DEFAULT_TRAIN_EVAL_MC_SAMPLES,
        "alpha": 1.0,
        "mc_samples": DEFAULT_MC_SAMPLES_TRAIN,
    }
    data = generate_antiderivative_data(
        n_train=cfg["n_train"],
        n_test=cfg["n_test"],
        n_sensors=cfg["n_sensors"],
        n_points_per_sample=cfg["n_points_per_sample"],
        length_scale=cfg["length_scale"],
        seed=cfg["seed"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = DEFAULT_BAYES_MODEL_NAME
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = {
        **cfg,
        "lr": 0.001,
        "batch_size": 64,
        "pretrain_checkpoint": str(pre_path),
        "epochs_bayes": args.epochs_bayes,
        "pi_weight": 0.2,
        "bc_weight": 1.0,
        "ic_weight": 1.0,
        "n_collocation": 128,
        "noise_std": 0.0,
        "noise_relative": True,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(full_cfg, f, indent=2, ensure_ascii=False)

    branch_d = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
    trunk_d = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
    det_model = DeepONet(branch_d, trunk_d, cfg["output_dim"], bias=True)
    det_model.to(device)
    det_model.load_state_dict(_load_det_state(pre_path, device), strict=True)

    b_br = BayesianFNNBranch(
        cfg["num_sensors"],
        cfg["branch_hidden"],
        cfg["output_dim"],
        prior_sigma=cfg["prior_sigma"],
    )
    b_tr = BayesianFNNTrunk(
        cfg["coord_dim"],
        cfg["trunk_hidden"],
        cfg["output_dim"],
        prior_sigma=cfg["prior_sigma"],
    )
    bayes_model = BayesianDeepONet(b_br, b_tr, bias=True, min_noise=1e-3)
    bayes_model.to(device)
    init_bayesian_fnn_from_deterministic(bayes_model, det_model, rho_init=-5.0)
    set_bayesian_prior_from_weights(bayes_model, full_cfg["prior_sigma_pretrained"])

    print(
        f"[smoke] PI-BT-DeepONet α-VI：最多 {args.epochs_bayes} epoch（早停可提前）-> {model_dir}",
        flush=True,
    )
    train_antiderivative(
        bayes_model,
        data,
        lr=full_cfg["lr"],
        epochs=args.epochs_bayes,
        batch_size=full_cfg["batch_size"],
        log_dir=str(model_dir),
        device=device,
        bayes_method="alpha_vi",
        alpha=cfg["alpha"],
        mc_samples=cfg["mc_samples"],
        eval_mc_samples=cfg["eval_mc_samples"],
        pi_constraint="antiderivative",
        pi_weight=full_cfg["pi_weight"],
        bc_weight=full_cfg["bc_weight"],
        ic_weight=full_cfg["ic_weight"],
        n_collocation=full_cfg["n_collocation"],
        seed=cfg["seed"],
        checkpoint_every=10,
        checkpoint_dir=str(ckpt_dir),
        eval_every=5,
        early_stop=True,
        early_stop_patience=DEFAULT_EARLY_STOP_PATIENCE,
        early_stop_metric="rel_l2",
    )

    plot_script = ROOT / "scripts/paper/plot_exp1_antiderivative_classic_curves.py"
    cmd = [
        sys.executable,
        str(plot_script),
        "--run_dir",
        str(run_dir),
        "--model_name",
        model_name,
        "--branch",
        "fnn",
        "--dpi",
        "150",
        "--out_dir",
        str(ROOT / "thesis" / "figures"),
        "--out_name",
        "exp1_antiderivative_classic_curves.png",
    ]
    print(f"[smoke] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print(f"[smoke] figure -> {ROOT / 'thesis' / 'figures' / 'exp1_antiderivative_classic_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
