"""一次性：FNN + 反导数 PI 小训练 + 经典 f 曲线图（无现成 checkpoint 时用于验证）。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.generators.antiderivative import generate_antiderivative_data
from src.models import DeepONet, FNNBranch, FNNTrunk
from src.training.trainer import train_antiderivative


def main() -> None:
    cfg = {
        "n_train": 200,
        "n_test": 50,
        "n_sensors": 50,
        "n_points_per_sample": 10,
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
        "eval_mc_samples": 20,
        "alpha": 1.0,
        "mc_samples": 3,
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
    branch = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
    trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
    model = DeepONet(branch, trunk, cfg["output_dim"], bias=True)

    run_dir = ROOT / "experiments/paper/exp1_antiderivative_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = "pi_deeponet"
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True)
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    full_cfg = {
        **cfg,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 50,
        "pi_weight": 0.1,
        "bc_weight": 1.0,
        "ic_weight": 1.0,
        "n_collocation": 128,
        "noise_std": 0.0,
        "noise_relative": True,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(full_cfg, f, indent=2, ensure_ascii=False)

    print(
        f"[smoke] PI-DeepONet antiderivative (FNN), n_collocation={full_cfg['n_collocation']}, "
        f"{full_cfg['epochs']} epochs on {device} -> {model_dir}",
        flush=True,
    )
    train_antiderivative(
        model,
        data,
        lr=full_cfg["lr"],
        epochs=full_cfg["epochs"],
        batch_size=full_cfg["batch_size"],
        log_dir=str(model_dir),
        device=device,
        bayes_method="deterministic",
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
        early_stop_patience=20,
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
