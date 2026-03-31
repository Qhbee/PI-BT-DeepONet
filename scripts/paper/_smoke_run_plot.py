"""一次性：FNN PI-DeepONet（pi_deeponet）训练 + 单算子出图（无 checkpoint 时用于验证）。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.generators.poisson_2d import generate_poisson_2d_data
from src.models import DeepONet, FNNBranch, FNNTrunk
from src.training.trainer import train_poisson_2d


def main() -> None:
    # 与 exp2 主实验同量级略小；FNN + PI，PDE 配点偏多
    cfg = {
        "n_train": 200,
        "n_test": 50,
        "nx": 10,
        "ny": 10,
        "n_points_per_sample": 25,
        "max_mode": 2,
        "length_scale": 2.0,
        "seed": 42,
        "query_sampling": "uniform",
        "output_dim": 20,
        "branch_hidden": [64, 64],
        "trunk_hidden": [64, 64],
        "coord_dim": 2,
    }
    cfg["num_sensors"] = cfg["nx"] * cfg["ny"]
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
        return_coeffs=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    branch = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
    trunk = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
    model = DeepONet(branch, trunk, cfg["output_dim"], bias=True)

    run_dir = ROOT / "experiments/paper/exp2_poisson_2d/run_pi_deeponet_smoke"
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
        "epochs": 60,
        "pi_weight": 0.1,
        "bc_weight": 1.0,
        "ic_weight": 0.0,
        "n_collocation": 256,
        "eval_mc_samples": 20,
        "noise_std": 0.0,
        "noise_relative": True,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(full_cfg, f, indent=2, ensure_ascii=False)

    print(
        f"[smoke] PI-DeepONet (FNN), n_collocation={full_cfg['n_collocation']}, "
        f"{full_cfg['epochs']} epochs on {device} -> {model_dir}",
        flush=True,
    )
    train_poisson_2d(
        model,
        data,
        lr=full_cfg["lr"],
        epochs=full_cfg["epochs"],
        batch_size=full_cfg["batch_size"],
        log_dir=str(model_dir),
        device=device,
        bayes_method="deterministic",
        pi_constraint="poisson_2d",
        pi_weight=full_cfg["pi_weight"],
        bc_weight=full_cfg["bc_weight"],
        ic_weight=0.0,
        n_collocation=full_cfg["n_collocation"],
        seed=cfg["seed"],
        checkpoint_every=10,
        checkpoint_dir=str(ckpt_dir),
        eval_every=5,
        early_stop=True,
        early_stop_patience=25,
        early_stop_metric="rel_l2",
    )

    plot_script = ROOT / "scripts/paper/plot_exp2_poisson_single_case.py"
    cmd = [
        sys.executable,
        str(plot_script),
        "--run_dir",
        str(run_dir),
        "--model_name",
        model_name,
        "--branch",
        "fnn",
        "--grid_n",
        "64",
        "--dpi",
        "150",
    ]
    print(f"[smoke] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    out = run_dir / model_name / "figures_single_case"
    print(f"[smoke] figures -> {out}", flush=True)
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
