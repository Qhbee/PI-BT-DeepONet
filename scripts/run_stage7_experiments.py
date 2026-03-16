"""Stage 7: PI extension experiments - 3 PI modes + Transformer + Bayesian.

Validates: standard_pi, hard_bc_pi, s_pinn with branch=transformer, bayes_method=alpha_vi.
main.py already supports PI extension; this script demonstrates compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generators import generate_diffusion_reaction_data
from src.physics.hard_bc import HardBCWrapper
from src.training.trainer import train_operator

from main import _build_model

# ULTRA: 极小算例，仅验证能跑通
ULTRA = os.environ.get("STAGE7_ULTRA", "1").lower() in ("1", "true", "yes")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None, help="Override epochs (default: from config)")
    p.add_argument("--fast", action="store_true", help="减小 nx/nt 加速（nx=30,nt=31）")
    p.add_argument("--faster", action="store_true", help="更小 nx/nt（nx=15,nt=16），约 30min/5epoch，可 resume 续训")
    p.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs (0=disable)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--seed", type=int, default=None, help="Override seed (default: from data_cfg)")
    args = p.parse_args()

    out_dir = Path("experiments/stage7")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("configs/diffusion_reaction_standard_pi.yaml", "standard_pi"),
        ("configs/diffusion_reaction_hard_bc.yaml", "hard_bc_pi"),
        ("configs/diffusion_reaction_s_pinn.yaml", "s_pinn"),
    ]

    branch_type = "transformer"
    bayes_method = "alpha_vi"
    coord_dim = 2
    n_outputs = 1
    input_channels = 1
    results = []

    for config_path, mode_name in configs:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            print(f"[SKIP] {config_path} not found")
            continue

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        case = cfg.get("case", "diffusion_reaction")
        base_model_cfg = cfg.get("model", {})
        physics_cfg = cfg.get("physics", {})
        train_cfg = cfg.get("training", {})
        data_cfg = cfg.get("data", {})

        if args.faster:
            data_cfg = {**data_cfg, "nx": 15, "nt": 16}
        elif args.fast:
            data_cfg = {**data_cfg, "nx": 30, "nt": 31}
        if ULTRA:
            data_cfg = {
                "n_train": 12,
                "n_test": 4,
                "n_sensors": 16,
                "nx": 12,
                "nt": 11,
                "D": 0.01,
                "k": -0.01,
                "seed": 123,
            }
            train_cfg = {**train_cfg, "epochs": args.epochs if args.epochs is not None else 2, "batch_size": 16}
            physics_cfg = {**physics_cfg, "n_collocation": 16, "reaction_k": -0.01}
            base_model_cfg = {
                **base_model_cfg,
                "num_sensors": 16,
                "output_dim": 32,
                "branch_hidden": [32, 32],
                "trunk_hidden": [32, 32],
                "transformer_d_model": 16,
                "transformer_nhead": 2,
                "transformer_num_layers": 1,
            }

        physics_mode = physics_cfg.get("physics_mode", "standard_pi")
        data = generate_diffusion_reaction_data(**data_cfg)

        model_cfg = {
            **base_model_cfg,
            "branch_type": branch_type,
            "bayes_method": bayes_method,
            "trunk_type": "fnn",
        }

        run_name = f"{mode_name}_transformer_bayes"
        print(f"\n{'='*60}")
        suf = " [ULTRA]" if ULTRA else (" [FASTER]" if args.faster else (" [FAST]" if args.fast else ""))
        print(f"Stage 7: {run_name}{suf}")
        print("=" * 60)

        model, _ = _build_model(
            model_cfg,
            train_cfg,
            coord_dim=coord_dim,
            n_outputs=n_outputs,
            input_channels=input_channels,
        )
        if physics_mode == "hard_bc_pi":
            model = HardBCWrapper(model, case)

        t0 = time.perf_counter()
        _, metrics = train_operator(
            model,
            data,
            case=case,
            lr=train_cfg.get("lr", 0.001),
            epochs=train_cfg.get("epochs", 15),
            batch_size=train_cfg.get("batch_size", 128),
            log_dir=str(out_dir / run_name),
            bayes_method=bayes_method,
            alpha=train_cfg.get("alpha", 1.0),
            mc_samples=train_cfg.get("mc_samples", 2),
            eval_mc_samples=train_cfg.get("eval_mc_samples", 5),
            pi_constraint=physics_cfg.get("pi_constraint", "none"),
            pi_weight=physics_cfg.get("pi_weight", 0.0),
            bc_weight=physics_cfg.get("bc_weight", 0.0),
            ic_weight=physics_cfg.get("ic_weight", 0.0),
            physics_mode=physics_mode,
            n_collocation=physics_cfg.get("n_collocation", 256),
            diffusion_D=physics_cfg.get("diffusion_D", 0.01),
            reaction_k=physics_cfg.get("reaction_k", -0.01),
            seed=args.seed if args.seed is not None else data_cfg.get("seed", 123),
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume if (args.resume and run_name in args.resume) else None,
        )
        elapsed = time.perf_counter() - t0

        # 保存每 epoch 的 loss/rel_l2/test_mse 历史
        hist = metrics.get("history", [])
        if hist:
            hist_path = out_dir / run_name / "training_history.json"
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
            print(f"  [History] {len(hist)} 条 -> {hist_path}")

        inner = model.model if hasattr(model, "model") else model
        n_params = count_params(inner)
        results.append({
            "mode": run_name,
            "params": n_params,
            "time_s": elapsed,
            "loss": metrics["loss"],
            "rel_l2": metrics["rel_l2"],
            "test_mse": metrics["test_mse"],
        })
        print(f"  loss={metrics['loss']:.6f} rel_l2={metrics['rel_l2']:.6f} test_mse={metrics['test_mse']:.6f} time={elapsed:.1f}s")

    epochs_val = args.epochs if args.epochs is not None else train_cfg.get("epochs", 15)
    csv_path = out_dir / f"stage7_summary_epochs{epochs_val}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("mode,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['mode']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[CSV] {len(results)} 条结果已保存到 {csv_path}")


if __name__ == "__main__":
    main()
