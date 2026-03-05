"""Ablation script: branch_type x bayes_method x pi_constraint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_generator
from src.data.generators import (  # noqa: F401
    generate_antiderivative_data,
    generate_burgers_data,
    generate_darcy_data,
    generate_diffusion_reaction_data,
)
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
from src.training import train_operator


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(cfg: dict, coord_dim: int, branch_type: str, bayes_method: str):
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    num_sensors = int(model_cfg.get("num_sensors", 50))
    output_dim = int(model_cfg.get("output_dim", 20))
    branch_hidden = model_cfg.get("branch_hidden", [20, 20])
    trunk_hidden = model_cfg.get("trunk_hidden", [20, 20])

    if bayes_method == "deterministic":
        if branch_type == "transformer":
            branch = TransformerBranch(
                num_sensors,
                output_dim,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
            )
        else:
            branch = FNNBranch(num_sensors, branch_hidden, output_dim)
        trunk = FNNTrunk(coord_dim, trunk_hidden, output_dim)
        model = DeepONet(branch, trunk, output_dim, bias=True)
    else:
        if branch_type == "transformer":
            branch = BayesianTransformerBranch(
                num_sensors=num_sensors,
                output_dim=output_dim,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
                prior_sigma=model_cfg.get("prior_sigma", 1.0),
            )
        else:
            branch = BayesianFNNBranch(
                num_sensors=num_sensors,
                hidden_dims=branch_hidden,
                output_dim=output_dim,
                prior_sigma=model_cfg.get("prior_sigma", 1.0),
            )
        trunk = BayesianFNNTrunk(
            input_dim=coord_dim,
            hidden_dims=trunk_hidden,
            output_dim=output_dim,
            prior_sigma=model_cfg.get("prior_sigma", 1.0),
        )
        model = BayesianDeepONet(
            branch=branch,
            trunk=trunk,
            bias=True,
            min_noise=model_cfg.get("min_noise", 1e-3),
        )

    alpha = train_cfg.get("alpha", 1.0)
    return model, alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="antiderivative")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    default_cfg = Path(f"configs/{args.case}.yaml")
    cfg_path = Path(args.config) if args.config else default_cfg
    if not cfg_path.exists():
        cfg_path = Path("configs/base.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    case = args.case
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    physics_cfg = cfg.get("physics", {})
    data_cfg = cfg.get("data", {})
    data_cfg.setdefault("n_sensors", int(model_cfg.get("num_sensors", 50)))

    generator = get_generator(case)
    data = generator(**data_cfg)
    y_train = data["y_train"]
    coord_dim = y_train.shape[-1] if y_train.ndim == 3 else 1

    combos = []
    for bayes_method in ("deterministic", "alpha_vi"):
        for branch_type in ("fnn", "transformer"):
            for pi_constraint in ("none", case):
                combos.append((bayes_method, branch_type, pi_constraint))

    results = []
    out_dir = Path(f"experiments/ablation_{case}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for bayes_method, branch_type, pi_constraint in combos:
        tag = f"{case}_{bayes_method}_{branch_type}_{pi_constraint}"
        model, alpha = build_model(cfg, coord_dim=coord_dim, branch_type=branch_type, bayes_method=bayes_method)
        n_params = count_params(model)
        t0 = time.perf_counter()
        _, metrics = train_operator(
            model=model,
            data=data,
            case=case,
            lr=train_cfg.get("lr", 0.001),
            epochs=args.epochs,
            batch_size=train_cfg.get("batch_size", 64),
            log_dir=out_dir / tag,
            device=device,
            bayes_method=bayes_method,
            alpha=alpha,
            mc_samples=train_cfg.get("mc_samples", 3),
            eval_mc_samples=train_cfg.get("eval_mc_samples", 20),
            kl_weight=train_cfg.get("kl_weight"),
            pi_constraint=pi_constraint,
            pi_weight=physics_cfg.get("pi_weight", 0.1) if pi_constraint != "none" else 0.0,
            bc_weight=physics_cfg.get("bc_weight", 0.0) if pi_constraint != "none" else 0.0,
            ic_weight=physics_cfg.get("ic_weight", 0.0) if pi_constraint != "none" else 0.0,
            n_collocation=physics_cfg.get("n_collocation", 128),
            burgers_nu=physics_cfg.get("burgers_nu", 0.01 / torch.pi),
            diffusion_D=physics_cfg.get("diffusion_D", 0.01),
            reaction_k=physics_cfg.get("reaction_k", 0.1),
        )
        elapsed = time.perf_counter() - t0
        results.append(
            {
                "tag": tag,
                "params": n_params,
                "time_s": elapsed,
                "loss": metrics["loss"],
                "rel_l2": metrics["rel_l2"],
                "test_mse": metrics["test_mse"],
            }
        )

    print("\n" + "=" * 120)
    print(f"Ablation results ({case}, epochs={args.epochs})")
    print("=" * 120)
    print(f"{'Tag':<55} {'Params':>12} {'Time(s)':>10} {'Loss':>12} {'RelL2':>12} {'TestMSE':>14}")
    print("-" * 120)
    for r in results:
        print(
            f"{r['tag']:<55} {r['params']:>12,} {r['time_s']:>10.1f} "
            f"{r['loss']:>12.6f} {r['rel_l2']:>12.6f} {r['test_mse']:>14.6f}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
