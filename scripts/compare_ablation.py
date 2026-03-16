"""Ablation script: branch_type x bayes_method x pi_constraint."""

from __future__ import annotations

import argparse
import inspect
import json
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
    generate_ns_beltrami_ic2field_data,
    generate_ns_beltrami_parametric_data,
    generate_ns_kovasznay_bc2field_data,
    generate_ns_kovasznay_parametric_data,
)
from src.models import (
    BayesianDeepONet,
    BayesianFNNBranch,
    BayesianFNNTrunk,
    BayesianMultiOutputDeepONet,
    BayesianTransformerBranch,
    BayesianTransformerMultiCLSBranch,
    BayesianTransformerMultiOutputBranch,
    DeepONet,
    FNNBranch,
    FNNTrunk,
    MultiOutputDeepONet,
    TransformerBranch,
    TransformerMultiCLSBranch,
    TransformerMultiOutputBranch,
)
from src.training import train_operator


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(cfg: dict, coord_dim: int, branch_type: str, bayes_method: str):
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    num_sensors = int(model_cfg.get("num_sensors", 50))
    output_dim = int(model_cfg.get("output_dim", 20))
    n_outputs = int(model_cfg.get("n_outputs", 1))
    p_group = int(model_cfg.get("p_group", output_dim))
    model_type = model_cfg.get("model_type", "deeponet")
    if n_outputs > 1 and model_type == "deeponet":
        model_type = "deeponet_multi_output"
    is_multi = model_type == "deeponet_multi_output"
    total_out = n_outputs * p_group if is_multi else output_dim
    input_channels = int(model_cfg.get("input_channels", 1))
    branch_hidden = model_cfg.get("branch_hidden", [20, 20])
    trunk_hidden = model_cfg.get("trunk_hidden", [20, 20])

    if bayes_method == "deterministic":
        if branch_type == "transformer_multicls":
            branch = TransformerMultiCLSBranch(
                num_sensors=num_sensors,
                n_outputs=n_outputs,
                p_group=p_group,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
                input_channels=input_channels,
            )
        elif branch_type == "transformer_multi_output":
            branch = TransformerMultiOutputBranch(
                num_sensors=num_sensors,
                n_outputs=n_outputs,
                p_group=p_group,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
                input_channels=input_channels,
            )
        elif branch_type == "transformer":
            branch = TransformerBranch(
                num_sensors,
                total_out,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
                input_channels=input_channels,
            )
        else:
            branch = FNNBranch(num_sensors, branch_hidden, total_out)
        trunk = FNNTrunk(coord_dim, trunk_hidden, total_out)
        model = (
            MultiOutputDeepONet(branch=branch, trunk=trunk, n_outputs=n_outputs, p_group=p_group, bias=True)
            if is_multi
            else DeepONet(branch, trunk, output_dim, bias=True)
        )
    else:
        if branch_type == "transformer_multicls":
            branch = BayesianTransformerMultiCLSBranch(
                num_sensors=num_sensors,
                n_outputs=n_outputs,
                p_group=p_group,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
                prior_sigma=model_cfg.get("prior_sigma", 1.0),
                input_channels=input_channels,
            )
        elif branch_type == "transformer_multi_output":
            branch = BayesianTransformerMultiOutputBranch(
                num_sensors=num_sensors,
                n_outputs=n_outputs,
                p_group=p_group,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
                prior_sigma=model_cfg.get("prior_sigma", 1.0),
                input_channels=input_channels,
            )
        elif branch_type == "transformer":
            branch = BayesianTransformerBranch(
                num_sensors=num_sensors,
                output_dim=total_out,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
                prior_sigma=model_cfg.get("prior_sigma", 1.0),
                input_channels=input_channels,
            )
        else:
            branch = BayesianFNNBranch(
                num_sensors=num_sensors,
                hidden_dims=branch_hidden,
                output_dim=total_out,
                prior_sigma=model_cfg.get("prior_sigma", 1.0),
            )
        trunk = BayesianFNNTrunk(
            input_dim=coord_dim,
            hidden_dims=trunk_hidden,
            output_dim=total_out,
            prior_sigma=model_cfg.get("prior_sigma", 1.0),
        )
        model = (
            BayesianMultiOutputDeepONet(
                branch=branch,
                trunk=trunk,
                n_outputs=n_outputs,
                p_group=p_group,
                bias=True,
                min_noise=model_cfg.get("min_noise", 1e-3),
            )
            if is_multi
            else BayesianDeepONet(
                branch=branch,
                trunk=trunk,
                bias=True,
                min_noise=model_cfg.get("min_noise", 1e-3),
            )
        )

    alpha = train_cfg.get("alpha", 1.0)
    return model, alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="antiderivative")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint every N epochs (0=disable)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default: from data_cfg)")
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

    generator = get_generator(case)
    if "n_sensors" in inspect.signature(generator).parameters:
        data_cfg.setdefault("n_sensors", int(model_cfg.get("num_sensors", 50)))
    data = generator(**data_cfg)
    y_train = data["y_train"]
    coord_dim = y_train.shape[-1] if y_train.ndim == 3 else 1
    s_train = data["s_train"]
    n_outputs = s_train.shape[-1] if s_train.ndim == 3 else 1
    model_cfg["n_outputs"] = int(n_outputs)
    u_train = data["u_train"]
    if u_train.ndim == 3:
        model_cfg["input_channels"] = int(u_train.shape[-1])
    elif case.startswith("ns_") and u_train.ndim == 2 and u_train.shape[1] <= 4:
        model_cfg["input_channels"] = int(u_train.shape[1])

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
            ns_nu=physics_cfg.get("ns_nu", 1.0 / 40.0),
            ns_beltrami_nu=physics_cfg.get("ns_beltrami_nu", 1.0),
            pressure_gauge_weight=physics_cfg.get("pressure_gauge_weight", 0.0),
            seed=args.seed if args.seed is not None else data_cfg.get("seed", 123),
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume if (args.resume and tag in args.resume) else None,
        )
        elapsed = time.perf_counter() - t0

        # 保存每 epoch 的 loss/rel_l2/test_mse 历史
        hist = metrics.get("history", [])
        if hist:
            hist_path = out_dir / tag / "training_history.json"
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
            print(f"  [History] {tag}: {len(hist)} 条 -> {hist_path}")

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

    # 保存 CSV
    csv_path = out_dir / f"stage5_ablation_{case}_epochs{args.epochs}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("tag,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['tag']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[CSV] {csv_path}")

    print("\n" + "=" * 120)
    print(f"Stage 5 Ablation results ({case}, epochs={args.epochs})")
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
