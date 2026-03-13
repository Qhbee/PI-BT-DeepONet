"""PI-BT-DeepONet: training entry with case dispatch."""

import argparse
import inspect
import yaml
from pathlib import Path

import torch

from src.data import get_generator
from src.data.generators import (  # noqa: F401 - force registration side effects
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
from src.physics.hard_bc import HardBCWrapper
from src.training.trainer import train_operator


def _build_model(
    model_cfg: dict,
    train_cfg: dict,
    coord_dim: int,
    n_outputs: int,
    input_channels: int,
):
    num_sensors = int(model_cfg.get("num_sensors", 100))
    output_dim = int(model_cfg.get("output_dim", 40))  # single-output p or per-group p
    p_group = int(model_cfg.get("p_group", output_dim))
    branch_type = model_cfg.get("branch_type", "fnn")
    model_type = model_cfg.get("model_type", "deeponet")
    if n_outputs > 1 and model_type == "deeponet":
        model_type = "deeponet_multi_output"
    is_multi_output = model_type == "deeponet_multi_output"
    branch_output_dim = n_outputs * p_group if is_multi_output else output_dim
    trunk_type = model_cfg.get("trunk_type", "fnn")
    bayes_method = model_cfg.get("bayes_method", model_cfg.get("uq_mode", "deterministic"))
    branch_hidden = model_cfg.get("branch_hidden", [40, 40])
    trunk_hidden = model_cfg.get("trunk_hidden", [40, 40])

    if trunk_type != "fnn":
        raise ValueError("Current implementation supports trunk_type='fnn' only.")

    if bayes_method == "alpha_vi":
        if branch_type.startswith("transformer"):
            print(
                f"Using BayesianTransformerBranch + BayesianFNNTrunk "
                f"(alpha={train_cfg.get('alpha', 1.0)})"
            )
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
            else:
                branch = BayesianTransformerBranch(
                    num_sensors=num_sensors,
                    output_dim=branch_output_dim,
                    d_model=model_cfg.get("transformer_d_model", 32),
                    nhead=model_cfg.get("transformer_nhead", 4),
                    num_layers=model_cfg.get("transformer_num_layers", 2),
                    dropout=model_cfg.get("transformer_dropout", 0.1),
                    prior_sigma=model_cfg.get("prior_sigma", 1.0),
                    input_channels=input_channels,
                )
        else:
            print(f"Using BayesianFNNBranch + BayesianFNNTrunk (alpha={train_cfg.get('alpha', 1.0)})")
            branch = BayesianFNNBranch(
                num_sensors=num_sensors,
                hidden_dims=branch_hidden,
                output_dim=branch_output_dim,
                prior_sigma=model_cfg.get("prior_sigma", 1.0),
            )
        trunk = BayesianFNNTrunk(
            input_dim=coord_dim,
            hidden_dims=trunk_hidden,
            output_dim=branch_output_dim,
            prior_sigma=model_cfg.get("prior_sigma", 1.0),
        )
        if is_multi_output:
            model = BayesianMultiOutputDeepONet(
                branch=branch,
                trunk=trunk,
                n_outputs=n_outputs,
                p_group=p_group,
                bias=True,
                min_noise=model_cfg.get("min_noise", 1e-3),
            )
        else:
            model = BayesianDeepONet(
                branch=branch,
                trunk=trunk,
                bias=True,
                min_noise=model_cfg.get("min_noise", 1e-3),
            )
    elif bayes_method == "deterministic":
        if branch_type.startswith("transformer"):
            print(
                f"Using TransformerBranch (d_model={model_cfg.get('transformer_d_model', 32)}, "
                f"nhead={model_cfg.get('transformer_nhead', 4)})"
            )
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
            else:
                branch = TransformerBranch(
                    num_sensors=num_sensors,
                    output_dim=branch_output_dim,
                    d_model=model_cfg.get("transformer_d_model", 32),
                    nhead=model_cfg.get("transformer_nhead", 4),
                    num_layers=model_cfg.get("transformer_num_layers", 2),
                    dropout=model_cfg.get("transformer_dropout", 0.1),
                    input_channels=input_channels,
                )
        else:
            print("Using FNNBranch")
            branch = FNNBranch(num_sensors, branch_hidden, branch_output_dim)
        trunk = FNNTrunk(coord_dim, trunk_hidden, branch_output_dim)
        if is_multi_output:
            model = MultiOutputDeepONet(
                branch=branch,
                trunk=trunk,
                n_outputs=n_outputs,
                p_group=p_group,
                bias=True,
            )
        else:
            model = DeepONet(branch, trunk, output_dim, bias=True)
    else:
        raise ValueError(f"Unsupported bayes_method: {bayes_method}")
    return model, bayes_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    physics_cfg = config.get("physics", {})
    data_cfg = config.get("data", {})
    case = config.get("case", "antiderivative")
    num_sensors = int(model_cfg.get("num_sensors", 50))

    generator = get_generator(case)
    gen_sig = inspect.signature(generator)
    if "n_sensors" in gen_sig.parameters:
        data_cfg.setdefault("n_sensors", num_sensors)

    print(f"Generating data for case={case} ...")
    data = generator(**data_cfg)
    if "domain" not in data:
        raise ValueError("Data generator must provide 'domain' for collocation sampling.")

    u_train = data["u_train"]
    y_train = data["y_train"]
    s_train = data["s_train"]
    coord_dim = int(y_train.shape[-1]) if y_train.ndim == 3 else 1
    n_outputs = int(s_train.shape[-1]) if s_train.ndim == 3 else 1
    if u_train.ndim == 3:
        input_channels = int(u_train.shape[-1])
    elif case.startswith("ns_") and u_train.ndim == 2 and u_train.shape[1] <= 4:
        input_channels = int(u_train.shape[1])
    else:
        input_channels = int(model_cfg.get("input_channels", 1))

    model_cfg = dict(model_cfg)
    model_cfg["coord_dim"] = coord_dim
    model_cfg["n_outputs"] = n_outputs
    model_cfg["input_channels"] = input_channels
    model, bayes_method = _build_model(
        model_cfg,
        train_cfg,
        coord_dim=coord_dim,
        n_outputs=n_outputs,
        input_channels=input_channels,
    )

    physics_mode = physics_cfg.get("physics_mode", "standard_pi")
    if physics_mode == "hard_bc_pi" and case in ("diffusion_reaction",):
        print(f"Wrapping model with HardBCWrapper for case={case}")
        model = HardBCWrapper(model, case)

    log_subdir = f"{case}" if physics_mode == "standard_pi" else f"{case}_{physics_mode}"
    print("Training...")
    _, _ = train_operator(
        model,
        data,
        case=case,
        lr=train_cfg.get("lr", 0.001),
        epochs=train_cfg.get("epochs", 10000),
        batch_size=train_cfg.get("batch_size", 256),
        log_dir=f"experiments/{log_subdir}",
        bayes_method=bayes_method,
        alpha=train_cfg.get("alpha", 1.0),
        mc_samples=train_cfg.get("mc_samples", 3),
        eval_mc_samples=train_cfg.get("eval_mc_samples", 20),
        kl_weight=train_cfg.get("kl_weight"),
        pi_constraint=physics_cfg.get("pi_constraint", "none"),
        pi_weight=physics_cfg.get("pi_weight", 0.0),
        bc_weight=physics_cfg.get("bc_weight", 0.0),
        ic_weight=physics_cfg.get("ic_weight", 0.0),
        physics_mode=physics_cfg.get("physics_mode", "standard_pi"),
        n_collocation=physics_cfg.get("n_collocation", 256),
        diffusion_D=physics_cfg.get("diffusion_D", 0.01),
        reaction_k=physics_cfg.get("reaction_k", 0.1),
        burgers_nu=physics_cfg.get("burgers_nu", 0.01 / torch.pi),
        ns_nu=physics_cfg.get("ns_nu", 1.0 / 40.0),
        ns_beltrami_nu=physics_cfg.get("ns_beltrami_nu", 1.0),
        pressure_gauge_weight=physics_cfg.get("pressure_gauge_weight", 0.0),
    )
    print("Done.")


if __name__ == "__main__":
    main()
