"""PI-BT-DeepONet: training entry with case dispatch."""

import yaml
from pathlib import Path

import torch

from src.data import get_generator
from src.data.generators import (  # noqa: F401 - force registration side effects
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
from src.training.trainer import train_operator


def _build_model(model_cfg: dict, train_cfg: dict, coord_dim: int):
    num_sensors = int(model_cfg.get("num_sensors", 100))
    output_dim = int(model_cfg.get("output_dim", 40))
    branch_type = model_cfg.get("branch_type", "fnn")
    trunk_type = model_cfg.get("trunk_type", "fnn")
    bayes_method = model_cfg.get("bayes_method", model_cfg.get("uq_mode", "deterministic"))
    branch_hidden = model_cfg.get("branch_hidden", [40, 40])
    trunk_hidden = model_cfg.get("trunk_hidden", [40, 40])

    if trunk_type != "fnn":
        raise ValueError("Current implementation supports trunk_type='fnn' only.")

    if bayes_method == "alpha_vi":
        if branch_type == "transformer":
            print(
                f"Using BayesianTransformerBranch + BayesianFNNTrunk "
                f"(alpha={train_cfg.get('alpha', 1.0)})"
            )
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
            print(f"Using BayesianFNNBranch + BayesianFNNTrunk (alpha={train_cfg.get('alpha', 1.0)})")
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
    elif bayes_method == "deterministic":
        if branch_type == "transformer":
            print(
                f"Using TransformerBranch (d_model={model_cfg.get('transformer_d_model', 32)}, "
                f"nhead={model_cfg.get('transformer_nhead', 4)})"
            )
            branch = TransformerBranch(
                num_sensors,
                output_dim,
                d_model=model_cfg.get("transformer_d_model", 32),
                nhead=model_cfg.get("transformer_nhead", 4),
                num_layers=model_cfg.get("transformer_num_layers", 2),
                dropout=model_cfg.get("transformer_dropout", 0.1),
            )
        else:
            print("Using FNNBranch")
            branch = FNNBranch(num_sensors, branch_hidden, output_dim)
        trunk = FNNTrunk(coord_dim, trunk_hidden, output_dim)
        model = DeepONet(branch, trunk, output_dim, bias=True)
    else:
        raise ValueError(f"Unsupported bayes_method: {bayes_method}")
    return model, bayes_method


def main():
    config_path = Path("configs/base.yaml")
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
    data_cfg.setdefault("n_sensors", num_sensors)

    print(f"Generating data for case={case} ...")
    generator = get_generator(case)
    data = generator(**data_cfg)
    if "domain" not in data:
        raise ValueError("Data generator must provide 'domain' for collocation sampling.")

    y_train = data["y_train"]
    coord_dim = int(y_train.shape[-1]) if y_train.ndim == 3 else 1
    model_cfg = dict(model_cfg)
    model_cfg["coord_dim"] = coord_dim
    model, bayes_method = _build_model(model_cfg, train_cfg, coord_dim=coord_dim)

    print("Training...")
    _, _ = train_operator(
        model,
        data,
        case=case,
        lr=train_cfg.get("lr", 0.001),
        epochs=train_cfg.get("epochs", 10000),
        batch_size=train_cfg.get("batch_size", 256),
        log_dir=f"experiments/{case}",
        bayes_method=bayes_method,
        alpha=train_cfg.get("alpha", 1.0),
        mc_samples=train_cfg.get("mc_samples", 3),
        eval_mc_samples=train_cfg.get("eval_mc_samples", 20),
        kl_weight=train_cfg.get("kl_weight"),
        pi_constraint=physics_cfg.get("pi_constraint", "none"),
        pi_weight=physics_cfg.get("pi_weight", 0.0),
        bc_weight=physics_cfg.get("bc_weight", 0.0),
        ic_weight=physics_cfg.get("ic_weight", 0.0),
        n_collocation=physics_cfg.get("n_collocation", 256),
        diffusion_D=physics_cfg.get("diffusion_D", 0.01),
        reaction_k=physics_cfg.get("reaction_k", 0.1),
        burgers_nu=physics_cfg.get("burgers_nu", 0.01 / torch.pi),
    )
    print("Done.")


if __name__ == "__main__":
    main()
