"""PI-BT-DeepONet: Training entry for antiderivative (stage 1)."""

import yaml
from pathlib import Path

import torch

from src.models import DeepONet, FNNBranch, FNNTrunk, TransformerBranch
from src.data.generators.antiderivative import generate_antiderivative_data
from src.training.trainer import train_antiderivative


def main():
    config_path = Path("configs/base.yaml")
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    num_sensors = model_cfg.get("num_sensors", 100)
    coord_dim = model_cfg.get("coord_dim", 1)
    output_dim = model_cfg.get("output_dim", 40)
    branch_type = model_cfg.get("branch_type", "fnn")
    branch_hidden = model_cfg.get("branch_hidden", [40, 40])
    trunk_hidden = model_cfg.get("trunk_hidden", [40, 40])

    if branch_type == "transformer":
        print(f"Using TransformerBranch (d_model={model_cfg.get('transformer_d_model', 32)}, nhead={model_cfg.get('transformer_nhead', 4)})")
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

    print("Generating antiderivative data...")
    data = generate_antiderivative_data(
        n_train=300,
        n_test=100,
        n_sensors=num_sensors,
        n_points_per_sample=10,
        length_scale=0.5,
        seed=42,
    )

    print("Training...")
    train_antiderivative(
        model,
        data,
        lr=train_cfg.get("lr", 0.001),
        epochs=train_cfg.get("epochs", 10000),
        batch_size=train_cfg.get("batch_size", 256),
        log_dir="experiments/antiderivative",
    )
    print("Done.")


if __name__ == "__main__":
    main()
