"""Compare 2x2x2 combinations: branch x bayes x pi_constraint, 2 epochs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import torch

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
from src.data.generators.antiderivative import generate_antiderivative_data
from src.training.trainer import train_antiderivative


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(bayes_method: str, branch_type: str, num_sensors: int, output_dim: int, branch_hidden: list, trunk_hidden: list, coord_dim: int):
    if bayes_method == "deterministic":
        if branch_type == "transformer":
            branch = TransformerBranch(num_sensors, output_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1)
        else:
            branch = FNNBranch(num_sensors, branch_hidden, output_dim)
        trunk = FNNTrunk(coord_dim, trunk_hidden, output_dim)
        return DeepONet(branch, trunk, output_dim, bias=True)
    else:
        if branch_type == "transformer":
            branch = BayesianTransformerBranch(
                num_sensors, output_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1, prior_sigma=1.0
            )
        else:
            branch = BayesianFNNBranch(num_sensors, branch_hidden, output_dim, prior_sigma=1.0)
        trunk = BayesianFNNTrunk(coord_dim, trunk_hidden, output_dim, prior_sigma=1.0)
        return BayesianDeepONet(branch, trunk, bias=True, min_noise=1e-3)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = generate_antiderivative_data(
        n_train=300,
        n_test=100,
        n_sensors=50,
        n_points_per_sample=10,
        length_scale=0.5,
        seed=42,
    )

    output_dim = 20
    branch_hidden = [20, 20]
    trunk_hidden = [20, 20]
    num_sensors = 50
    coord_dim = 1

    combos = []
    for bayes in ("deterministic", "alpha_vi"):
        for branch in ("fnn", "transformer"):
            for pi in ("none", "antiderivative"):
                combos.append((bayes, branch, pi))

    results = []
    root = Path("experiments/compare_8_combos")
    root.mkdir(parents=True, exist_ok=True)

    for bayes_method, branch_type, pi_constraint in combos:
        tag = f"{bayes_method}_{branch_type}_{pi_constraint}"
        print(f"\n{'='*60}\n{tag}\n{'='*60}")

        model = build_model(bayes_method, branch_type, num_sensors, output_dim, branch_hidden, trunk_hidden, coord_dim)
        n_params = count_params(model)

        pi_weight = 0.1 if pi_constraint == "antiderivative" else 0.0
        n_collocation = 128 if pi_constraint == "antiderivative" else 0

        t0 = time.perf_counter()
        _, metrics = train_antiderivative(
            model,
            data,
            lr=0.001,
            epochs=2,
            batch_size=64,
            log_dir=root / tag,
            device=device,
            bayes_method=bayes_method,
            alpha=1.0,
            mc_samples=3,
            eval_mc_samples=20,
            pi_constraint=pi_constraint,
            pi_weight=pi_weight,
            n_collocation=n_collocation,
        )
        elapsed = time.perf_counter() - t0

        results.append({
            "tag": tag,
            "params": n_params,
            "time_s": elapsed,
            "loss": metrics["loss"],
            "rel_l2": metrics["rel_l2"],
            "test_mse": metrics["test_mse"],
        })

    print("\n" + "=" * 110)
    print("2x2x2 组合对比 (2 epochs: branch x bayes x pi)")
    print("=" * 110)
    header = f"{'组合':<45} {'参数量':>12} {'时间(s)':>10} {'loss':>14} {'rel_l2':>10} {'test_mse':>12}"
    print(header)
    print("-" * 110)
    for r in results:
        print(f"{r['tag']:<45} {r['params']:>12,} {r['time_s']:>10.1f} {r['loss']:>14.6f} {r['rel_l2']:>10.6f} {r['test_mse']:>12.6f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
