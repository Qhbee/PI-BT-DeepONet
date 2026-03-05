"""Compare 4 combinations: fnn/transformer x deterministic/alpha_vi, 100 epochs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from pathlib import Path

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

    combos = [
        ("deterministic", "fnn"),
        ("deterministic", "transformer"),
        ("alpha_vi", "fnn"),
        ("alpha_vi", "transformer"),
    ]

    results = []
    root = Path("experiments/compare_4_combos")
    root.mkdir(parents=True, exist_ok=True)

    for bayes_method, branch_type in combos:
        print(f"\n{'='*50}\n{bayes_method} + {branch_type}\n{'='*50}")

        if bayes_method == "deterministic":
            if branch_type == "transformer":
                branch = TransformerBranch(
                    num_sensors,
                    output_dim,
                    d_model=32,
                    nhead=4,
                    num_layers=2,
                    dropout=0.1,
                )
            else:
                branch = FNNBranch(num_sensors, branch_hidden, output_dim)
            trunk = FNNTrunk(coord_dim, trunk_hidden, output_dim)
            model = DeepONet(branch, trunk, output_dim, bias=True)
        else:
            if branch_type == "transformer":
                branch = BayesianTransformerBranch(
                    num_sensors,
                    output_dim,
                    d_model=32,
                    nhead=4,
                    num_layers=2,
                    dropout=0.1,
                    prior_sigma=1.0,
                )
            else:
                branch = BayesianFNNBranch(
                    num_sensors,
                    branch_hidden,
                    output_dim,
                    prior_sigma=1.0,
                )
            trunk = BayesianFNNTrunk(
                coord_dim,
                trunk_hidden,
                output_dim,
                prior_sigma=1.0,
            )
            model = BayesianDeepONet(branch, trunk, bias=True, min_noise=1e-3)

        n_params = count_params(model)
        t0 = time.perf_counter()
        _, metrics = train_antiderivative(
            model,
            data,
            lr=0.001,
            epochs=10,
            batch_size=64,
            log_dir=root / f"{bayes_method}_{branch_type}",
            device=device,
            bayes_method=bayes_method,
            alpha=1.0,
            mc_samples=3,
            eval_mc_samples=20,
        )
        elapsed = time.perf_counter() - t0

        results.append({
            "bayes_method": bayes_method,
            "branch_type": branch_type,
            "params": n_params,
            "time_s": elapsed,
            "loss": metrics["loss"],
            "rel_l2": metrics["rel_l2"],
            "test_mse": metrics["test_mse"],
        })

    # 打印表格
    print("\n" + "=" * 95)
    print("4 组合对比 (10 epochs, antiderivative)")
    print("=" * 95)
    header = f"{'组合':<30} {'参数量':>12} {'时间(s)':>10} {'时间(min)':>10} {'loss':>12} {'rel_l2':>10} {'test_mse':>12}"
    print(header)
    print("-" * 95)
    for r in results:
        combo = f"{r['bayes_method']} + {r['branch_type']}"
        print(f"{combo:<30} {r['params']:>12,} {r['time_s']:>10.1f} {r['time_s']/60:>10.2f} {r['loss']:>12.6f} {r['rel_l2']:>10.6f} {r['test_mse']:>12.6f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
