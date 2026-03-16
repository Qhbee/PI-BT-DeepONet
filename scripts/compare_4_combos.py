"""Compare 4 combinations: fnn/transformer x deterministic/alpha_vi (Stage 3)."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint every N epochs (0=disable)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = generate_antiderivative_data(
        n_train=300,
        n_test=100,
        n_sensors=50,
        n_points_per_sample=10,
        length_scale=0.5,
        seed=args.seed,
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
        log_dir = root / f"{bayes_method}_{branch_type}"
        _, metrics = train_antiderivative(
            model,
            data,
            lr=0.001,
            epochs=args.epochs,
            batch_size=64,
            log_dir=str(log_dir),
            device=device,
            bayes_method=bayes_method,
            alpha=1.0,
            mc_samples=3,
            eval_mc_samples=20,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume if (args.resume and f"{bayes_method}_{branch_type}" in args.resume) else None,
        )
        elapsed = time.perf_counter() - t0

        # 保存每 epoch 的 loss/rel_l2/test_mse 历史
        hist = metrics.get("history", [])
        if hist:
            hist_path = Path(log_dir) / "training_history.json"
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
            print(f"  [History] {len(hist)} 条 -> {hist_path}")

        results.append({
            "bayes_method": bayes_method,
            "branch_type": branch_type,
            "params": n_params,
            "time_s": elapsed,
            "loss": metrics["loss"],
            "rel_l2": metrics["rel_l2"],
            "test_mse": metrics["test_mse"],
        })

    # 保存 CSV
    csv_path = root / f"stage3_compare_4_combos_epochs{args.epochs}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("bayes_method,branch_type,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['bayes_method']},{r['branch_type']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[CSV] {csv_path}")

    # 打印表格
    print("\n" + "=" * 95)
    print(f"Stage 3: 4 组合对比 (epochs={args.epochs}, antiderivative)")
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
