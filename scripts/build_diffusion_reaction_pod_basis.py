"""Build POD basis from diffusion_reaction operator data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generators import generate_diffusion_reaction_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--output", type=str, default="artifacts/pod")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--nt", type=int, default=21)
    args = parser.parse_args()

    data = generate_diffusion_reaction_data(
        n_train=args.n_train,
        n_test=10,
        n_sensors=20,
        nx=args.nx,
        nt=args.nt,
        seed=123,
    )
    s_train = data["s_train"]  # (n_samples, n_points)
    y_train = data["y_train"]  # (n_samples, n_points, 2)
    n_points = s_train.shape[1]
    rank = min(args.rank, s_train.shape[0], n_points)

    snapshots = s_train.astype(np.float64)
    mean_field = snapshots.mean(axis=0)
    snapshots_centered = snapshots - mean_field
    _, _, vt = np.linalg.svd(snapshots_centered, full_matrices=False)
    basis = vt[:rank].T.astype(np.float32)  # (n_points, rank)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_grid = np.unique(y_train[0, :, 0])
    x_grid = np.unique(y_train[0, :, 1])
    np.savez(
        out_dir / "diffusion_reaction_pod.npz",
        basis=basis,
        mean_field=mean_field.astype(np.float32),
        rank=np.array([rank], dtype=np.int32),
        t_grid=t_grid.astype(np.float32),
        x_grid=x_grid.astype(np.float32),
    )
    print(f"Saved POD basis (rank={rank}) to {out_dir / 'diffusion_reaction_pod.npz'}")


if __name__ == "__main__":
    main()
