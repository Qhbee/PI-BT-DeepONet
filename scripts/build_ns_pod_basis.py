"""Build per-component POD bases from NS operator datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_generator
from src.data.generators import (  # noqa: F401
    generate_ns_beltrami_ic2field_data,
    generate_ns_beltrami_parametric_data,
    generate_ns_kovasznay_bc2field_data,
    generate_ns_kovasznay_parametric_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="ns_kovasznay_parametric")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--output", type=str, default="artifacts/pod")
    args = parser.parse_args()

    generator = get_generator(args.case)
    data = generator()
    s_train = data["s_train"]  # (n_samples, n_points, n_out)
    if s_train.ndim != 3:
        raise ValueError(f"Expected multi-output s_train with shape (N,P,C), got {s_train.shape}.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_out = s_train.shape[-1]
    rank = min(args.rank, s_train.shape[0], s_train.shape[1])

    result: dict[str, np.ndarray] = {}
    for c in range(n_out):
        snapshots = s_train[..., c]  # (N, P)
        snapshots_centered = snapshots - snapshots.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(snapshots_centered, full_matrices=False)
        basis = vt[:rank].T.astype(np.float32)  # (P, rank)
        result[f"basis_comp_{c}"] = basis

    result["rank"] = np.array([rank], dtype=np.int32)
    np.savez(out_dir / f"{args.case}_pod_rank{rank}.npz", **result)
    print(f"Saved POD basis to {out_dir / f'{args.case}_pod_rank{rank}.npz'}")


if __name__ == "__main__":
    main()
