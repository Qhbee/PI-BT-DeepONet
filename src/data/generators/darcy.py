"""Simple Darcy operator data generator (manufactured 2D solution)."""

from __future__ import annotations

import numpy as np

from src.data.registry import register


@register("darcy")
def generate_darcy_data(
    n_train: int = 200,
    n_test: int = 50,
    n_sensors: int = 64,
    nx: int = 32,
    ny: int = 32,
    seed: int | None = 7,
) -> dict:
    """
    Generate a lightweight Darcy dataset with manufactured solution.

    PDE:
        -div(k grad p) = f
    with constant permeability k per sample and p=0 on domain boundary.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)

    def solve_for_k(kvals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = kvals.shape[0]
        # Manufactured pressure with homogeneous Dirichlet boundary.
        p = np.sin(np.pi * xx)[None, :, :] * np.sin(np.pi * yy)[None, :, :]
        p = np.broadcast_to(p, (n_samples, ny, nx)).astype(np.float32)
        p[:, 0, :] = 0.0
        p[:, -1, :] = 0.0
        p[:, :, 0] = 0.0
        p[:, :, -1] = 0.0

        # Represent permeability input as a flat sensor vector.
        # Here each sample has scalar k repeated on sensors for simplicity.
        u_in = np.repeat(kvals[:, None], n_sensors, axis=1).astype(np.float32)
        y_out = np.broadcast_to(xy[None, :, :], (n_samples, xy.shape[0], 2)).copy()
        s_out = p.reshape(n_samples, -1).astype(np.float32)
        return u_in, y_out, s_out

    k_train = np.random.uniform(0.5, 2.0, size=(n_train,)).astype(np.float32)
    k_test = np.random.uniform(0.5, 2.0, size=(n_test,)).astype(np.float32)
    u_train, y_train, s_train = solve_for_k(k_train)
    u_test, y_test, s_test = solve_for_k(k_test)

    x_sensors = np.linspace(0.0, 1.0, n_sensors, dtype=np.float32)
    return {
        "u_train": u_train,
        "y_train": y_train,
        "s_train": s_train,
        "u_test": u_test,
        "y_test": y_test,
        "s_test": s_test,
        "x_sensors": x_sensors,
        "k_train": k_train,
        "k_test": k_test,
        "domain": {"min": [0.0, 0.0], "max": [1.0, 1.0]},
    }
