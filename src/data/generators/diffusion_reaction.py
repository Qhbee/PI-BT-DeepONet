"""Diffusion-reaction operator data generator (1D in space, time dependent)."""

from __future__ import annotations

import numpy as np

from src.data.registry import register


def _sample_source(
    n_samples: int,
    x_sensors: np.ndarray,
    length_scale: float = 0.25,
) -> np.ndarray:
    dist = np.subtract.outer(x_sensors, x_sensors) ** 2
    cov = np.exp(-dist / (2.0 * length_scale**2))
    chol = np.linalg.cholesky(cov + 1e-6 * np.eye(x_sensors.shape[0], dtype=np.float32))
    z = np.random.randn(n_samples, x_sensors.shape[0]).astype(np.float32)
    return (chol @ z.T).T.astype(np.float32)


def _solve_diffusion_reaction(
    source_x: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    D: float,
    k: float,
) -> np.ndarray:
    """Explicit FD solver for s_t = D s_xx + k s^2 + source(x)."""
    nx = x_grid.shape[0]
    nt = t_grid.shape[0]
    dx = float(x_grid[1] - x_grid[0])
    dt = float(t_grid[1] - t_grid[0])

    s = np.zeros((nt, nx), dtype=np.float32)
    # Zero-Dirichlet BC and zero IC.
    for n in range(1, nt):
        prev = s[n - 1]
        lap = np.zeros_like(prev)
        lap[1:-1] = (prev[2:] - 2.0 * prev[1:-1] + prev[:-2]) / (dx * dx)
        next_s = prev + dt * (D * lap + k * (prev**2) + source_x)
        next_s[0] = 0.0
        next_s[-1] = 0.0
        s[n] = next_s
    return s


@register("diffusion_reaction")
def generate_diffusion_reaction_data(
    n_train: int = 300,
    n_test: int = 100,
    n_sensors: int = 100,
    nx: int = 100,
    nt: int = 101,
    D: float = 0.01,
    k: float = 0.1,
    seed: int | None = 123,
) -> dict:
    """Generate data for diffusion-reaction operator learning."""
    if seed is not None:
        np.random.seed(seed)

    x_sensors = np.linspace(0.0, 1.0, n_sensors, dtype=np.float32)
    x_grid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    t_grid = np.linspace(0.0, 1.0, nt, dtype=np.float32)

    xx, tt = np.meshgrid(x_grid, t_grid)
    y_points = np.stack([tt.reshape(-1), xx.reshape(-1)], axis=-1).astype(np.float32)

    def solve_batch(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_src = _sample_source(n_samples, x_sensors)
        src_grid = np.array(
            [np.interp(x_grid, x_sensors, u_src[i]) for i in range(n_samples)],
            dtype=np.float32,
        )
        sol = np.array(
            [_solve_diffusion_reaction(src_grid[i], x_grid, t_grid, D=D, k=k) for i in range(n_samples)],
            dtype=np.float32,
        )
        s = sol.reshape(n_samples, -1)
        y = np.broadcast_to(y_points[None, :, :], (n_samples, y_points.shape[0], 2)).copy()
        return u_src.astype(np.float32), y.astype(np.float32), s.astype(np.float32)

    u_train, y_train, s_train = solve_batch(n_train)
    u_test, y_test, s_test = solve_batch(n_test)

    return {
        "u_train": u_train,
        "y_train": y_train,
        "s_train": s_train,
        "u_test": u_test,
        "y_test": y_test,
        "s_test": s_test,
        "x_sensors": x_sensors,
        "domain": {"min": [0.0, 0.0], "max": [1.0, 1.0]},
    }
