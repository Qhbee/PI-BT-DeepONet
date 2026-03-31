"""2D Poisson operator: -∇²p = f on [0,1]², p=0 on boundary. Analytical solution via Fourier."""

from __future__ import annotations

import numpy as np

from src.data.registry import register


def _sample_fourier_coeffs(
    n_samples: int,
    max_mode: int,
    length_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Fourier coefficients a_mn for f = Σ a_mn sin(mπx) sin(nπy)."""
    n_modes = max_mode * max_mode
    dist = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        m1, n1 = i // max_mode + 1, i % max_mode + 1
        for j in range(n_modes):
            m2, n2 = j // max_mode + 1, j % max_mode + 1
            dist[i, j] = (m1 - m2) ** 2 + (n1 - n2) ** 2
    cov = np.exp(-dist / (2 * length_scale**2))
    cov += 1e-6 * np.eye(n_modes)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_samples, n_modes)).astype(np.float32)
    return (L @ z.T).T.astype(np.float32)


def _fourier_to_field(
    coeffs: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    max_mode: int,
) -> np.ndarray:
    """Evaluate f(x,y) = Σ a_mn sin(mπx) sin(nπy) on grid (xx, yy)."""
    out = np.zeros_like(xx, dtype=np.float32)
    for idx in range(coeffs.shape[0]):
        m, n = idx // max_mode + 1, idx % max_mode + 1
        out += coeffs[idx] * np.sin(m * np.pi * xx) * np.sin(n * np.pi * yy)
    return out


def _fourier_to_solution(
    coeffs: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    max_mode: int,
) -> np.ndarray:
    """Evaluate p(x,y) = Σ a_mn/(π²(m²+n²)) sin(mπx) sin(nπy) on grid."""
    out = np.zeros_like(xx, dtype=np.float32)
    for idx in range(coeffs.shape[0]):
        m, n = idx // max_mode + 1, idx % max_mode + 1
        lam = np.pi**2 * (m**2 + n**2)
        out += (coeffs[idx] / lam) * np.sin(m * np.pi * xx) * np.sin(n * np.pi * yy)
    return out


def fourier_to_solution(
    coeffs: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    max_mode: int,
) -> np.ndarray:
    """Ground-truth solution p on a grid; public alias for visualization and tests.

    Args:
        coeffs: 1D array of length max_mode**2 (single sample) or flattened.
    """
    c = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    return _fourier_to_solution(c, xx, yy, max_mode)


@register("poisson_2d")
def generate_poisson_2d_data(
    n_train: int = 300,
    n_test: int = 100,
    nx: int = 16,
    ny: int = 16,
    n_points_per_sample: int = 256,
    max_mode: int = 8,
    length_scale: float = 2.0,
    seed: int | None = 42,
    query_sampling: str = "uniform",
    return_coeffs: bool = False,
) -> dict:
    """
    2D Poisson: -∇²p = f on [0,1]², p=0 on boundary.
    Analytical: f = Σ a_mn sin(mπx) sin(nπy) → p = Σ a_mn/(π²(m²+n²)) sin(mπx) sin(nπy).

    Args:
        n_train, n_test: sample counts
        nx, ny: source grid resolution (n_sensors = nx*ny)
        n_points_per_sample: query points per sample
        max_mode: Fourier modes m,n in [1, max_mode]
        length_scale: GRF length scale for coefficient sampling
        seed: random seed
        query_sampling: "uniform" (regular grid) or "random" (uniform in [0,1]²)
        return_coeffs: If True, include coeffs_train / coeffs_test (shape n_samples × max_mode²)
            aligned with u_train / u_test for analytic p on arbitrary grids.
    """
    rng = np.random.default_rng(seed)
    n_sensors = nx * ny

    x_grid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y_grid = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xx_sens, yy_sens = np.meshgrid(x_grid, y_grid, indexing="xy")

    def sample_batch(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coeffs = _sample_fourier_coeffs(n_samples, max_mode, length_scale, rng)
        f_at_sensors = np.array(
            [_fourier_to_field(coeffs[i], xx_sens, yy_sens, max_mode) for i in range(n_samples)],
            dtype=np.float32,
        )
        u_flat = f_at_sensors.reshape(n_samples, -1)

        if query_sampling == "uniform":
            n_side = int(np.sqrt(n_points_per_sample))
            n_side = max(2, n_side)
            n_pts = n_side * n_side
            x_pts = np.linspace(0.0, 1.0, n_side, dtype=np.float32)
            y_pts = np.linspace(0.0, 1.0, n_side, dtype=np.float32)
            xx_pts, yy_pts = np.meshgrid(x_pts, y_pts, indexing="xy")
            xy_query = np.stack([xx_pts.reshape(-1), yy_pts.reshape(-1)], axis=-1)
            xy_query = np.broadcast_to(xy_query[None, :, :], (n_samples, n_pts, 2)).copy()
        else:
            xy_query = rng.uniform(0.0, 1.0, (n_samples, n_points_per_sample, 2)).astype(np.float32)

        p_at_query = np.array(
            [
                _fourier_to_solution(coeffs[i], xy_query[i, :, 0], xy_query[i, :, 1], max_mode)
                for i in range(n_samples)
            ],
            dtype=np.float32,
        )
        p_at_query = p_at_query[:, :, np.newaxis]
        return u_flat, xy_query.astype(np.float32), p_at_query, coeffs

    u_train, y_train, s_train, coeffs_train = sample_batch(n_train)
    u_test, y_test, s_test, coeffs_test = sample_batch(n_test)

    x_sensors = np.stack([xx_sens.reshape(-1), yy_sens.reshape(-1)], axis=-1).astype(np.float32)
    out: dict = {
        "u_train": u_train,
        "y_train": y_train,
        "s_train": s_train,
        "u_test": u_test,
        "y_test": y_test,
        "s_test": s_test,
        "x_sensors": x_sensors,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "grid_shape": (ny, nx),
        "domain": {"min": [0.0, 0.0], "max": [1.0, 1.0]},
    }
    if return_coeffs:
        out["coeffs_train"] = coeffs_train
        out["coeffs_test"] = coeffs_test
    return out
