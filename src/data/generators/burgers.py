"""Burgers operator data generator (periodic 1D)."""

from __future__ import annotations

import numpy as np

from src.data.registry import register


def _burgers_rhs(u: np.ndarray, k: np.ndarray, nu: float) -> np.ndarray:
    """RHS of periodic Burgers in Fourier form."""
    u_hat = np.fft.rfft(u)
    ux = np.fft.irfft(1j * k * u_hat, n=u.shape[0]).real
    uxx = np.fft.irfft(-(k**2) * u_hat, n=u.shape[0]).real
    return -u * ux + nu * uxx


def _solve_burgers_periodic(
    u0: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    nu: float,
) -> np.ndarray:
    """RK4 solver for u_t + u u_x = nu u_xx on periodic [0,1]."""
    nx = x_grid.shape[0]
    dt_out = float(t_grid[1] - t_grid[0])
    dx = 1.0 / nx
    # rfft wavenumbers for domain length 1
    k = 2.0 * np.pi * np.fft.rfftfreq(nx, d=1.0 / nx)

    out = np.zeros((t_grid.shape[0], nx), dtype=np.float32)
    u = u0.astype(np.float64).copy()
    out[0] = u.astype(np.float32)
    for i in range(1, t_grid.shape[0]):
        max_u = max(1e-6, float(np.max(np.abs(u))))
        dt_cfl = 0.4 * min(dx / max_u, dx * dx / max(1e-6, 2.0 * nu))
        n_sub = max(1, int(np.ceil(dt_out / dt_cfl)))
        dt = dt_out / n_sub
        for _ in range(n_sub):
            k1 = _burgers_rhs(u, k, nu)
            k2 = _burgers_rhs(u + 0.5 * dt * k1, k, nu)
            k3 = _burgers_rhs(u + 0.5 * dt * k2, k, nu)
            k4 = _burgers_rhs(u + dt * k3, k, nu)
            u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = u.astype(np.float32)
    return out


@register("burgers")
def generate_burgers_data(
    n_train: int = 300,
    n_test: int = 100,
    n_sensors: int = 128,
    nx: int = 128,
    nt: int = 101,
    nu: float = 0.05 / np.pi,
    seed: int | None = 42,
) -> dict:
    """Generate Burgers operator data using randomized sine-series ICs."""
    if seed is not None:
        np.random.seed(seed)

    x_grid = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    t_grid = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    x_sensors = np.linspace(0.0, 1.0, n_sensors, dtype=np.float32)

    xx, tt = np.meshgrid(x_grid, t_grid)
    y_points = np.stack([tt.reshape(-1), xx.reshape(-1)], axis=-1).astype(np.float32)

    def sample_ic(n_samples: int) -> np.ndarray:
        coeff1 = np.random.uniform(0.1, 0.5, size=(n_samples, 1)).astype(np.float32)
        coeff2 = np.random.uniform(-0.1, 0.1, size=(n_samples, 1)).astype(np.float32)
        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(n_samples, 1)).astype(np.float32)
        x = x_sensors[None, :]
        u0 = coeff1 * np.sin(2.0 * np.pi * x + phase) + coeff2 * np.sin(4.0 * np.pi * x)
        return u0.astype(np.float32)

    def solve_batch(u0_sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = u0_sensors.shape[0]
        # Interpolate IC from sensor grid to solver grid
        u0_grid = np.array(
            [np.interp(x_grid, x_sensors, u0_sensors[i]) for i in range(n_samples)],
            dtype=np.float32,
        )
        sol = np.array(
            [_solve_burgers_periodic(u0_grid[i], x_grid, t_grid, nu) for i in range(n_samples)],
            dtype=np.float32,
        )
        s = sol.reshape(n_samples, -1)
        y = np.broadcast_to(y_points[None, :, :], (n_samples, y_points.shape[0], 2)).copy()
        return u0_sensors, y.astype(np.float32), s.astype(np.float32)

    u_train0 = sample_ic(n_train)
    u_test0 = sample_ic(n_test)
    u_train, y_train, s_train = solve_batch(u_train0)
    u_test, y_test, s_test = solve_batch(u_test0)

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
