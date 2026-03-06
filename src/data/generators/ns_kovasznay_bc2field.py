"""Kovasznay NS dataset (route B): inlet velocity sequence -> full field (u, v, p)."""

from __future__ import annotations

import numpy as np

from src.data.registry import register


def _kovasznay_solution_np(x: np.ndarray, y: np.ndarray, re: float) -> np.ndarray:
    zeta = 0.5 * re - np.sqrt(0.25 * re * re + 4.0 * np.pi * np.pi)
    u = 1.0 - np.exp(zeta * x) * np.cos(2.0 * np.pi * y)
    v = (zeta / (2.0 * np.pi)) * np.exp(zeta * x) * np.sin(2.0 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2.0 * zeta * x))
    return np.stack([u, v, p], axis=-1).astype(np.float32)


@register("ns_kovasznay_bc2field")
def generate_ns_kovasznay_bc2field_data(
    n_train: int = 200,
    n_test: int = 50,
    n_sensors: int = 64,
    re_min: float = 20.0,
    re_max: float = 60.0,
    nx: int = 64,
    ny: int = 64,
    seed: int | None = 77,
) -> dict:
    """
    Route B for Kovasznay:
    - Branch input: inlet sequence at x=-0.5 with channels (u,v), shape (batch, m, 2)
    - Output: full field (u,v,p)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(-0.5, 1.0, nx, dtype=np.float32)
    y = np.linspace(-0.5, 1.5, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)

    y_inlet = np.linspace(-0.5, 1.5, n_sensors, dtype=np.float32)
    x_inlet = np.full_like(y_inlet, -0.5, dtype=np.float32)

    def build(re_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = re_values.shape[0]
        s = np.array(
            [_kovasznay_solution_np(xx, yy, float(re)).reshape(-1, 3) for re in re_values],
            dtype=np.float32,
        )
        y_out = np.broadcast_to(coords[None, :, :], (n, coords.shape[0], 2)).copy()

        inlet = []
        for re in re_values:
            uvp_in = _kovasznay_solution_np(x_inlet, y_inlet, float(re))
            inlet.append(uvp_in[:, :2])  # (m,2)
        u_in = np.array(inlet, dtype=np.float32)
        return u_in, y_out, s

    re_train = np.random.uniform(re_min, re_max, size=(n_train,)).astype(np.float32)
    re_test = np.random.uniform(re_min, re_max, size=(n_test,)).astype(np.float32)
    u_train, y_train, s_train = build(re_train)
    u_test, y_test, s_test = build(re_test)

    return {
        "u_train": u_train,
        "y_train": y_train,
        "s_train": s_train,
        "u_test": u_test,
        "y_test": y_test,
        "s_test": s_test,
        "params_train": re_train[:, None].astype(np.float32),
        "params_test": re_test[:, None].astype(np.float32),
        "sensor_coords": np.stack([x_inlet, y_inlet], axis=-1).astype(np.float32),
        "domain": {"min": [-0.5, -0.5], "max": [1.0, 1.5]},
    }
