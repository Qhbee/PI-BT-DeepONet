"""Beltrami NS dataset (route B): t=0 initial sequence -> full field (u, v, w, p)."""

from __future__ import annotations

import numpy as np

from src.data.registry import register


def _beltrami_solution_np(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    d: float,
) -> np.ndarray:
    exp_decay = np.exp(-(d**2) * t)
    u = -a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * exp_decay
    v = -a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * exp_decay
    w = -a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * exp_decay
    p = -0.5 * (a**2) * (
        np.exp(2.0 * a * x)
        + np.exp(2.0 * a * y)
        + np.exp(2.0 * a * z)
        + 2.0 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z))
        + 2.0 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x))
        + 2.0 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))
    ) * np.exp(-2.0 * (d**2) * t)
    return np.stack([u, v, w, p], axis=-1).astype(np.float32)


@register("ns_beltrami_ic2field")
def generate_ns_beltrami_ic2field_data(
    n_train: int = 120,
    n_test: int = 30,
    a_min: float = 0.5,
    a_max: float = 1.5,
    d_min: float = 0.5,
    d_max: float = 1.5,
    nt: int = 6,
    nx: int = 12,
    ny: int = 12,
    nz: int = 12,
    sensor_side: int = 4,
    seed: int | None = 2026,
) -> dict:
    """
    Route B for Beltrami:
    - Branch input: t=0 initial velocity sequence on a coarse 3D sensor grid, shape (batch, m, 3)
    - Output: full field (u,v,w,p)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
    tt, xx, yy, zz = np.meshgrid(t, x, y, z, indexing="ij")
    coords = np.stack([tt.reshape(-1), xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1).astype(np.float32)

    sx = np.linspace(-1.0, 1.0, sensor_side, dtype=np.float32)
    sy = np.linspace(-1.0, 1.0, sensor_side, dtype=np.float32)
    sz = np.linspace(-1.0, 1.0, sensor_side, dtype=np.float32)
    sxx, syy, szz = np.meshgrid(sx, sy, sz, indexing="ij")
    st0 = np.zeros_like(sxx, dtype=np.float32)
    sensor_coords = np.stack([st0.reshape(-1), sxx.reshape(-1), syy.reshape(-1), szz.reshape(-1)], axis=-1).astype(np.float32)

    def build(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = params.shape[0]
        s = np.array(
            [
                _beltrami_solution_np(tt, xx, yy, zz, float(params[i, 0]), float(params[i, 1])).reshape(-1, 4)
                for i in range(n)
            ],
            dtype=np.float32,
        )
        y_out = np.broadcast_to(coords[None, :, :], (n, coords.shape[0], 4)).copy()

        u_in = []
        for i in range(n):
            a_i = float(params[i, 0])
            d_i = float(params[i, 1])
            uvwp0 = _beltrami_solution_np(st0, sxx, syy, szz, a_i, d_i).reshape(-1, 4)
            u_in.append(uvwp0[:, :3])  # (m,3)
        u_in = np.array(u_in, dtype=np.float32)
        return u_in, y_out, s

    ad_train = np.stack(
        [
            np.random.uniform(a_min, a_max, size=(n_train,)),
            np.random.uniform(d_min, d_max, size=(n_train,)),
        ],
        axis=-1,
    ).astype(np.float32)
    ad_test = np.stack(
        [
            np.random.uniform(a_min, a_max, size=(n_test,)),
            np.random.uniform(d_min, d_max, size=(n_test,)),
        ],
        axis=-1,
    ).astype(np.float32)
    u_train, y_train, s_train = build(ad_train)
    u_test, y_test, s_test = build(ad_test)

    return {
        "u_train": u_train,
        "y_train": y_train,
        "s_train": s_train,
        "u_test": u_test,
        "y_test": y_test,
        "s_test": s_test,
        "params_train": ad_train,
        "params_test": ad_test,
        "sensor_coords": sensor_coords,
        "domain": {"min": [0.0, -1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0, 1.0]},
    }
