"""Antiderivative operator: ds/dx = u(x), s(0)=0. Solution: s(x) = int_0^x u(tau) d tau."""

import numpy as np

from src.data.registry import register

def sample_segmented_grf_u(
    n_samples: int,
    n_sensors: int,
    length_scale: float,
    domain: tuple = (0.0, 1.0),
    *,
    seed: int | None = None,
    min_segments: int = 2,
    max_segments: int = 4,
) -> np.ndarray:
    """分段 GRF：每一段在传感器子网格上独立抽取同核长 ``length_scale`` 的 GRF；段与段在接缝处一般不相等，故 \(u\) 不连续、\(s=\int u\) 仍连续。"""
    if n_samples <= 0:
        return np.zeros((0, n_sensors), dtype=np.float32)
    if n_sensors < 2:
        return sample_grf(n_samples, n_sensors, length_scale, domain)
    rng = np.random.default_rng(seed)
    x_full = np.linspace(domain[0], domain[1], n_sensors).astype(np.float64)
    u = np.zeros((n_samples, n_sensors), dtype=np.float32)
    max_seg_eff = min(max_segments, n_sensors)
    low = min(min_segments, max_seg_eff)
    if low > max_seg_eff:
        return sample_grf(n_samples, n_sensors, length_scale, domain)
    for i in range(n_samples):
        n_seg = int(rng.integers(low, max_seg_eff + 1))
        n_breaks = n_seg - 1
        breaks = np.sort(
            rng.choice(np.arange(1, n_sensors, dtype=np.int64), size=n_breaks, replace=False)
        )
        edges = np.concatenate((np.array([0], dtype=np.int64), breaks, np.array([n_sensors], dtype=np.int64)))
        for j in range(len(edges) - 1):
            lo, hi = int(edges[j]), int(edges[j + 1])
            seg_len = hi - lo
            if seg_len < 1:
                continue
            sub_x = x_full[lo:hi]
            dist = np.subtract.outer(sub_x, sub_x) ** 2
            cov = np.exp(-dist / (2 * length_scale**2))
            L = np.linalg.cholesky(cov + 1e-6 * np.eye(seg_len))
            z = rng.standard_normal(seg_len)
            u[i, lo:hi] = (L @ z).astype(np.float32)
    return u


def sample_piecewise_constant_u(
    n_samples: int,
    n_sensors: int,
    domain: tuple = (0.0, 1.0),
    *,
    seed: int | None = None,
    min_segments: int = 2,
    max_segments: int = 6,
) -> np.ndarray:
    """分段常值 \(u(x)\)：在随机分界点处跳变，用于覆盖不光滑输入（反导数 \(s\) 仍连续、分段仿射）。"""
    if n_samples <= 0:
        return np.zeros((0, n_sensors), dtype=np.float32)
    rng = np.random.default_rng(seed)
    u = np.zeros((n_samples, n_sensors), dtype=np.float32)
    for i in range(n_samples):
        n_seg = int(rng.integers(min_segments, max_segments + 1))
        if n_seg <= 1 or n_sensors < 2:
            u[i, :] = rng.normal(0.0, 1.0)
            continue
        breaks = np.sort(rng.choice(np.arange(1, n_sensors, dtype=np.int64), size=n_seg - 1, replace=False))
        edges = np.concatenate((np.array([0], dtype=np.int64), breaks, np.array([n_sensors], dtype=np.int64)))
        for j in range(len(edges) - 1):
            lo, hi = int(edges[j]), int(edges[j + 1])
            u[i, lo:hi] = np.float32(rng.normal(0.0, 1.0))
    return u


def sample_grf(n_samples: int, n_sensors: int, length_scale: float = 0.2, domain: tuple = (0.0, 1.0)) -> np.ndarray:
    """Sample from Gaussian Random Field with RBF kernel."""
    x = np.linspace(domain[0], domain[1], n_sensors)
    dist = np.subtract.outer(x, x) ** 2
    cov = np.exp(-dist / (2 * length_scale**2))
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(n_sensors))
    z = np.random.randn(n_samples, n_sensors)
    return (L @ z.T).T.astype(np.float32)


def solve_antiderivative(u_at_sensors: np.ndarray, x_sensors: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Compute s(x) = int_0^x u(tau) d tau via trapezoidal rule.
    x_query: (n_samples, n_points) query coordinates per sample.
    """
    n_samples = u_at_sensors.shape[0]
    dx = np.diff(x_sensors)
    s_sensors = np.zeros((n_samples, len(x_sensors)), dtype=np.float32)
    s_sensors[:, 1:] = np.cumsum(dx * (u_at_sensors[:, :-1] + u_at_sensors[:, 1:]) / 2, axis=1)
    s = np.array([np.interp(x_query[i], x_sensors, s_sensors[i]) for i in range(n_samples)], dtype=np.float32)
    return s


@register("antiderivative")
def generate_antiderivative_data(
    n_train: int = 3000,
    n_test: int = 1000,
    n_sensors: int = 100,
    n_points_per_sample: int = 20,
    length_scale: float = 0.5,
    domain: tuple = (0.0, 1.0),
    seed: int | None = None,
    discontinuous_train: int = 0,
    discontinuous_test: int = 0,
):
    """Generate training and test data for antiderivative operator.

    discontinuous_train / discontinuous_test: 将对应条数的样本替换为分段 GRF \(u\)（段数每条随机 \(n_{\mathrm{seg}}\in\{2,3,4\}\)，
    段间一般不连续），其余为全局光滑 GRF；总数仍为 n_train / n_test。
    """
    if seed is not None:
        np.random.seed(seed)

    x_sensors = np.linspace(domain[0], domain[1], n_sensors).astype(np.float32)

    # Train
    u_train = sample_grf(n_train, n_sensors, length_scale, domain)
    n_dt = int(np.clip(discontinuous_train, 0, n_train))
    if n_dt > 0:
        idx = np.random.choice(n_train, size=n_dt, replace=False)
        u_train[idx] = sample_segmented_grf_u(
            n_dt,
            n_sensors,
            length_scale,
            domain,
            seed=None if seed is None else int(seed) + 10_000,
            min_segments=2,
            max_segments=4,
        )
    y_train = np.random.uniform(domain[0], domain[1], (n_train, n_points_per_sample)).astype(np.float32)
    y_train = np.sort(y_train, axis=1)  # for stable integral
    s_train = solve_antiderivative(u_train, x_sensors, y_train)

    # Test
    u_test = sample_grf(n_test, n_sensors, length_scale, domain)
    n_dv = int(np.clip(discontinuous_test, 0, n_test))
    if n_dv > 0:
        idx_t = np.random.choice(n_test, size=n_dv, replace=False)
        u_test[idx_t] = sample_segmented_grf_u(
            n_dv,
            n_sensors,
            length_scale,
            domain,
            seed=None if seed is None else int(seed) + 20_000,
            min_segments=2,
            max_segments=4,
        )
    y_test = np.linspace(domain[0], domain[1], 100)
    y_test = np.broadcast_to(y_test, (n_test, 100)).astype(np.float32)
    s_test = solve_antiderivative(u_test, x_sensors, y_test)

    return {
        "u_train": u_train,
        "y_train": y_train,
        "s_train": s_train,
        "u_test": u_test,
        "y_test": y_test,
        "s_test": s_test,
        "x_sensors": x_sensors,
        "domain": {"min": [float(domain[0])], "max": [float(domain[1])]},
    }
