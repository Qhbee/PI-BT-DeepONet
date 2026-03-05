"""Antiderivative operator: ds/dx = u(x), s(0)=0. Solution: s(x) = int_0^x u(tau) d tau."""

import numpy as np


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


def generate_antiderivative_data(
    n_train: int = 3000,
    n_test: int = 1000,
    n_sensors: int = 100,
    n_points_per_sample: int = 20,
    length_scale: float = 0.5,
    domain: tuple = (0.0, 1.0),
    seed: int | None = None,
):
    """Generate training and test data for antiderivative operator."""
    if seed is not None:
        np.random.seed(seed)

    x_sensors = np.linspace(domain[0], domain[1], n_sensors).astype(np.float32)

    # Train
    u_train = sample_grf(n_train, n_sensors, length_scale, domain)
    y_train = np.random.uniform(domain[0], domain[1], (n_train, n_points_per_sample)).astype(np.float32)
    y_train = np.sort(y_train, axis=1)  # for stable integral
    s_train = solve_antiderivative(u_train, x_sensors, y_train)

    # Test
    u_test = sample_grf(n_test, n_sensors, length_scale, domain)
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
    }
