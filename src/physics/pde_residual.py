"""PDE residual computation, dispatched by pde_type."""

import torch


def _interp1d_batched(
    x_sensors: torch.Tensor,
    u: torch.Tensor,
    y_query: torch.Tensor,
) -> torch.Tensor:
    """
    Batched 1D linear interpolation: u at x_sensors, query at y_query.

    Args:
        x_sensors: (m,) sorted
        u: (batch, m)
        y_query: (batch, n) or (batch, n, 1)

    Returns:
        u_at_y: (batch, n) interpolated values
    """
    if y_query.dim() == 3:
        y_query = y_query.squeeze(-1)
    batch, n = y_query.shape
    m = x_sensors.shape[0]
    device = u.device
    x_sensors = x_sensors.to(device=device, dtype=u.dtype)

    idx_right = torch.searchsorted(x_sensors, y_query, right=False)
    idx_right = torch.clamp(idx_right, 1, m - 1)
    idx_left = idx_right - 1

    x_left = x_sensors[idx_left]
    x_right = x_sensors[idx_right]
    u_left = u.gather(1, idx_left)
    u_right = u.gather(1, idx_right)

    t = (y_query - x_left) / (x_right - x_left + 1e-8)
    u_interp = u_left + t * (u_right - u_left)
    return u_interp


def _antiderivative_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    x_sensors: torch.Tensor,
    sample: bool = True,
) -> torch.Tensor:
    """
    Antiderivative ODE: ds/dy = u(y). Residual R = (dG/dy - u(y))^2.
    dG/dy is computed via autograd.

    Args:
        model: DeepONet or BayesianDeepONet
        u: (batch, num_sensors)
        y: (batch, n_colloc, 1) collocation points
        x_sensors: (num_sensors,) sensor positions
    """
    y_req = y.detach().clone().requires_grad_(True)

    is_bayesian = _is_bayesian_model(model)
    if is_bayesian:
        pred, _, _, _ = model(u, y_req, sample=sample)
    else:
        pred = model(u, y_req)

    dG_dy = torch.autograd.grad(
        outputs=pred.sum(),
        inputs=y_req,
        create_graph=True,
    )[0]
    if y_req.dim() == 3 and dG_dy.dim() == 3 and dG_dy.size(-1) == 1:
        dG_dy = dG_dy.squeeze(-1)

    # Interpolation term is treated as a fixed target.
    u_at_y = _interp1d_batched(x_sensors, u, y_req.detach())
    residual = (dG_dy - u_at_y) ** 2
    return residual.mean()


def _is_bayesian_model(model: torch.nn.Module) -> bool:
    """Check if model has Bayesian-style forward (returns 4-tuple)."""
    import inspect
    sig = inspect.signature(model.forward)
    return "sample" in sig.parameters


def compute_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    x_sensors: torch.Tensor,
    pde_type: str,
    sample: bool = True,
) -> torch.Tensor:
    """
    Compute PDE residual loss.

    Args:
        model: DeepONet or BayesianDeepONet
        u: (batch, num_sensors) input function at sensors
        y: (batch, n_colloc, coord_dim) collocation points
        x_sensors: (num_sensors,) or (coord_dim, num_sensors) sensor positions
        pde_type: "none" | "antiderivative"
        sample: for Bayesian models, whether to sample weights

    Returns:
        scalar residual loss
    """
    if pde_type == "none":
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    if pde_type == "antiderivative":
        if x_sensors.dim() > 1:
            x_sensors = x_sensors[0]
        return _antiderivative_residual(model, u, y, x_sensors, sample=sample)
    raise ValueError(f"Unknown pde_type: {pde_type}")
