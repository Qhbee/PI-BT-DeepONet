"""PDE residual computation, dispatched by pde_type."""

import inspect

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


def _predict(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    sample: bool = True,
) -> torch.Tensor:
    """Run deterministic or Bayesian model and return predictive mean."""
    if _is_bayesian_model(model):
        pred, _, _, _ = model(u, y, sample=sample)
        return pred
    return model(u, y)


def _first_and_second_x_derivatives(
    pred: torch.Tensor,
    y_req: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return first derivatives (t,x) and second derivative x-x for 2D coords."""
    grad_pred = torch.autograd.grad(
        outputs=pred.sum(),
        inputs=y_req,
        create_graph=True,
    )[0]
    d_dt = grad_pred[..., 0]
    d_dx = grad_pred[..., 1]
    grad_dx = torch.autograd.grad(
        outputs=d_dx.sum(),
        inputs=y_req,
        create_graph=True,
    )[0]
    d_xx = grad_dx[..., 1]
    return d_dt, d_dx, d_xx


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

    pred = _predict(model, u, y_req, sample=sample)

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


def _burgers_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    x_sensors: torch.Tensor,
    nu: float = 0.01 / 3.141592653589793,
    sample: bool = True,
) -> torch.Tensor:
    """
    Burgers PDE:
        s_t + s*s_x - nu*s_xx = 0
    with y=(t,x).
    """
    y_req = y.detach().clone().requires_grad_(True)
    pred = _predict(model, u, y_req, sample=sample)
    d_dt, d_dx, d_xx = _first_and_second_x_derivatives(pred, y_req)
    residual = d_dt + pred * d_dx - nu * d_xx
    return (residual**2).mean()


def _diffusion_reaction_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    x_sensors: torch.Tensor,
    D: float = 0.01,
    k_reaction: float = 0.1,
    sample: bool = True,
) -> torch.Tensor:
    """
    Diffusion-reaction PDE:
        s_t - D*s_xx - k*s^2 - source(x) = 0
    with y=(t,x) and source interpolated from branch input u.
    """
    y_req = y.detach().clone().requires_grad_(True)
    pred = _predict(model, u, y_req, sample=sample)
    d_dt, _, d_xx = _first_and_second_x_derivatives(pred, y_req)
    x_query = y_req[..., 1]
    source_x = _interp1d_batched(x_sensors, u, x_query.detach())
    residual = d_dt - D * d_xx - k_reaction * (pred**2) - source_x
    return (residual**2).mean()


def _darcy_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    sample: bool = True,
) -> torch.Tensor:
    """
    Simplified Darcy residual:
        -k*(p_xx + p_yy) = 0, with scalar k estimated from branch input mean.
    """
    y_req = y.detach().clone().requires_grad_(True)
    pred = _predict(model, u, y_req, sample=sample)
    grad_pred = torch.autograd.grad(
        outputs=pred.sum(),
        inputs=y_req,
        create_graph=True,
    )[0]
    px = grad_pred[..., 0]
    py = grad_pred[..., 1]
    pxx = torch.autograd.grad(outputs=px.sum(), inputs=y_req, create_graph=True)[0][..., 0]
    pyy = torch.autograd.grad(outputs=py.sum(), inputs=y_req, create_graph=True)[0][..., 1]
    k_eff = u.mean(dim=-1, keepdim=True)
    if pxx.dim() == 2:
        k_eff = k_eff.expand(-1, pxx.shape[1])
    residual = -k_eff * (pxx + pyy)
    return (residual**2).mean()


def compute_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    x_sensors: torch.Tensor,
    pde_type: str,
    sample: bool = True,
    nu: float = 0.01 / 3.141592653589793,
    diffusion_D: float = 0.01,
    reaction_k: float = 0.1,
) -> torch.Tensor:
    """
    Compute PDE residual loss.

    Args:
        model: DeepONet or BayesianDeepONet
        u: (batch, num_sensors) input function at sensors
        y: (batch, n_colloc, coord_dim) collocation points
        x_sensors: (num_sensors,) or (coord_dim, num_sensors) sensor positions
        pde_type: "none" | "antiderivative" | "burgers" | "diffusion_reaction" | "darcy"
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
    if pde_type == "burgers":
        return _burgers_residual(model, u, y, x_sensors, nu=nu, sample=sample)
    if pde_type == "diffusion_reaction":
        return _diffusion_reaction_residual(
            model,
            u,
            y,
            x_sensors,
            D=diffusion_D,
            k_reaction=reaction_k,
            sample=sample,
        )
    if pde_type == "darcy":
        return _darcy_residual(model, u, y, sample=sample)
    raise ValueError(f"Unknown pde_type: {pde_type}")
