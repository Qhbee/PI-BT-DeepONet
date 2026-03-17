"""PDE residual computation, dispatched by pde_type."""

import inspect

import torch

from .ns_residual import beltrami_vp_residual, kovasznay_vp_residual, pressure_gauge_loss


def _interp2d_batched(
    x_sensors: torch.Tensor,
    u: torch.Tensor,
    y_query: torch.Tensor,
) -> torch.Tensor:
    """
    Batched 2D bilinear interpolation on a regular grid.
    x_sensors: (n_sensors, 2) grid points [x,y], u: (batch, n_sensors), y_query: (batch, n, 2).
    Grid assumed row-major: index = j*nx + i for (x_i, y_j).
    Returns: (batch, n) interpolated values.
    """
    if x_sensors.dim() != 2 or x_sensors.shape[1] != 2:
        raise ValueError("x_sensors must be (n_sensors, 2) for 2D interpolation.")
    x_vals = torch.unique(x_sensors[:, 0]).sort().values
    y_vals = torch.unique(x_sensors[:, 1]).sort().values
    nx, ny = x_vals.shape[0], y_vals.shape[0]
    device = u.device
    dtype = u.dtype
    x_vals = x_vals.to(device=device, dtype=dtype)
    y_vals = y_vals.to(device=device, dtype=dtype)

    xq = y_query[..., 0].clamp(x_vals[0], x_vals[-1])
    yq = y_query[..., 1].clamp(y_vals[0], y_vals[-1])

    idx_x = torch.searchsorted(x_vals, xq, right=False)
    idx_y = torch.searchsorted(y_vals, yq, right=False)
    idx_x = torch.clamp(idx_x, 1, nx - 1)
    idx_y = torch.clamp(idx_y, 1, ny - 1)

    i0, i1 = idx_x - 1, idx_x
    j0, j1 = idx_y - 1, idx_y
    x0, x1 = x_vals[i0], x_vals[i1]
    y0, y1 = y_vals[j0], y_vals[j1]
    wx = (xq - x0) / (x1 - x0 + 1e-8)
    wy = (yq - y0) / (y1 - y0 + 1e-8)

    flat_00 = j0 * nx + i0
    flat_10 = j0 * nx + i1
    flat_01 = j1 * nx + i0
    flat_11 = j1 * nx + i1

    u00 = u.gather(1, flat_00)
    u10 = u.gather(1, flat_10)
    u01 = u.gather(1, flat_01)
    u11 = u.gather(1, flat_11)

    return (1 - wx) * (1 - wy) * u00 + wx * (1 - wy) * u10 + (1 - wx) * wy * u01 + wx * wy * u11


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
    y_query = y_query.contiguous()
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
    out = model(u, y, sample=sample) if _has_sample_param(model) else model(u, y)
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


def _has_sample_param(model: torch.nn.Module) -> bool:
    """Check if model.forward has sample parameter."""
    sig = inspect.signature(model.forward)
    return "sample" in sig.parameters


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
        s_t - D*s_xx - k*s - source(x) = 0
    with y=(t,x) and source interpolated from branch input u.
    k<0 for stability (sink term).
    """
    y_req = y.detach().clone().requires_grad_(True)
    pred = _predict(model, u, y_req, sample=sample)
    d_dt, _, d_xx = _first_and_second_x_derivatives(pred, y_req)
    x_query = y_req[..., 1]
    source_x = _interp1d_batched(x_sensors, u, x_query.detach())
    residual = d_dt - D * d_xx - k_reaction * pred - source_x
    return (residual**2).mean()


def _poisson_2d_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    x_sensors: torch.Tensor,
    sample: bool = True,
) -> torch.Tensor:
    """
    2D Poisson: -∇²p = f on [0,1]², p=0 on boundary.
    Residual: -∇²p - f = 0, with f interpolated from u at query points.
    """
    y_req = y.detach().clone().requires_grad_(True)
    pred = _predict(model, u, y_req, sample=sample)
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
    grad_pred = torch.autograd.grad(
        outputs=pred.sum(),
        inputs=y_req,
        create_graph=True,
    )[0]
    px = grad_pred[..., 0]
    py = grad_pred[..., 1]
    pxx = torch.autograd.grad(outputs=px.sum(), inputs=y_req, create_graph=True)[0][..., 0]
    pyy = torch.autograd.grad(outputs=py.sum(), inputs=y_req, create_graph=True)[0][..., 1]
    laplacian = pxx + pyy
    f_at_query = _interp2d_batched(x_sensors, u, y_req.detach())
    residual = -laplacian - f_at_query
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


def _ns_kovasznay_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    params: torch.Tensor | None,
    nu_default: float = 1.0 / 40.0,
    sample: bool = True,
    pressure_gauge_weight: float = 0.0,
) -> torch.Tensor:
    y_req = y.detach().clone().requires_grad_(True)
    pred = _predict(model, u, y_req, sample=sample)
    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
    if pred.shape[-1] < 3:
        raise ValueError(f"ns_kovasznay requires at least 3 outputs (u,v,p), got {pred.shape[-1]}.")
    pred = pred[..., :3]

    if params is not None and params.shape[-1] >= 1:
        re = params[:, 0].to(pred.dtype)
    elif u.dim() == 2 and u.shape[-1] == 1:
        re = u[:, 0].to(pred.dtype)
    else:
        re = torch.full((u.shape[0],), 1.0 / max(nu_default, 1e-8), device=u.device, dtype=pred.dtype)
    re = torch.clamp(re, min=1e-6)
    nu = 1.0 / re

    loss = kovasznay_vp_residual(pred, y_req, nu=nu)
    if pressure_gauge_weight > 0:
        loss = loss + pressure_gauge_weight * pressure_gauge_loss(pred, mode="mean_zero")
    return loss


def _ns_beltrami_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    nu: float = 1.0,
    sample: bool = True,
    pressure_gauge_weight: float = 0.0,
) -> torch.Tensor:
    y_req = y.detach().clone().requires_grad_(True)
    pred = _predict(model, u, y_req, sample=sample)
    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
    if pred.shape[-1] < 4:
        raise ValueError(f"ns_beltrami requires at least 4 outputs (u,v,w,p), got {pred.shape[-1]}.")
    pred = pred[..., :4]
    loss = beltrami_vp_residual(pred, y_req, nu=nu)
    if pressure_gauge_weight > 0:
        loss = loss + pressure_gauge_weight * pressure_gauge_loss(pred, mode="mean_zero")
    return loss


def _diffusion_reaction_residual_s_pinn(
    model: torch.nn.Module,
    u: torch.Tensor,
    x_sensors: torch.Tensor,
    D: float,
    k_reaction: float,
    sample: bool,
    domain_min: torch.Tensor,
    domain_max: torch.Tensor,
    n_collocation: int,
    n_subdomains: int = 16,
) -> torch.Tensor:
    """
    S-PINN style: partition domain into subdomains, compute mean residual per subdomain.
    For 2D (t,x), use a grid of subdomains.
    """
    batch = u.shape[0]
    device = u.device
    dtype = u.dtype
    coord_dim = 2
    n_per_sub = max(4, n_collocation // n_subdomains)
    total_loss = 0.0
    count = 0
    # 4x4 grid for full mode; 2x2 for small n_collocation (ultra)
    grid = 2 if n_collocation <= 64 else 4
    nt, nx = grid, grid
    t_min, t_max = domain_min[0].item(), domain_max[0].item()
    x_min, x_max = domain_min[1].item(), domain_max[1].item()
    for it in range(nt):
        for ix in range(nx):
            t_low = t_min + (t_max - t_min) * it / nt
            t_high = t_min + (t_max - t_min) * (it + 1) / nt
            x_low = x_min + (x_max - x_min) * ix / nx
            x_high = x_min + (x_max - x_min) * (ix + 1) / nx
            t_c = torch.rand(batch, n_per_sub, 1, device=device, dtype=dtype) * (t_high - t_low) + t_low
            x_c = torch.rand(batch, n_per_sub, 1, device=device, dtype=dtype) * (x_high - x_low) + x_low
            y_sub = torch.cat([t_c, x_c], dim=-1)
            r = _diffusion_reaction_residual(model, u, y_sub, x_sensors, D=D, k_reaction=k_reaction, sample=sample)
            total_loss = total_loss + r
            count += 1
    return total_loss / max(1, count)


def compute_residual(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    x_sensors: torch.Tensor | None,
    pde_type: str,
    sample: bool = True,
    nu: float = 0.01 / 3.141592653589793,
    diffusion_D: float = 0.01,
    reaction_k: float = 0.1,
    params: torch.Tensor | None = None,
    ns_nu: float = 1.0 / 40.0,
    ns_beltrami_nu: float = 1.0,
    pressure_gauge_weight: float = 0.0,
    physics_mode: str = "standard_pi",
    domain_min: torch.Tensor | None = None,
    domain_max: torch.Tensor | None = None,
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
        physics_mode: "standard_pi" | "hard_bc_pi" | "s_pinn"

    Returns:
        scalar residual loss
    """
    if pde_type == "none":
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)

    if physics_mode == "s_pinn" and pde_type == "diffusion_reaction" and x_sensors is not None:
        if domain_min is None or domain_max is None:
            domain_min = torch.tensor([0.0, 0.0], device=u.device, dtype=u.dtype)
            domain_max = torch.tensor([1.0, 1.0], device=u.device, dtype=u.dtype)
        n_colloc = y.shape[1] if y.dim() == 3 else 64
        if x_sensors.dim() > 1:
            x_sensors = x_sensors[0]
        return _diffusion_reaction_residual_s_pinn(
            model, u, x_sensors, diffusion_D, reaction_k, sample,
            domain_min, domain_max, n_colloc, n_subdomains=16,
        )
    if pde_type == "antiderivative":
        if x_sensors is None:
            raise ValueError("x_sensors is required for antiderivative residual.")
        if x_sensors.dim() > 1:
            x_sensors = x_sensors[0]
        return _antiderivative_residual(model, u, y, x_sensors, sample=sample)
    if pde_type == "burgers":
        if x_sensors is None:
            raise ValueError("x_sensors is required for burgers residual.")
        return _burgers_residual(model, u, y, x_sensors, nu=nu, sample=sample)
    if pde_type == "diffusion_reaction":
        if x_sensors is None:
            raise ValueError("x_sensors is required for diffusion_reaction residual.")
        return _diffusion_reaction_residual(
            model,
            u,
            y,
            x_sensors,
            D=diffusion_D,
            k_reaction=reaction_k,
            sample=sample,
        )
    if pde_type == "poisson_2d":
        if x_sensors is None:
            raise ValueError("x_sensors is required for poisson_2d residual.")
        return _poisson_2d_residual(model, u, y, x_sensors, sample=sample)
    if pde_type == "darcy":
        return _darcy_residual(model, u, y, sample=sample)
    if pde_type == "ns_kovasznay":
        return _ns_kovasznay_residual(
            model=model,
            u=u,
            y=y,
            params=params,
            nu_default=ns_nu,
            sample=sample,
            pressure_gauge_weight=pressure_gauge_weight,
        )
    if pde_type == "ns_beltrami":
        return _ns_beltrami_residual(
            model=model,
            u=u,
            y=y,
            nu=ns_beltrami_nu,
            sample=sample,
            pressure_gauge_weight=pressure_gauge_weight,
        )
    raise ValueError(f"Unknown pde_type: {pde_type}")
