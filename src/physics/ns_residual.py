"""Navier-Stokes residuals and analytical references for Kovasznay/Beltrami."""

from __future__ import annotations

import torch


def _grad_scalar(field: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(field.sum(), y, create_graph=True)[0]


def _second_along(grad_field: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.autograd.grad(grad_field.sum(), y, create_graph=True)[0][..., dim]


def kovasznay_solution(
    y: torch.Tensor,
    re: torch.Tensor,
) -> torch.Tensor:
    """
    Analytical Kovasznay solution.

    Args:
        y: (batch, n_points, 2) with coords (x, y)
        re: (batch,) Reynolds number
    Returns:
        (batch, n_points, 3) -> (u, v, p)
    """
    x = y[..., 0]
    y_coord = y[..., 1]
    re = re.view(-1, 1).to(y.dtype)
    zeta = 0.5 * re - torch.sqrt(0.25 * (re**2) + 4.0 * (torch.pi**2))
    zeta = zeta.expand_as(x)

    u = 1.0 - torch.exp(zeta * x) * torch.cos(2.0 * torch.pi * y_coord)
    v = (zeta / (2.0 * torch.pi)) * torch.exp(zeta * x) * torch.sin(2.0 * torch.pi * y_coord)
    p = 0.5 * (1.0 - torch.exp(2.0 * zeta * x))
    return torch.stack([u, v, p], dim=-1)


def beltrami_solution(
    y: torch.Tensor,
    a: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """
    Analytical Beltrami flow solution.

    Args:
        y: (batch, n_points, 4) with coords (t, x, y, z)
        a: (batch,)
        d: (batch,)
    Returns:
        (batch, n_points, 4) -> (u, v, w, p)
    """
    t = y[..., 0]
    x = y[..., 1]
    y_coord = y[..., 2]
    z = y[..., 3]

    a = a.view(-1, 1).to(y.dtype)
    d = d.view(-1, 1).to(y.dtype)
    a_e = a.expand_as(x)
    d_e = d.expand_as(x)

    exp_decay = torch.exp(-(d_e**2) * t)
    u = -a_e * (
        torch.exp(a_e * x) * torch.sin(a_e * y_coord + d_e * z)
        + torch.exp(a_e * z) * torch.cos(a_e * x + d_e * y_coord)
    ) * exp_decay
    v = -a_e * (
        torch.exp(a_e * y_coord) * torch.sin(a_e * z + d_e * x)
        + torch.exp(a_e * x) * torch.cos(a_e * y_coord + d_e * z)
    ) * exp_decay
    w = -a_e * (
        torch.exp(a_e * z) * torch.sin(a_e * x + d_e * y_coord)
        + torch.exp(a_e * y_coord) * torch.cos(a_e * z + d_e * x)
    ) * exp_decay

    p = -0.5 * (a_e**2) * (
        torch.exp(2.0 * a_e * x)
        + torch.exp(2.0 * a_e * y_coord)
        + torch.exp(2.0 * a_e * z)
        + 2.0
        * torch.sin(a_e * x + d_e * y_coord)
        * torch.cos(a_e * z + d_e * x)
        * torch.exp(a_e * (y_coord + z))
        + 2.0
        * torch.sin(a_e * y_coord + d_e * z)
        * torch.cos(a_e * x + d_e * y_coord)
        * torch.exp(a_e * (z + x))
        + 2.0
        * torch.sin(a_e * z + d_e * x)
        * torch.cos(a_e * y_coord + d_e * z)
        * torch.exp(a_e * (x + y_coord))
    ) * torch.exp(-2.0 * (d_e**2) * t)
    return torch.stack([u, v, w, p], dim=-1)


def kovasznay_vp_residual(
    pred: torch.Tensor,
    y_req: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """
    Steady 2D incompressible NS residual:
        u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
        u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
        u_x + v_y = 0
    """
    u = pred[..., 0]
    v = pred[..., 1]
    p = pred[..., 2]

    gu = _grad_scalar(u, y_req)
    gv = _grad_scalar(v, y_req)
    gp = _grad_scalar(p, y_req)
    ux, uy = gu[..., 0], gu[..., 1]
    vx, vy = gv[..., 0], gv[..., 1]
    px, py = gp[..., 0], gp[..., 1]

    uxx = _second_along(ux, y_req, 0)
    uyy = _second_along(uy, y_req, 1)
    vxx = _second_along(vx, y_req, 0)
    vyy = _second_along(vy, y_req, 1)

    if nu.dim() == 1:
        nu_eff = nu.view(-1, 1).expand_as(ux)
    else:
        nu_eff = nu
    r1 = u * ux + v * uy + px - nu_eff * (uxx + uyy)
    r2 = u * vx + v * vy + py - nu_eff * (vxx + vyy)
    r3 = ux + vy
    return (r1.pow(2).mean() + r2.pow(2).mean() + r3.pow(2).mean()) / 3.0


def beltrami_vp_residual(
    pred: torch.Tensor,
    y_req: torch.Tensor,
    nu: float = 1.0,
) -> torch.Tensor:
    """
    Unsteady 3D incompressible NS residual with coords (t, x, y, z).
    """
    u = pred[..., 0]
    v = pred[..., 1]
    w = pred[..., 2]
    p = pred[..., 3]

    gu = _grad_scalar(u, y_req)
    gv = _grad_scalar(v, y_req)
    gw = _grad_scalar(w, y_req)
    gp = _grad_scalar(p, y_req)

    ut, ux, uy, uz = gu[..., 0], gu[..., 1], gu[..., 2], gu[..., 3]
    vt, vx, vy, vz = gv[..., 0], gv[..., 1], gv[..., 2], gv[..., 3]
    wt, wx, wy, wz = gw[..., 0], gw[..., 1], gw[..., 2], gw[..., 3]
    px, py, pz = gp[..., 1], gp[..., 2], gp[..., 3]

    uxx = _second_along(ux, y_req, 1)
    uyy = _second_along(uy, y_req, 2)
    uzz = _second_along(uz, y_req, 3)
    vxx = _second_along(vx, y_req, 1)
    vyy = _second_along(vy, y_req, 2)
    vzz = _second_along(vz, y_req, 3)
    wxx = _second_along(wx, y_req, 1)
    wyy = _second_along(wy, y_req, 2)
    wzz = _second_along(wz, y_req, 3)

    r1 = ut + u * ux + v * uy + w * uz + px - nu * (uxx + uyy + uzz)
    r2 = vt + u * vx + v * vy + w * vz + py - nu * (vxx + vyy + vzz)
    r3 = wt + u * wx + v * wy + w * wz + pz - nu * (wxx + wyy + wzz)
    r4 = ux + vy + wz
    return (r1.pow(2).mean() + r2.pow(2).mean() + r3.pow(2).mean() + r4.pow(2).mean()) / 4.0


def pressure_gauge_loss(pred: torch.Tensor, mode: str = "mean_zero") -> torch.Tensor:
    """Simple pressure gauge regularization to remove constant-shift ambiguity."""
    p = pred[..., -1]
    if mode == "mean_zero":
        return p.mean(dim=-1).pow(2).mean()
    if mode == "anchor_zero":
        if p.dim() == 1:
            return p[0].pow(2)
        return p[..., 0].pow(2).mean()
    raise ValueError(f"Unknown pressure gauge mode: {mode}")
