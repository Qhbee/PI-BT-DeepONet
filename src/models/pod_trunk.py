"""POD-based trunk: FNN-style (learned) and FixedPODTrunk (fixed basis + mean field)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _interp2d_grid(
    t: torch.Tensor,
    x: torch.Tensor,
    t_grid: torch.Tensor,
    x_grid: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Bilinear interpolation. values: (nt, nx) or (nt, nx, rank).
    t, x: (N,). Returns (N,) or (N, rank).
    """
    t_min, t_max = t_grid[0].item(), t_grid[-1].item()
    x_min, x_max = x_grid[0].item(), x_grid[-1].item()
    nt, nx = t_grid.shape[0], x_grid.shape[0]
    if nt < 2 or nx < 2:
        i0 = torch.zeros_like(t, dtype=torch.long, device=t.device)
        j0 = torch.zeros_like(x, dtype=torch.long, device=x.device)
        return values[i0, j0]
    ti = (t - t_min) / (t_max - t_min + 1e-8) * (nt - 1)
    xi = (x - x_min) / (x_max - x_min + 1e-8) * (nx - 1)
    ti = ti.clamp(0, nt - 1.001)
    xi = xi.clamp(0, nx - 1.001)
    i0 = ti.long().clamp(0, nt - 2)
    j0 = xi.long().clamp(0, nx - 2)
    i1, j1 = i0 + 1, j0 + 1
    wt = ti - i0.float()
    wx = xi - j0.float()
    v00 = values[i0, j0]
    v01 = values[i0, j1]
    v10 = values[i1, j0]
    v11 = values[i1, j1]
    if values.dim() == 2:
        v0 = v00 * (1 - wx) + v01 * wx
        v1 = v10 * (1 - wx) + v11 * wx
        return v0 * (1 - wt) + v1 * wt
    v0 = v00 * (1 - wx).unsqueeze(-1) + v01 * wx.unsqueeze(-1)
    v1 = v10 * (1 - wx).unsqueeze(-1) + v11 * wx.unsqueeze(-1)
    return v0 * (1 - wt).unsqueeze(-1) + v1 * wt.unsqueeze(-1)


class FixedPODTrunk(nn.Module):
    """
    Fixed POD basis trunk: output = phi(y) at query points.
    Loads basis (n_points, rank) and mean_field from npz.
    Uses bilinear interpolation for arbitrary (t,x).
    """

    def __init__(
        self,
        pod_path: str | Path,
        coord_dim: int = 2,
    ):
        super().__init__()
        data = np.load(pod_path, allow_pickle=False)
        basis = data["basis"]  # (n_points, rank)
        mean_field = data["mean_field"]  # (n_points,)
        t_grid = data["t_grid"]
        x_grid = data["x_grid"]
        n_points = basis.shape[0]
        rank = basis.shape[1]
        nt = len(t_grid)
        nx = len(x_grid)
        assert nt * nx == n_points
        self.basis = nn.Parameter(torch.from_numpy(basis).float(), requires_grad=False)
        self.mean_field = nn.Parameter(torch.from_numpy(mean_field).float(), requires_grad=False)
        self.basis_2d = self.basis.view(nt, nx, rank)
        self.mean_2d = self.mean_field.view(nt, nx)
        self.register_buffer("t_grid", torch.from_numpy(t_grid).float())
        self.register_buffer("x_grid", torch.from_numpy(x_grid).float())
        self.output_dim = rank
        self.input_dim = coord_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (batch, 2) or (batch, n_points, 2) with y[...,0]=t, y[...,1]=x
        Returns: (batch, rank) or (batch, n_points, rank) - basis values phi(y)
        """
        t_grid = self.t_grid.to(y.device)
        x_grid = self.x_grid.to(y.device)
        basis_2d = self.basis_2d.to(y.device)
        if y.dim() == 2:
            t, x = y[:, 0], y[:, 1]
            phi = _interp2d_grid(t, x, t_grid, x_grid, basis_2d)
            return phi
        batch, n_pts, _ = y.shape
        t = y[..., 0].reshape(-1)
        x = y[..., 1].reshape(-1)
        phi = _interp2d_grid(t, x, t_grid, x_grid, basis_2d)
        return phi.reshape(batch, n_pts, self.output_dim)

    def get_mean_at_y(self, y: torch.Tensor) -> torch.Tensor:
        """Return mean field interpolated at y. Shape (batch,) or (batch, n_points)."""
        t_grid = self.t_grid.to(y.device)
        x_grid = self.x_grid.to(y.device)
        mean_2d = self.mean_2d.to(y.device)
        if y.dim() == 2:
            t, x = y[:, 0], y[:, 1]
            return _interp2d_grid(t, x, t_grid, x_grid, mean_2d)
        batch, n_pts, _ = y.shape
        t = y[..., 0].reshape(-1)
        x = y[..., 1].reshape(-1)
        m = _interp2d_grid(t, x, t_grid, x_grid, mean_2d)
        return m.reshape(batch, n_pts)


class PODTrunk(nn.Module):
    """
    Lightweight POD trunk.

    Given coordinate input y, this module predicts coefficients onto a fixed POD basis
    (one basis per output component can be concatenated by the caller).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        act = nn.ReLU if activation == "relu" else nn.Tanh
        dims = [input_dim] + hidden_dims + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 2:
            return self.net(y)
        batch, n_points, _ = y.shape
        y_flat = y.reshape(-1, self.input_dim)
        out = self.net(y_flat)
        return out.reshape(batch, n_points, self.output_dim)
