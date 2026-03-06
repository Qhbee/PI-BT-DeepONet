"""DeepONet: G(u)(y) = b0 + sum(b_i * t_i)."""

import torch
import torch.nn as nn

from .branch import FNNBranch
from .trunk import FNNTrunk


class DeepONet(nn.Module):
    """Unstacked DeepONet: G(u)(y) = b0 + b(u)^T @ t(y)."""

    def __init__(
        self,
        branch: nn.Module,
        trunk: nn.Module,
        output_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.output_dim = output_dim
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            u: (batch, num_sensors) input function at sensors
            y: (batch, coord_dim) or (batch, n_points, coord_dim) query points
        Returns:
            (batch,) or (batch, n_points) G(u)(y)
        """
        b = self.branch(u)  # (batch, p)
        t = self.trunk(y)   # (batch, p) or (batch, n_points, p)

        if t.dim() == 2:
            out = (b * t).sum(dim=-1)
        else:
            out = (b.unsqueeze(1) * t).sum(dim=-1)  # (batch, n_points)
            if out.size(1) == 1:
                out = out.squeeze(1)

        if self.bias is not None:
            out = out + self.bias
        return out


class MultiOutputDeepONet(nn.Module):
    """Method-2 multi-output DeepONet with grouped dot products."""

    def __init__(
        self,
        branch: nn.Module,
        trunk: nn.Module,
        n_outputs: int,
        p_group: int,
        bias: bool = True,
    ):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.n_outputs = n_outputs
        self.p_group = p_group
        self.output_dim = n_outputs * p_group
        self.bias = nn.Parameter(torch.zeros(n_outputs)) if bias else None

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, num_sensors) or (batch, m, input_channels)
            y: (batch, coord_dim) or (batch, n_points, coord_dim)
        Returns:
            (batch, n_outputs) or (batch, n_points, n_outputs)
        """
        b = self.branch(u)  # (batch, n_outputs*p_group)
        t = self.trunk(y)   # (batch, n_outputs*p_group) or (batch, n_points, n_outputs*p_group)

        batch = b.shape[0]
        b_group = b.reshape(batch, self.n_outputs, self.p_group)
        if t.dim() == 2:
            t_group = t.reshape(batch, self.n_outputs, self.p_group)
            out = (b_group * t_group).sum(dim=-1)
        else:
            n_points = t.shape[1]
            t_group = t.reshape(batch, n_points, self.n_outputs, self.p_group)
            out = (b_group.unsqueeze(1) * t_group).sum(dim=-1)
            if out.size(1) == 1:
                out = out.squeeze(1)

        if self.bias is not None:
            out = out + self.bias
        return out
