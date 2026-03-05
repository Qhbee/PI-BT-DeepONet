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
