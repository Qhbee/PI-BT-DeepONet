"""Branch network: encodes input function at sensor locations."""

import torch
import torch.nn as nn


class FNNBranch(nn.Module):
    """FNN-based Branch network. Maps [u(x1),...,u(xm)] -> [b1,...,bp]."""

    def __init__(
        self,
        num_sensors: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.output_dim = output_dim

        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers = []
        dims = [num_sensors] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, num_sensors) input function values at sensors
        Returns:
            (batch, output_dim) branch output
        """
        return self.net(u)
