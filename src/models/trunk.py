"""Trunk network: encodes output domain coordinates."""

import torch
import torch.nn as nn


class FNNTrunk(nn.Module):
    """FNN-based Trunk network. Maps y -> [t1,...,tp]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 1:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, input_dim) or (batch, n_points, input_dim) coordinates
        Returns:
            (batch, output_dim) or (batch, n_points, output_dim) trunk output
        """
        if y.dim() == 2:
            return self.net(y)
        batch, n_points, _ = y.shape
        y_flat = y.reshape(-1, self.input_dim)
        t_flat = self.net(y_flat)
        return t_flat.reshape(batch, n_points, self.output_dim)
