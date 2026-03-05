"""Branch network: encodes input function at sensor locations."""

import torch
import torch.nn as nn

from .transformer_block import PositionalEncoding, TransformerEncoder


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


class TransformerBranch(nn.Module):
    """Transformer-based Branch network. Maps [u(x1),...,u(xm)] -> [b1,...,bp] via self-attention."""

    def __init__(
        self,
        num_sensors: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.output_dim = output_dim
        self.d_model = d_model

        self.embed = nn.Linear(1, d_model)  # 每个 u(xi) 是标量，嵌入为 d_model 维
        self.pos_enc = PositionalEncoding(d_model, max_len=num_sensors, dropout=dropout)
        self.transformer = TransformerEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.pool_to_output = nn.Linear(d_model, output_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, num_sensors) input function values at sensors
        Returns:
            (batch, output_dim) branch output
        """
        # (batch, m) -> (batch, m, 1) -> (batch, m, d_model)
        x = self.embed(u.unsqueeze(-1))
        x = self.pos_enc(x)
        x = self.transformer(x)  # (batch, m, d_model)
        x = x.mean(dim=1)  # (batch, d_model) mean pooling
        return self.pool_to_output(x)
