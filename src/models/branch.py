"""Branch network: encodes input function at sensor locations."""

import torch
import torch.nn as nn

from .transformer_block import PositionalEncoding, TransformerEncoder


def _to_sensor_tokens(
    u: torch.Tensor,
    input_channels: int,
    num_sensors: int,
    broadcast_params: bool = True,
) -> torch.Tensor:
    """
    Normalize branch input to token shape (batch, seq_len, input_channels).

    Supported inputs:
    - (batch, m) with input_channels=1
    - (batch, input_channels) parameter vectors (route A)
    - (batch, m, input_channels) sensor sequences (route B)
    """
    if u.dim() == 2:
        if input_channels == 1:
            # Classic scalar sensor sequence: (batch, m) -> (batch, m, 1)
            if u.shape[1] > 1:
                return u.unsqueeze(-1)
            # Scalar parameter route: (batch, 1) -> broadcast to pseudo sequence
            tokens = u.unsqueeze(1)  # (batch, 1, 1)
            if broadcast_params:
                return tokens.expand(-1, num_sensors, -1)
            return tokens

        # input_channels > 1: treat as parameter vector (batch, c)
        if u.shape[1] != input_channels:
            raise ValueError(
                f"Expected u shape (batch, {input_channels}) for parameter input, got {tuple(u.shape)}."
            )
        tokens = u.unsqueeze(1)  # (batch, 1, c)
        if broadcast_params:
            return tokens.expand(-1, num_sensors, -1)
        return tokens

    if u.dim() == 3:
        if u.shape[-1] != input_channels:
            raise ValueError(
                f"Expected u.shape[-1] == input_channels ({input_channels}), got {u.shape[-1]}."
            )
        return u

    raise ValueError(f"Unsupported branch input shape: {tuple(u.shape)}")


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
            u: (batch, num_sensors) or (batch, m, c) input values
        Returns:
            (batch, output_dim) branch output
        """
        if u.dim() == 3:
            u = u.reshape(u.shape[0], -1)
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
        input_channels: int = 1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.output_dim = output_dim
        self.d_model = d_model
        self.input_channels = input_channels

        self.embed = nn.Linear(input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_sensors, dropout=dropout)
        self.transformer = TransformerEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.pool_to_output = nn.Linear(d_model, output_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, m), (batch, input_channels), or (batch, m, input_channels)
        Returns:
            (batch, output_dim) branch output
        """
        tokens = _to_sensor_tokens(
            u,
            input_channels=self.input_channels,
            num_sensors=self.num_sensors,
            broadcast_params=True,
        )
        x = self.embed(tokens)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (batch, m, d_model)
        x = x.mean(dim=1)  # (batch, d_model) mean pooling
        return self.pool_to_output(x)


class TransformerMultiOutputBranch(TransformerBranch):
    """Hard-truncation baseline: one pooled token mapped to grouped outputs."""

    def __init__(
        self,
        num_sensors: int,
        n_outputs: int,
        p_group: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        input_channels: int = 1,
    ):
        self.n_outputs = n_outputs
        self.p_group = p_group
        super().__init__(
            num_sensors=num_sensors,
            output_dim=n_outputs * p_group,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            input_channels=input_channels,
        )


class TransformerMultiCLSBranch(nn.Module):
    """Multi-CLS branch: one learnable CLS token per output group."""

    def __init__(
        self,
        num_sensors: int,
        n_outputs: int,
        p_group: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        input_channels: int = 1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.n_outputs = n_outputs
        self.p_group = p_group
        self.input_channels = input_channels
        self.output_dim = n_outputs * p_group

        self.embed = nn.Linear(input_channels, d_model)
        self.cls_tokens = nn.Parameter(torch.zeros(n_outputs, d_model))
        nn.init.normal_(self.cls_tokens, mean=0.0, std=0.02)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_sensors + n_outputs, dropout=dropout)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.group_heads = nn.ModuleList([nn.Linear(d_model, p_group) for _ in range(n_outputs)])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        tokens = _to_sensor_tokens(
            u,
            input_channels=self.input_channels,
            num_sensors=self.num_sensors,
            broadcast_params=True,
        )
        sensor_tokens = self.embed(tokens)  # (batch, m, d_model)
        batch = sensor_tokens.shape[0]
        cls = self.cls_tokens.unsqueeze(0).expand(batch, -1, -1)  # (batch, n_outputs, d_model)
        x = torch.cat([cls, sensor_tokens], dim=1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        cls_out = x[:, : self.n_outputs, :]
        groups = [head(cls_out[:, i, :]) for i, head in enumerate(self.group_heads)]
        return torch.cat(groups, dim=-1)
