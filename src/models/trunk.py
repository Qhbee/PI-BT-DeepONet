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


class ExFNNTrunk(nn.Module):
    """Paper-style Ex-DeepONet trunk (Eq.9, Li et al. 2025).

    Hidden layers:  s_k = σ(c_k * (W·s_{k-1} + b)),  k = 2 .. m-1
    Output layer:   s_m = Σ c_m * s_{m-1}             (no Linear, no activation)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("ExFNNTrunk requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 1
        self.coeff_dim = sum(hidden_dims) + hidden_dims[-1]
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def _split_coeffs(self, coeffs: torch.Tensor) -> list[torch.Tensor]:
        sizes = self.hidden_dims + [self.hidden_dims[-1]]
        return list(torch.split(coeffs, sizes, dim=-1))

    def forward(self, y: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, input_dim) or (batch, n_points, input_dim)
            coeffs: (batch, coeff_dim) branch-generated layer coefficients
        Returns:
            (batch,) or (batch, n_points) model output
        """
        coeff_groups = self._split_coeffs(coeffs)
        c_hidden = coeff_groups[:-1]
        c_out = coeff_groups[-1]

        if y.dim() == 2:
            x = y
            for layer, ck in zip(self.layers, c_hidden):
                x = self.activation(layer(x) * ck)
            return (x * c_out).sum(dim=-1)

        batch, n_points, _ = y.shape
        x = y.reshape(-1, self.input_dim)
        for layer, ck in zip(self.layers, c_hidden):
            ck_exp = ck.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, ck.shape[-1])
            x = self.activation(layer(x) * ck_exp)
        c_out_exp = c_out.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, c_out.shape[-1])
        out = (x * c_out_exp).sum(dim=-1).reshape(batch, n_points)
        if out.size(1) == 1:
            out = out.squeeze(1)
        return out


class ExV2FNNTrunk(nn.Module):
    """Input-modulated Ex-DeepONet trunk with external dot product.

    s_0 = y,  c_0 = 1 (no modulation on first layer input)
    s_k = σ(W_{k-1} · (c_{k-1} · s_{k-1}) + b_{k-1}),  k = 1 .. m
    Output: <c_m, s_m>  (all layers have activation; final dot product is external)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("ExV2FNNTrunk requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 1
        self.coeff_dim = sum(hidden_dims)
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def _split_coeffs(self, coeffs: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(coeffs, self.hidden_dims, dim=-1))

    def forward(self, y: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, input_dim) or (batch, n_points, input_dim)
            coeffs: (batch, coeff_dim) branch-generated coefficients
        Returns:
            (batch,) or (batch, n_points) model output
        """
        coeff_groups = self._split_coeffs(coeffs)
        c_modulate = coeff_groups[:-1]
        c_out = coeff_groups[-1]

        if y.dim() == 2:
            x = y
            for i, layer in enumerate(self.layers):
                if i > 0:
                    x = x * c_modulate[i - 1]
                x = self.activation(layer(x))
            return (x * c_out).sum(dim=-1)

        batch, n_points, _ = y.shape
        x = y.reshape(-1, self.input_dim)
        for i, layer in enumerate(self.layers):
            if i > 0:
                ck = c_modulate[i - 1]
                ck_exp = ck.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, ck.shape[-1])
                x = x * ck_exp
            x = self.activation(layer(x))
        c_out_exp = c_out.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, c_out.shape[-1])
        out = (x * c_out_exp).sum(dim=-1).reshape(batch, n_points)
        if out.size(1) == 1:
            out = out.squeeze(1)
        return out
