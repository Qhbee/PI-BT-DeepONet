"""Bayesian modules for alpha-VI / VB DeepONet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import PositionalEncoding


def _log_gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Element-wise log-density of Gaussian N(mu, sigma^2)."""
    return -0.5 * math.log(2 * math.pi) - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma**2)


class BayesianLinear(nn.Module):
    """Bayesian linear layer with diagonal Gaussian posterior."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.02, 0.02))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-0.02, 0.02))
            self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        # 数据依赖先验 N(μ_pretrained, σ)：None 时用 N(0, σ)
        self.register_buffer("prior_weight_mean", None)
        self.register_buffer("prior_bias_mean", None)

    @staticmethod
    def _sigma_from_rho(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def _sample_weights(self, sample: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not sample:
            return self.weight_mu, self.bias_mu

        weight_sigma = self._sigma_from_rho(self.weight_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)

        if self.bias_mu is None:
            return weight, None

        bias_sigma = self._sigma_from_rho(self.bias_rho)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        return weight, bias

    def _log_prior_and_variational(
        self, weight: torch.Tensor, bias: torch.Tensor | None, sample: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sample:
            zero = torch.zeros((), device=weight.device, dtype=weight.dtype)
            return zero, zero

        prior_sigma = torch.full_like(weight, self.prior_sigma)
        w_prior_center = self.prior_weight_mean if self.prior_weight_mean is not None else torch.zeros_like(weight)
        log_prior = _log_gaussian(weight, w_prior_center, prior_sigma).sum()
        weight_sigma = self._sigma_from_rho(self.weight_rho)
        log_variational = _log_gaussian(weight, self.weight_mu, weight_sigma).sum()

        if bias is not None and self.bias_mu is not None and self.bias_rho is not None:
            bias_prior_sigma = torch.full_like(bias, self.prior_sigma)
            b_prior_center = self.prior_bias_mean if self.prior_bias_mean is not None else torch.zeros_like(bias)
            log_prior = log_prior + _log_gaussian(bias, b_prior_center, bias_prior_sigma).sum()
            bias_sigma = self._sigma_from_rho(self.bias_rho)
            log_variational = log_variational + _log_gaussian(bias, self.bias_mu, bias_sigma).sum()

        return log_prior, log_variational

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight, bias = self._sample_weights(sample=sample)
        out = F.linear(x, weight, bias)
        log_prior, log_variational = self._log_prior_and_variational(weight, bias, sample=sample)
        return out, log_prior, log_variational


class BayesianMLP(nn.Module):
    """MLP built from BayesianLinear layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        prior_sigma: float = 1.0,
        activation: str = "relu",
        activate_last: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activate_last = activate_last
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList(
            [BayesianLinear(dims[i], dims[i + 1], prior_sigma=prior_sigma) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_sequence = x.dim() == 3
        if is_sequence:
            batch, n_points, _ = x.shape
            x = x.reshape(-1, self.input_dim)

        log_prior_total = torch.zeros((), device=x.device, dtype=x.dtype)
        log_variational_total = torch.zeros((), device=x.device, dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            x, log_prior, log_variational = layer(x, sample=sample)
            log_prior_total = log_prior_total + log_prior
            log_variational_total = log_variational_total + log_variational
            if i < len(self.layers) - 1 or self.activate_last:
                x = self.activation(x)

        if is_sequence:
            x = x.reshape(batch, n_points, self.output_dim)

        return x, log_prior_total, log_variational_total


class BayesianFNNBranch(nn.Module):
    """Bayesian FNN branch."""

    def __init__(self, num_sensors: int, hidden_dims: list[int], output_dim: int, prior_sigma: float = 1.0):
        super().__init__()
        self.net = BayesianMLP(
            input_dim=num_sensors,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            prior_sigma=prior_sigma,
            activate_last=False,
        )

    def forward(self, u: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(u, sample=sample)


class BayesianFNNTrunk(nn.Module):
    """Bayesian FNN trunk."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, prior_sigma: float = 1.0):
        super().__init__()
        self.net = BayesianMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            prior_sigma=prior_sigma,
            activate_last=True,
        )

    def forward(self, y: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(y, sample=sample)


class BayesianExFNNTrunk(nn.Module):
    """Paper-style Bayesian Ex-DeepONet trunk (Eq.9, Li et al. 2025).

    Hidden layers:  s_k = σ(c_k * (W·s_{k-1} + b)),  k = 2 .. m-1
    Output layer:   s_m = Σ c_m * s_{m-1}             (no Linear, no activation)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        prior_sigma: float = 1.0,
        activation: str = "relu",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("BayesianExFNNTrunk requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 1
        self.coeff_dim = sum(hidden_dims) + hidden_dims[-1]
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [BayesianLinear(dims[i], dims[i + 1], prior_sigma=prior_sigma) for i in range(len(dims) - 1)]
        )

    def _split_coeffs(self, coeffs: torch.Tensor) -> list[torch.Tensor]:
        sizes = self.hidden_dims + [self.hidden_dims[-1]]
        return list(torch.split(coeffs, sizes, dim=-1))

    def forward(
        self,
        y: torch.Tensor,
        coeffs: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coeff_groups = self._split_coeffs(coeffs)
        c_hidden = coeff_groups[:-1]
        c_out = coeff_groups[-1]
        log_prior_total = torch.zeros((), device=coeffs.device, dtype=coeffs.dtype)
        log_variational_total = torch.zeros((), device=coeffs.device, dtype=coeffs.dtype)

        if y.dim() == 2:
            x = y
            for layer, ck in zip(self.layers, c_hidden):
                x, lp, lq = layer(x, sample=sample)
                x = self.activation(x * ck)
                log_prior_total = log_prior_total + lp
                log_variational_total = log_variational_total + lq
            return (x * c_out).sum(dim=-1), log_prior_total, log_variational_total

        batch, n_points, _ = y.shape
        x = y.reshape(-1, self.input_dim)
        for layer, ck in zip(self.layers, c_hidden):
            x, lp, lq = layer(x, sample=sample)
            ck_exp = ck.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, ck.shape[-1])
            x = self.activation(x * ck_exp)
            log_prior_total = log_prior_total + lp
            log_variational_total = log_variational_total + lq
        c_out_exp = c_out.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, c_out.shape[-1])
        out = (x * c_out_exp).sum(dim=-1).reshape(batch, n_points)
        if out.size(1) == 1:
            out = out.squeeze(1)
        return out, log_prior_total, log_variational_total


class BayesianExV2FNNTrunk(nn.Module):
    """Bayesian input-modulated Ex-DeepONet trunk with external dot product.

    s_0 = y,  c_0 = 1 (no modulation on first layer input)
    s_k = σ(W_{k-1} · (c_{k-1} · s_{k-1}) + b_{k-1}),  k = 1 .. m
    Output: <c_m, s_m>  (all layers have activation; final dot product is external)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        prior_sigma: float = 1.0,
        activation: str = "relu",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("BayesianExV2FNNTrunk requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 1
        self.coeff_dim = sum(hidden_dims)
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [BayesianLinear(dims[i], dims[i + 1], prior_sigma=prior_sigma) for i in range(len(dims) - 1)]
        )

    def _split_coeffs(self, coeffs: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(coeffs, self.hidden_dims, dim=-1))

    def forward(
        self,
        y: torch.Tensor,
        coeffs: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coeff_groups = self._split_coeffs(coeffs)
        c_modulate = coeff_groups[:-1]
        c_out = coeff_groups[-1]
        log_prior_total = torch.zeros((), device=coeffs.device, dtype=coeffs.dtype)
        log_variational_total = torch.zeros((), device=coeffs.device, dtype=coeffs.dtype)

        if y.dim() == 2:
            x = y
            for i, layer in enumerate(self.layers):
                if i > 0:
                    x = x * c_modulate[i - 1]
                x, lp, lq = layer(x, sample=sample)
                x = self.activation(x)
                log_prior_total = log_prior_total + lp
                log_variational_total = log_variational_total + lq
            return (x * c_out).sum(dim=-1), log_prior_total, log_variational_total

        batch, n_points, _ = y.shape
        x = y.reshape(-1, self.input_dim)
        for i, layer in enumerate(self.layers):
            if i > 0:
                ck = c_modulate[i - 1]
                ck_exp = ck.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, ck.shape[-1])
                x = x * ck_exp
            x, lp, lq = layer(x, sample=sample)
            x = self.activation(x)
            log_prior_total = log_prior_total + lp
            log_variational_total = log_variational_total + lq
        c_out_exp = c_out.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, c_out.shape[-1])
        out = (x * c_out_exp).sum(dim=-1).reshape(batch, n_points)
        if out.size(1) == 1:
            out = out.squeeze(1)
        return out, log_prior_total, log_variational_total


class BayesianMultiHeadSelfAttention(nn.Module):
    """Bayesian multi-head self-attention."""

    def __init__(self, d_model: int, nhead: int, prior_sigma: float = 1.0, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = BayesianLinear(d_model, d_model, prior_sigma=prior_sigma)
        self.k_proj = BayesianLinear(d_model, d_model, prior_sigma=prior_sigma)
        self.v_proj = BayesianLinear(d_model, d_model, prior_sigma=prior_sigma)
        self.out_proj = BayesianLinear(d_model, d_model, prior_sigma=prior_sigma)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q, lp_q, lq_q = self.q_proj(x, sample=sample)
        k, lp_k, lq_k = self.k_proj(x, sample=sample)
        v, lp_v, lq_v = self.v_proj(x, sample=sample)

        q = q.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = self.attn_dropout(torch.softmax(attn_scores, dim=-1))
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        out, lp_o, lq_o = self.out_proj(context, sample=sample)
        log_prior = lp_q + lp_k + lp_v + lp_o
        log_variational = lq_q + lq_k + lq_v + lq_o
        return out, log_prior, log_variational


class BayesianTransformerEncoderLayer(nn.Module):
    """Bayesian Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        prior_sigma: float = 1.0,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or (d_model * 4)
        self.self_attn = BayesianMultiHeadSelfAttention(
            d_model=d_model, nhead=nhead, prior_sigma=prior_sigma, dropout=dropout
        )
        self.linear1 = BayesianLinear(d_model, dim_feedforward, prior_sigma=prior_sigma)
        self.linear2 = BayesianLinear(dim_feedforward, d_model, prior_sigma=prior_sigma)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prior_total = torch.zeros((), device=x.device, dtype=x.dtype)
        log_variational_total = torch.zeros((), device=x.device, dtype=x.dtype)

        attn_in = self.norm1(x)
        attn_out, lp_attn, lq_attn = self.self_attn(attn_in, sample=sample)
        x = x + self.dropout1(attn_out)
        log_prior_total = log_prior_total + lp_attn
        log_variational_total = log_variational_total + lq_attn

        ff_in = self.norm2(x)
        ff, lp1, lq1 = self.linear1(ff_in, sample=sample)
        ff = F.gelu(ff)
        ff = self.dropout_ff(ff)
        ff, lp2, lq2 = self.linear2(ff, sample=sample)
        x = x + self.dropout2(ff)
        log_prior_total = log_prior_total + lp1 + lp2
        log_variational_total = log_variational_total + lq1 + lq2

        return x, log_prior_total, log_variational_total


class BayesianTransformerEncoder(nn.Module):
    """Stacked Bayesian Transformer encoder."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int = 2,
        prior_sigma: float = 1.0,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BayesianTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    prior_sigma=prior_sigma,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prior_total = torch.zeros((), device=x.device, dtype=x.dtype)
        log_variational_total = torch.zeros((), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, lp, lq = layer(x, sample=sample)
            log_prior_total = log_prior_total + lp
            log_variational_total = log_variational_total + lq
        return x, log_prior_total, log_variational_total


def _to_sensor_tokens(
    u: torch.Tensor,
    input_channels: int,
    num_sensors: int,
    broadcast_params: bool = True,
) -> torch.Tensor:
    """Normalize branch input to (batch, seq_len, input_channels)."""
    if u.dim() == 2:
        if input_channels == 1:
            if u.shape[1] > 1:
                return u.unsqueeze(-1)
            tokens = u.unsqueeze(1)  # (batch, 1, 1)
            return tokens.expand(-1, num_sensors, -1) if broadcast_params else tokens
        if u.shape[1] != input_channels:
            raise ValueError(
                f"Expected u shape (batch, {input_channels}) for parameter input, got {tuple(u.shape)}."
            )
        tokens = u.unsqueeze(1)
        return tokens.expand(-1, num_sensors, -1) if broadcast_params else tokens
    if u.dim() == 3:
        if u.shape[-1] != input_channels:
            raise ValueError(
                f"Expected u.shape[-1] == input_channels ({input_channels}), got {u.shape[-1]}."
            )
        return u
    raise ValueError(f"Unsupported branch input shape: {tuple(u.shape)}")


class BayesianTransformerBranch(nn.Module):
    """Bayesian Transformer branch with stochastic linear projections."""

    def __init__(
        self,
        num_sensors: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        prior_sigma: float = 1.0,
        input_channels: int = 1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_channels = input_channels
        self.embed = BayesianLinear(input_channels, d_model, prior_sigma=prior_sigma)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_sensors, dropout=dropout)
        self.transformer = BayesianTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            prior_sigma=prior_sigma,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.pool_to_output = BayesianLinear(d_model, output_dim, prior_sigma=prior_sigma)

    def forward(self, u: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = _to_sensor_tokens(
            u,
            input_channels=self.input_channels,
            num_sensors=self.num_sensors,
            broadcast_params=True,
        )
        x, lp0, lq0 = self.embed(tokens, sample=sample)
        x = self.pos_enc(x)
        x, lp1, lq1 = self.transformer(x, sample=sample)
        x = x.mean(dim=1)
        x, lp2, lq2 = self.pool_to_output(x, sample=sample)
        return x, lp0 + lp1 + lp2, lq0 + lq1 + lq2


class BayesianTransformerMultiOutputBranch(BayesianTransformerBranch):
    """Bayesian hard-truncation branch for method-2 multi-output DeepONet."""

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
        prior_sigma: float = 1.0,
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
            prior_sigma=prior_sigma,
            input_channels=input_channels,
        )


class BayesianTransformerMultiCLSBranch(nn.Module):
    """Bayesian multi-CLS branch: one stochastic head per output token."""

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
        prior_sigma: float = 1.0,
        input_channels: int = 1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.n_outputs = n_outputs
        self.p_group = p_group
        self.input_channels = input_channels

        self.embed = BayesianLinear(input_channels, d_model, prior_sigma=prior_sigma)
        self.cls_tokens = nn.Parameter(torch.zeros(n_outputs, d_model))
        nn.init.normal_(self.cls_tokens, mean=0.0, std=0.02)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_sensors + n_outputs, dropout=dropout)
        self.transformer = BayesianTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            prior_sigma=prior_sigma,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.group_heads = nn.ModuleList(
            [BayesianLinear(d_model, p_group, prior_sigma=prior_sigma) for _ in range(n_outputs)]
        )

    def forward(self, u: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = _to_sensor_tokens(
            u,
            input_channels=self.input_channels,
            num_sensors=self.num_sensors,
            broadcast_params=True,
        )
        sensor_tokens, lp0, lq0 = self.embed(tokens, sample=sample)
        batch = sensor_tokens.shape[0]
        cls = self.cls_tokens.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat([cls, sensor_tokens], dim=1)
        x = self.pos_enc(x)
        x, lp1, lq1 = self.transformer(x, sample=sample)
        cls_out = x[:, : self.n_outputs, :]

        groups = []
        log_prior = lp0 + lp1
        log_variational = lq0 + lq1
        for i, head in enumerate(self.group_heads):
            g, lp, lq = head(cls_out[:, i, :], sample=sample)
            groups.append(g)
            log_prior = log_prior + lp
            log_variational = log_variational + lq
        out = torch.cat(groups, dim=-1)
        return out, log_prior, log_variational


class BayesianDeepONet(nn.Module):
    """Bayesian DeepONet with pluggable Bayesian branch/trunk modules."""

    def __init__(self, branch: nn.Module, trunk: nn.Module, bias: bool = True, min_noise: float = 1e-3):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None
        self.noise_unconstrained = nn.Parameter(torch.tensor(-3.0))
        self.min_noise = min_noise

    def forward(
        self, u: torch.Tensor, y: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_out, lp_b, lq_b = self.branch(u, sample=sample)
        trunk_out, lp_t, lq_t = self.trunk(y, sample=sample)

        if trunk_out.dim() == 2:
            pred_mean = (branch_out * trunk_out).sum(dim=-1)
        else:
            pred_mean = (branch_out.unsqueeze(1) * trunk_out).sum(dim=-1)
            if pred_mean.size(1) == 1:
                pred_mean = pred_mean.squeeze(1)

        if self.bias is not None:
            pred_mean = pred_mean + self.bias
        pred_std = torch.ones_like(pred_mean) * (F.softplus(self.noise_unconstrained) + self.min_noise)

        log_prior = lp_b + lp_t
        log_variational = lq_b + lq_t
        return pred_mean, pred_std, log_prior, log_variational


class BayesianMultiOutputDeepONet(nn.Module):
    """Bayesian method-2 multi-output DeepONet with grouped dot products."""

    def __init__(
        self,
        branch: nn.Module,
        trunk: nn.Module,
        n_outputs: int,
        p_group: int,
        bias: bool = True,
        min_noise: float = 1e-3,
    ):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.n_outputs = n_outputs
        self.p_group = p_group
        self.bias = nn.Parameter(torch.zeros(n_outputs)) if bias else None
        self.noise_unconstrained = nn.Parameter(torch.tensor(-3.0))
        self.min_noise = min_noise

    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_out, lp_b, lq_b = self.branch(u, sample=sample)
        trunk_out, lp_t, lq_t = self.trunk(y, sample=sample)

        batch = branch_out.shape[0]
        b_group = branch_out.reshape(batch, self.n_outputs, self.p_group)
        if trunk_out.dim() == 2:
            t_group = trunk_out.reshape(batch, self.n_outputs, self.p_group)
            pred_mean = (b_group * t_group).sum(dim=-1)
        else:
            n_points = trunk_out.shape[1]
            t_group = trunk_out.reshape(batch, n_points, self.n_outputs, self.p_group)
            pred_mean = (b_group.unsqueeze(1) * t_group).sum(dim=-1)
            if pred_mean.size(1) == 1:
                pred_mean = pred_mean.squeeze(1)

        if self.bias is not None:
            pred_mean = pred_mean + self.bias
        pred_std = torch.ones_like(pred_mean) * (F.softplus(self.noise_unconstrained) + self.min_noise)
        return pred_mean, pred_std, lp_b + lp_t, lq_b + lq_t


class BayesianPODDeepONet(nn.Module):
    """Bayesian POD-DeepONet with Bayesian branch and fixed POD trunk."""

    def __init__(self, branch: nn.Module, trunk: nn.Module, bias: bool = True, min_noise: float = 1e-3):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None
        self.noise_unconstrained = nn.Parameter(torch.tensor(-3.0))
        self.min_noise = min_noise

    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coeffs, lp_b, lq_b = self.branch(u, sample=sample)
        phi = self.trunk(y)
        if hasattr(self.trunk, "get_mean_at_y"):
            mean_field = self.trunk.get_mean_at_y(y)
        else:
            mean_field = torch.zeros(coeffs.shape[0], device=coeffs.device, dtype=coeffs.dtype)
            if phi.dim() == 3:
                mean_field = mean_field.unsqueeze(1).expand(-1, phi.shape[1])
        if phi.dim() == 2:
            pred_mean = mean_field + (coeffs * phi).sum(dim=-1)
        else:
            pred_mean = mean_field + (coeffs.unsqueeze(1) * phi).sum(dim=-1)
            if pred_mean.size(1) == 1:
                pred_mean = pred_mean.squeeze(1)
        if self.bias is not None:
            pred_mean = pred_mean + self.bias
        pred_std = torch.ones_like(pred_mean) * (F.softplus(self.noise_unconstrained) + self.min_noise)
        return pred_mean, pred_std, lp_b, lq_b


class BayesianExDeepONet(nn.Module):
    """Paper-style Bayesian Ex-DeepONet with branch modulation on all trunk layers."""

    def __init__(self, branch: nn.Module, trunk: nn.Module, bias: bool = True, min_noise: float = 1e-3):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None
        self.noise_unconstrained = nn.Parameter(torch.tensor(-3.0))
        self.min_noise = min_noise

    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coeffs, lp_b, lq_b = self.branch(u, sample=sample)
        pred_mean, lp_t, lq_t = self.trunk(y, coeffs, sample=sample)
        if self.bias is not None:
            pred_mean = pred_mean + self.bias
        pred_std = torch.ones_like(pred_mean) * (F.softplus(self.noise_unconstrained) + self.min_noise)
        return pred_mean, pred_std, lp_b + lp_t, lq_b + lq_t


def init_bayesian_fnn_from_deterministic(
    bayes_model: BayesianDeepONet,
    det_model: nn.Module,
    rho_init: float = -5.0,
) -> None:
    """用确定性 DeepONet (FNN branch+trunk) 的权重初始化 BayesianDeepONet 的 mu，rho 设为 rho_init。"""
    det_branch = det_model.branch
    det_trunk = det_model.trunk
    bayes_branch = bayes_model.branch
    bayes_trunk = bayes_model.trunk

    def _copy_sequential_to_bayesian_mlp(seq_net: nn.Sequential, bayes_mlp: BayesianMLP) -> None:
        linear_layers = [m for m in seq_net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == len(bayes_mlp.layers), (
            f"Layer count mismatch: {len(linear_layers)} vs {len(bayes_mlp.layers)}"
        )
        for det_lin, bayes_lin in zip(linear_layers, bayes_mlp.layers):
            assert isinstance(bayes_lin, BayesianLinear)
            bayes_lin.weight_mu.data.copy_(det_lin.weight.data)
            bayes_lin.weight_rho.data.fill_(rho_init)
            if det_lin.bias is not None:
                bayes_lin.bias_mu.data.copy_(det_lin.bias.data)
                bayes_lin.bias_rho.data.fill_(rho_init)

    _copy_sequential_to_bayesian_mlp(det_branch.net, bayes_branch.net)
    _copy_sequential_to_bayesian_mlp(det_trunk.net, bayes_trunk.net)

    if det_model.bias is not None and bayes_model.bias is not None:
        bayes_model.bias.data.copy_(det_model.bias.data)


def init_bayesian_transformer_from_deterministic(
    bayes_model: BayesianDeepONet,
    det_model: nn.Module,
    rho_init: float = -5.0,
) -> None:
    """用确定性 DeepONet (Transformer branch + FNN trunk) 初始化 BayesianDeepONet (BayesianTransformer + BayesianFNN)。"""
    det_branch = det_model.branch
    det_trunk = det_model.trunk
    bayes_branch = bayes_model.branch
    bayes_trunk = bayes_model.trunk

    # Trunk: FNN -> BayesianFNN (同 init_bayesian_fnn_from_deterministic)
    def _copy_sequential_to_bayesian_mlp(seq_net: nn.Sequential, bayes_mlp: BayesianMLP) -> None:
        linear_layers = [m for m in seq_net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == len(bayes_mlp.layers), (
            f"Trunk layer count mismatch: {len(linear_layers)} vs {len(bayes_mlp.layers)}"
        )
        for det_lin, bayes_lin in zip(linear_layers, bayes_mlp.layers):
            assert isinstance(bayes_lin, BayesianLinear)
            bayes_lin.weight_mu.data.copy_(det_lin.weight.data)
            bayes_lin.weight_rho.data.fill_(rho_init)
            if det_lin.bias is not None:
                bayes_lin.bias_mu.data.copy_(det_lin.bias.data)
                bayes_lin.bias_rho.data.fill_(rho_init)

    _copy_sequential_to_bayesian_mlp(det_trunk.net, bayes_trunk.net)

    # Branch: Transformer embed, pool_to_output (Linear -> BayesianLinear)
    bayes_branch.embed.weight_mu.data.copy_(det_branch.embed.weight.data)
    bayes_branch.embed.weight_rho.data.fill_(rho_init)
    if det_branch.embed.bias is not None and bayes_branch.embed.bias_mu is not None:
        bayes_branch.embed.bias_mu.data.copy_(det_branch.embed.bias.data)
        bayes_branch.embed.bias_rho.data.fill_(rho_init)

    bayes_branch.pool_to_output.weight_mu.data.copy_(det_branch.pool_to_output.weight.data)
    bayes_branch.pool_to_output.weight_rho.data.fill_(rho_init)
    if det_branch.pool_to_output.bias is not None and bayes_branch.pool_to_output.bias_mu is not None:
        bayes_branch.pool_to_output.bias_mu.data.copy_(det_branch.pool_to_output.bias.data)
        bayes_branch.pool_to_output.bias_rho.data.fill_(rho_init)

    # Transformer encoder: PyTorch TransformerEncoderLayer -> BayesianTransformerEncoderLayer
    # det: branch.transformer.encoder (nn.TransformerEncoder) with .layers (ModuleList of TransformerEncoderLayer)
    # bayes: branch.transformer.layers (ModuleList of BayesianTransformerEncoderLayer)
    det_enc = det_branch.transformer.encoder
    d_model = bayes_branch.transformer.layers[0].self_attn.d_model
    n_layers = len(bayes_branch.transformer.layers)
    assert len(det_enc.layers) == n_layers, f"Transformer layer count mismatch: {len(det_enc.layers)} vs {n_layers}"

    for i in range(n_layers):
        det_layer = det_enc.layers[i]
        bayes_layer = bayes_branch.transformer.layers[i]
        # self_attn: in_proj_weight (3*d_model, d_model) -> q_proj, k_proj, v_proj
        in_proj = det_layer.self_attn.in_proj_weight
        q, k, v = torch.split(in_proj, d_model, dim=0)
        bayes_layer.self_attn.q_proj.weight_mu.data.copy_(q)
        bayes_layer.self_attn.q_proj.weight_rho.data.fill_(rho_init)
        bayes_layer.self_attn.k_proj.weight_mu.data.copy_(k)
        bayes_layer.self_attn.k_proj.weight_rho.data.fill_(rho_init)
        bayes_layer.self_attn.v_proj.weight_mu.data.copy_(v)
        bayes_layer.self_attn.v_proj.weight_rho.data.fill_(rho_init)
        if det_layer.self_attn.in_proj_bias is not None:
            qb, kb, vb = torch.split(det_layer.self_attn.in_proj_bias, d_model, dim=0)
            bayes_layer.self_attn.q_proj.bias_mu.data.copy_(qb)
            bayes_layer.self_attn.q_proj.bias_rho.data.fill_(rho_init)
            bayes_layer.self_attn.k_proj.bias_mu.data.copy_(kb)
            bayes_layer.self_attn.k_proj.bias_rho.data.fill_(rho_init)
            bayes_layer.self_attn.v_proj.bias_mu.data.copy_(vb)
            bayes_layer.self_attn.v_proj.bias_rho.data.fill_(rho_init)
        bayes_layer.self_attn.out_proj.weight_mu.data.copy_(det_layer.self_attn.out_proj.weight.data)
        bayes_layer.self_attn.out_proj.weight_rho.data.fill_(rho_init)
        if det_layer.self_attn.out_proj.bias is not None:
            bayes_layer.self_attn.out_proj.bias_mu.data.copy_(det_layer.self_attn.out_proj.bias.data)
            bayes_layer.self_attn.out_proj.bias_rho.data.fill_(rho_init)
        # linear1, linear2 (FFN)
        bayes_layer.linear1.weight_mu.data.copy_(det_layer.linear1.weight.data)
        bayes_layer.linear1.weight_rho.data.fill_(rho_init)
        if det_layer.linear1.bias is not None:
            bayes_layer.linear1.bias_mu.data.copy_(det_layer.linear1.bias.data)
            bayes_layer.linear1.bias_rho.data.fill_(rho_init)
        bayes_layer.linear2.weight_mu.data.copy_(det_layer.linear2.weight.data)
        bayes_layer.linear2.weight_rho.data.fill_(rho_init)
        if det_layer.linear2.bias is not None:
            bayes_layer.linear2.bias_mu.data.copy_(det_layer.linear2.bias.data)
            bayes_layer.linear2.bias_rho.data.fill_(rho_init)

    if det_model.bias is not None and bayes_model.bias is not None:
        bayes_model.bias.data.copy_(det_model.bias.data)


def set_bayesian_prior_from_weights(model: nn.Module, prior_sigma: float | None = None) -> None:
    """将先验设为 N(μ_current, σ)，使 KL 将后验拉向当前权重（如预训练解）而非 N(0,1)。
    prior_sigma: 若提供，则覆盖各层的 prior_sigma，使先验更集中（如 0.1）以更好保留预训练解。"""
    for m in model.modules():
        if isinstance(m, BayesianLinear):
            m.prior_weight_mean = m.weight_mu.detach().clone()
            if m.bias_mu is not None:
                m.prior_bias_mean = m.bias_mu.detach().clone()
            if prior_sigma is not None:
                m.prior_sigma = prior_sigma

