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
        log_prior = _log_gaussian(weight, torch.zeros_like(weight), prior_sigma).sum()
        weight_sigma = self._sigma_from_rho(self.weight_rho)
        log_variational = _log_gaussian(weight, self.weight_mu, weight_sigma).sum()

        if bias is not None and self.bias_mu is not None and self.bias_rho is not None:
            bias_prior_sigma = torch.full_like(bias, self.prior_sigma)
            log_prior = log_prior + _log_gaussian(bias, torch.zeros_like(bias), bias_prior_sigma).sum()
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
    ):
        super().__init__()
        self.embed = BayesianLinear(1, d_model, prior_sigma=prior_sigma)
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
        x, lp0, lq0 = self.embed(u.unsqueeze(-1), sample=sample)
        x = self.pos_enc(x)
        x, lp1, lq1 = self.transformer(x, sample=sample)
        x = x.mean(dim=1)
        x, lp2, lq2 = self.pool_to_output(x, sample=sample)
        return x, lp0 + lp1 + lp2, lq0 + lq1 + lq2


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
