"""Hard boundary constraint: output transform for Dirichlet BC / IC.

Solution decomposition: s_pred = g(y) + ell(y) * N(u)(y)
where ell(y)=0 on boundary, ell(y)>0 in interior.
For zero BC/IC: g=0, so s_pred = ell(y) * N(u)(y).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _distance_function_diffusion_reaction(y: torch.Tensor) -> torch.Tensor:
    """
    ell(t,x) = t * x * (1-x) for domain t in [0,1], x in [0,1].
    - IC at t=0: ell=0
    - BC at x=0, x=1: ell=0
    - Interior: ell>0
    y: (..., 2) with y[...,0]=t, y[...,1]=x
    """
    t = y[..., 0]
    x = y[..., 1]
    return t * x * (1.0 - x)


def distance_function(y: torch.Tensor, case: str) -> torch.Tensor:
    """Return ell(y) for given case. Shape matches y[..., :1]."""
    if case == "diffusion_reaction":
        ell = _distance_function_diffusion_reaction(y)
    else:
        raise ValueError(f"Hard BC distance function not implemented for case={case}")
    if ell.dim() < y.dim():
        ell = ell.unsqueeze(-1)
    return ell


def _apply_ell(ell: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Broadcast and multiply ell * out. ell and out must have compatible leading dims."""
    if ell.dim() > out.dim():
        ell = ell.squeeze(-1)
    while ell.dim() < out.dim():
        ell = ell.unsqueeze(-1)
    if ell.shape[-1] == 1 and out.shape[-1] > 1:
        ell = ell.expand_as(out)
    return ell * out


class HardBCWrapper(nn.Module):
    """
    Wraps a DeepONet to enforce hard Dirichlet BC/IC via output transform.
    Output: s_pred = ell(y) * raw_output, so s_pred=0 on boundary by construction.
    """

    def __init__(self, model: nn.Module, case: str):
        super().__init__()
        self.model = model
        self.case = case

    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor,
        sample: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import inspect
        sig = inspect.signature(self.model.forward)
        if "sample" in sig.parameters:
            raw = self.model(u, y, sample=sample)
        else:
            raw = self.model(u, y)

        ell = distance_function(y, self.case)
        if ell.dim() == 2 and ell.shape[-1] == 1:
            ell = ell.squeeze(-1)

        if isinstance(raw, tuple):
            pred_mean, pred_std, log_prior, log_variational = raw
            pred_mean = _apply_ell(ell, pred_mean)
            pred_std = _apply_ell(ell, pred_std).clamp(min=1e-6)  # avoid scale=0 for Normal
            return pred_mean, pred_std, log_prior, log_variational
        return _apply_ell(ell, raw)
