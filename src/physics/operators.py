"""Differential operators for PDE residuals (autograd-based)."""

import torch


def gradient_scalar_wrt_coords(
    pred: torch.Tensor,
    y: torch.Tensor,
    grad_outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute gradient of scalar pred w.r.t. coordinates y via autograd.

    Args:
        pred: (batch, n_points) or (batch,) scalar predictions
        y: (batch, n_points, coord_dim) or (batch, coord_dim) coordinates, must have requires_grad=True
        grad_outputs: optional, defaults to ones

    Returns:
        grad: same shape as y, d(pred)/d(y)
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(pred)
    grad = torch.autograd.grad(
        outputs=pred,
        inputs=y,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    return grad


def second_derivative_along_dim(
    pred: torch.Tensor,
    y: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Compute second derivative d2(pred)/d(y_dim)^2 for scalar pred."""
    grad = torch.autograd.grad(outputs=pred.sum(), inputs=y, create_graph=True)[0]
    first = grad[..., dim]
    second = torch.autograd.grad(outputs=first.sum(), inputs=y, create_graph=True)[0][..., dim]
    return second


def laplacian_scalar(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian of scalar pred w.r.t. coordinates y."""
    coord_dim = y.shape[-1]
    lap = 0.0
    for d in range(coord_dim):
        lap = lap + second_derivative_along_dim(pred, y, d)
    return lap
