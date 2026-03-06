"""Physics-informed constraints: PDE residuals."""

from .pde_residual import compute_residual
from .ns_residual import beltrami_solution, kovasznay_solution, pressure_gauge_loss

__all__ = ["compute_residual", "kovasznay_solution", "beltrami_solution", "pressure_gauge_loss"]
