"""Data generators for different PDEs/ODEs."""

from .antiderivative import generate_antiderivative_data
from .burgers import generate_burgers_data
from .darcy import generate_darcy_data
from .diffusion_reaction import generate_diffusion_reaction_data
from .poisson_2d import generate_poisson_2d_data
from .ns_beltrami_ic2field import generate_ns_beltrami_ic2field_data
from .ns_beltrami_parametric import generate_ns_beltrami_parametric_data
from .ns_kovasznay_bc2field import generate_ns_kovasznay_bc2field_data
from .ns_kovasznay_parametric import generate_ns_kovasznay_parametric_data

__all__ = [
    "generate_antiderivative_data",
    "generate_burgers_data",
    "generate_diffusion_reaction_data",
    "generate_darcy_data",
    "generate_poisson_2d_data",
    "generate_ns_kovasznay_parametric_data",
    "generate_ns_beltrami_parametric_data",
    "generate_ns_kovasznay_bc2field_data",
    "generate_ns_beltrami_ic2field_data",
]
