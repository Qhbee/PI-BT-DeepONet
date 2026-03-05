"""Data generators for different PDEs/ODEs."""

from .antiderivative import generate_antiderivative_data
from .burgers import generate_burgers_data
from .darcy import generate_darcy_data
from .diffusion_reaction import generate_diffusion_reaction_data

__all__ = [
    "generate_antiderivative_data",
    "generate_burgers_data",
    "generate_diffusion_reaction_data",
    "generate_darcy_data",
]
