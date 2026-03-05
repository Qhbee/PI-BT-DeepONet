"""Data: datasets and generators."""

from .registry import get_generator, register

__all__ = ["register", "get_generator"]
