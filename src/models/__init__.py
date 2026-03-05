"""Models: DeepONet, Branch, Trunk."""

from .deeponet import DeepONet
from .branch import FNNBranch
from .trunk import FNNTrunk

__all__ = ["DeepONet", "FNNBranch", "FNNTrunk"]
