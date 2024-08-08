from typing import Tuple

import torch

from tensorframes.lframes import LFrames


class WrappedLearnedLocalFramesModule(torch.nn.Module):
    """Dummy class for WrappedLearnedLocalFramesModule."""

    def __init__(self):
        """TODO: to be defined."""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor | None,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> Tuple[torch.Tensor, LFrames]:
        """TODO: to be defined."""
        lframes = None
        return x, lframes
