from typing import Tuple

import torch

from tensorframes.lframes import LFrames


class UpdateLFramesModule(torch.nn.Module):
    """Dummy class for UpdateLFramesModule."""

    def __init__(self):
        """To be defined."""
        super().__init__()

    def forward(
        self, x: torch.Tensor | None, lframes: LFrames, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """To be defined."""
        return x, lframes
