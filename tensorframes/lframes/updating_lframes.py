from typing import Tuple

import torch

from tensorframes.lframes import LFrames
from tensorframes.nn.mlp import MLPWrapped
from tensorframes.reps.utils import parse_reps
from tensorframes.utils.quaternions import quaternions_to_matrix


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


class QuaternionsUpdateLFrames(torch.nn.Module):
    def __init__(
        self,
        in_reps,
        hidden_channels,
        learning_rate_factor=1.0,
        init_zero_angle=True,
        eps=1e-6,
        **mlp_kwargs,
    ):
        super().__init__()
        self.in_reps = parse_reps(in_reps)
        self.learning_rate_factor = learning_rate_factor
        self.eps = eps

        self.mlp = MLPWrapped(
            in_channels=self.in_reps.dim,
            hidden_channels=hidden_channels + [5],
            **mlp_kwargs,
        )
        self.coeffs_transform = self.in_reps.get_transform_class()

    def forward(
        self, x: torch.Tensor, lframes: LFrames, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, LFrames]:
        out = self.mlp(x, batch=batch)
        denominator = torch.where(out[..., 0].abs() < self.eps, self.eps, out[..., 0])
        angle = torch.arctan2(out[..., 1], denominator)
        axis = torch.nn.functional.normalize(out[..., 2:], p=2, dim=-1)
        rot_matr = quaternions_to_matrix(
            torch.cat(
                [torch.cos(angle / 2).unsqueeze(-1), torch.sin(angle / 2).unsqueeze(-1) * axis],
                dim=-1,
            )
        )

        new_lframes = LFrames(torch.einsum("ijk, ikn -> ijn", rot_matr, lframes.matrices))
        new_x = self.coeffs_transform(x, LFrames(rot_matr))

        return new_x, new_lframes
