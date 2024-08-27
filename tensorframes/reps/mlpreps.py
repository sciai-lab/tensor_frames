from typing import Union

import torch
from torchvision.ops import MLP

from tensorframes.lframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.reps.reps import Reps

transform_dict = {}


class MLPReps(Reps):
    """The MLPReps class.

    Represents a tensor representation using a Multi-Layer Perceptron (MLP).
    """

    def __init__(self, dim: int, reps_id: Union[str, int]) -> None:
        """Initialize the MLPReps object.

        Args:
            dim (int): The dimension of the representations.
            reps_id (Union[str, int]): The ID of the representations. (to use the same representation in different places)
        """
        super().__init__()
        self._dim = dim
        self._reps_id = str(reps_id)

    def __repr__(self) -> str:
        """Returns a string representation of the MLPReps object.

        Returns:
            str: A string representation of the MLPReps object.
        """
        return f"MLPReps(dim={self.dim}), reps_id={self._reps_id}"

    @property
    def dim(self) -> int:
        """Returns the dimension of the object.

        Returns:
            int: The dimension of the object.
        """
        return self._dim

    def get_transform_class(self) -> "MLPRepsTransform":
        """Returns an instance of MLPRepsTransform associated with the current MLPReps object.

        Returns:
            MLPRepsTransform: An instance of MLPRepsTransform.
        """

        if transform_dict.get(self._reps_id) is None:
            transform_dict[self._reps_id] = MLPRepsTransform(self.dim)

        return transform_dict[self._reps_id]


class MLPRepsTransform(torch.nn.Module):
    """The MLPRepsTransform class.

    Applies a MLP to the input tensor.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the MLPRepsTransform object.

        Args:
            dim (int): The dimension of the representations.
        """
        super().__init__()
        self.dim = dim
        self.mlp = MLP(
            in_channels=dim + 9, hidden_channels=[dim] + [dim], activation_layer=torch.nn.SiLU
        )

    def forward(self, x: torch.Tensor, lframes: Union[LFrames, ChangeOfLFrames]) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.
            lframes (Union[LFrames, ChangeOfLFrames]): LFrames or ChangeOfLFrames object.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        mlp_input = torch.cat([x, lframes.matrices.flatten(start_dim=-2, end_dim=-1)], dim=1)

        return self.mlp(mlp_input)
