from typing import Union

import torch
from torchvision.ops import MLP

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.reps import Reps

transform_dict = {}

MLP_REPS_COUNT = 0


class MLPReps(Reps):
    """The MLPReps class.

    Represents a representation using a Multi-Layer Perceptron (MLP).
    """

    def __init__(self, dim: int, reps_id: Union[str, int] = "count", spatial_dim: int = 3) -> None:
        """Initialize the MLPReps object.

        Args:
            dim (int): The dimension of the representations.
            reps_id (Union[str, int], optional): The ID of the representations. (to use the same representation in different places). Defaults to "count".
            spatial_dim (int, optional): The spatial dimension. Defaults to 3.
        """
        super().__init__()
        self._dim = dim
        self._reps_id = str(reps_id)
        assert not self._reps_id.startswith("_"), "The reps_id cannot start with an underscore."
        if self._reps_id == "count":
            global MLP_REPS_COUNT
            self._reps_id = "_" + str(MLP_REPS_COUNT)
            MLP_REPS_COUNT += 1

        self.spatial_dim = spatial_dim

    def __repr__(self) -> str:
        """Returns a string representation of the MLPReps object.

        Returns:
            str: A string representation of the MLPReps object.
        """
        return f"MLPReps(dim={self.dim}, reps_id={self._reps_id}, spatial_dim={self.spatial_dim})"

    @property
    def dim(self) -> int:
        """Returns the dimension of the object.

        Returns:
            int: The dimension of the object.
        """
        return self._dim

    def __add__(self, mlpreps) -> "MLPReps":
        """Adds two `MLPReps` objects together. Does not work with ids.

        Args:
            MLPReps (MLPReps): The `MLPReps` object to add.

        Returns:
            MLPReps: The sum of the two `MLPReps` objects.
        """
        return MLPReps(self.dim + mlpreps.dim, spatial_dim=self.spatial_dim)

    def get_transform_class(self) -> "MLPRepsTransform":
        """Returns an instance of MLPRepsTransform associated with the current MLPReps object.

        Returns:
            MLPRepsTransform: An instance of MLPRepsTransform.
        """

        if transform_dict.get(self._reps_id) is None:
            transform_dict[self._reps_id] = MLPRepsTransform(self.dim, self.spatial_dim)

        return transform_dict[self._reps_id]


class MLPRepsTransform(torch.nn.Module):
    """The MLPRepsTransform class.

    Applies a MLP to the input tensor.
    """

    def __init__(self, dim: int, spatial_dim: int) -> None:
        """Initialize the MLPRepsTransform object.

        Args:
            dim (int): The dimension of the representations.
            spatial_dim (int): The spatial dimension.
        """
        super().__init__()
        self.dim = dim
        self.mlp = MLP(
            in_channels=dim + spatial_dim**2,
            hidden_channels=[dim] + [dim],
            activation_layer=torch.nn.SiLU,
        )

    def forward(self, x: torch.Tensor, lframes: LFrames) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.
            lframes (LFrames): LFrames or ChangeOfLFrames object.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        mlp_input = torch.cat([x, lframes.matrices.flatten(start_dim=-2, end_dim=-1)], dim=1)

        return self.mlp(mlp_input)
