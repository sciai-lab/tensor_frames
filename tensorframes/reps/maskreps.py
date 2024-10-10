from typing import Union

import torch
from torchvision.ops import MLP

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.reps import Reps

transform_dict = {}

MASK_REPS_COUNT = 0


class MaskReps(Reps):
    """The MaskReps class.

    Represents a representation using a Multi-Layer Perceptron (MLP), which computes a vector. The
    vector is then multiplied element-wise with the input tensor.
    """

    def __init__(self, dim: int, reps_id: Union[str, int] = "count", spatial_dim: int = 3) -> None:
        """Initialize the MaskReps object.

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
            global MASK_REPS_COUNT
            self._reps_id = "_" + str(MASK_REPS_COUNT)
            MASK_REPS_COUNT += 1

        self.spatial_dim = spatial_dim

    def __repr__(self) -> str:
        """Returns a string representation of the MaskReps object.

        Returns:
            str: A string representation of the MaskReps object.
        """
        return f"MaskReps(dim={self.dim}, reps_id={self._reps_id}, spatial_dim={self.spatial_dim})"

    @property
    def dim(self) -> int:
        """Returns the dimension of the object.

        Returns:
            int: The dimension of the object.
        """
        return self._dim

    def __add__(self, maskreps) -> "MaskReps":
        """Adds two `MaskReps` objects together. Does not work with ids.

        Args:
            MaskReps (MaskReps): The `MaskReps` object to add.

        Returns:
            MaskReps: The sum of the two `MaskReps` objects.
        """
        return MaskReps(self.dim + maskreps.dim, spatial_dim=self.spatial_dim)

    def get_transform_class(self) -> "MaskRepsTransform":
        """Returns an instance of MaskRepsTransform associated with the current MaskReps object.

        Returns:
            MaskRepsTransform: An instance of MaskRepsTransform.
        """

        if transform_dict.get(self._reps_id) is None:
            transform_dict[self._reps_id] = MaskRepsTransform(self.dim, self.spatial_dim)

        return transform_dict[self._reps_id]


class MaskRepsTransform(torch.nn.Module):
    """The MaskRepsTransform class.

    Applies a MLP to the input tensor.
    """

    def __init__(self, dim: int, spatial_dim: int) -> None:
        """Initialize the MaskRepsTransform object.

        Args:
            dim (int): The dimension of the representations.
            spatial_dim (int): The spatial dimension.
        """
        super().__init__()
        self.dim = dim
        self.mlp = MLP(
            in_channels=spatial_dim**2,
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

        return self.mlp(lframes.matrices.flatten(start_dim=-2, end_dim=-1)) * x
