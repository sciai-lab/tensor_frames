import warnings
from typing import Tuple, Union

import torch
from torchvision.ops import MLP

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.reps import Reps

transform_dict = {}

BLOCK_REPS_COUNT = 0


class BlockReps(Reps):
    """The BlockReps class.

    Represents a representation using a Multi-Layer Perceptron (MLP), which computes a vector. The
    vector is then multiplied element-wise with the input tensor.
    """

    def __init__(
        self, mult_dim: str, reps_id: Union[str, int] = "count", spatial_dim: int = 3
    ) -> None:
        """Initialize the BlockReps object.

        Args:
            mult_dim (str): A string representing the multiplicity and dimension of the block, e.g. "4x16", meaning that one applies a 16 by 16 matrix 4 times.
            reps_id (Union[str, int], optional): The ID of the representations. (to use the same representation in different places). Defaults to "count".
            spatial_dim (int, optional): The spatial dimension. Defaults to 3.
            overwrite_dim (int | None, optional): The dimension to overwrite the computed dimension with. Only useful for hacky additions with other reps. Defaults to None.
        """
        super().__init__()
        self.multiplicity, self.block_dim = self.parse_mult_dim(mult_dim)
        self._dim = self.multiplicity * self.block_dim
        self._reps_id = str(reps_id)
        assert not self._reps_id.startswith("_"), "The reps_id cannot start with an underscore."
        if self._reps_id == "count":
            global BLOCK_REPS_COUNT
            self._reps_id = "_" + str(BLOCK_REPS_COUNT)
            BLOCK_REPS_COUNT += 1

        self.spatial_dim = spatial_dim

    def parse_mult_dim(self, mult_dim: str) -> Tuple[int, int]:
        """Parses the input multiplicity and dimension. Special case is.

        Args:
            mult_dim (str): The input multiplicity and dimension. Special case: "4x16_32" means that one applies a 16 by 16 matrix 4 times, but overwrites the dimension with 32.

        Returns:
            Tuple[int, int]: The parsed multiplicity and block dimension.
        """
        # remove any whitespace:
        mult_dim = mult_dim.replace(" ", "")
        return tuple(map(int, mult_dim.split("x")))

    def __repr__(self) -> str:
        """Returns a string representation of the BlockReps object.

        Returns:
            str: A string representation of the BlockReps object.
        """
        return (
            f"BlockReps(dim={self.dim}, reps_id={self._reps_id}, spatial_dim={self.spatial_dim})"
        )

    @property
    def dim(self) -> int:
        """Returns the dimension of the object.

        Returns:
            int: The dimension of the object.
        """
        return self._dim

    def __add__(self, blockreps: Reps) -> "BlockReps":
        """Adds two `BlockReps` objects together. Does not work with ids.

        Args:
            blockreps (Reps): The `BlockReps` object to add.

        Returns:
            BlockReps: The sum of the two `BlockReps` objects.
        """
        if not isinstance(blockreps, BlockReps):
            # assume that the blockreps are already correct:
            return self

        assert self.spatial_dim == blockreps.spatial_dim, "Spatial dimensions must be the same."
        assert self.block_dim == blockreps.block_dim, "Block dimensions must be the same."
        return BlockReps(
            mult_dim=f"{self.multiplicity + blockreps.multiplicity}x{self.block_dim}",
            spatial_dim=self.spatial_dim,
        )

    def get_transform_class(self) -> "BlockRepsTransform":
        """Returns an instance of BlockRepsTransform associated with the current BlockReps object.

        Returns:
            BlockRepsTransform: An instance of BlockRepsTransform.
        """

        if transform_dict.get(self._reps_id) is None:
            transform_dict[self._reps_id] = BlockRepsTransform(
                block_dim=self.block_dim,
                multiplicity=self.multiplicity,
                spatial_dim=self.spatial_dim,
            )

        return transform_dict[self._reps_id]


class BlockRepsTransform(torch.nn.Module):
    """The BlockRepsTransform class.

    Applies a MLP to the input tensor.
    """

    def __init__(self, block_dim: int, multiplicity: int, spatial_dim: int) -> None:
        """Initialize the BlockRepsTransform object.

        Args:
            dim (int): The dimension of the representations.
            spatial_dim (int): The spatial dimension.
        """
        super().__init__()
        self.block_dim = block_dim
        self.multiplicity = multiplicity
        self.mlp = MLP(
            in_channels=spatial_dim**2,
            hidden_channels=[
                block_dim**2,
            ]
            * 2,
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
        if self.block_dim == 0 or self.multiplicity == 0:
            return x

        # check if x can be divided by multiplicity and block_dim:
        if (
            x.shape[-1] % self.block_dim != 0
            or (x.shape[-1] // self.block_dim) % self.multiplicity != 0
        ):
            warnings.warn(
                f"Input tensor shape {x.shape} cannot be reshaped into (N, {self.multiplicity}, {self.block_dim}). No transformation is applied."
            )
            return x

        block_matrix = self.mlp(lframes.matrices.flatten(start_dim=-2, end_dim=-1))
        block_matrix = block_matrix.view(
            -1, self.block_dim, self.block_dim
        )  # (N, block_dim, block_dim)
        x = x.view(-1, self.multiplicity, self.block_dim)  # (N, multiplicity, block_dim)
        x = torch.einsum("nmk,nlk->nml", x, block_matrix)  # (N, multiplicity, block_dim)
        x = x.flatten(start_dim=1)  # (N, multiplicity * block_dim)
        return x
