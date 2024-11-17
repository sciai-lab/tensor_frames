from typing import Union

import torch
from e3nn.o3 import rand_matrix
from torch import Tensor
from torch_geometric.nn import knn, radius
from torch_geometric.utils import scatter

from tensorframes.lframes.gram_schmidt import gram_schmidt
from tensorframes.lframes.lframes import LFrames


class LFramesPredictionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> LFrames:
        assert NotImplementedError, "Subclasses must implement this method."


class PCALFrames(LFramesPredictionModule):
    """Computes local frames using PCA."""

    def __init__(
        self, r: float, max_num_neighbors: int = 64, exceptional_choice: str = "random"
    ) -> None:
        """Initializes an instance of the PCALFrames class.

        Args:
            radius (float): The radius for the PCA computation.
            max_neighbors (int, optional): The maximum number of neighbors to consider. Defaults to 10.
            exceptional_choice (str, optional): The choice for exceptional case (with zero neighbors). Defaults to "random".
        """
        super().__init__()
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.exceptional_choice = exceptional_choice

    def forward(
        self, pos: Tensor, idx: Union[Tensor, None] = None, batch: Union[Tensor, None] = None
    ) -> LFrames:
        """Forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor of shape (N, D) representing the positions of N points in D-dimensional space.
            idx (Tensor, optional): The indices of the points to consider. If None, all points are considered. Defaults to None.
            batch (Tensor, optional): The batch indices of the points. If None, a batch of zeros is used. Defaults to None.

        Returns:
            LFrames: The computed local frames as an instance of the LFrames class.
        """
        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)

        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.max_num_neighbors
        )
        # print("average number of neighbors: ", len(row) / len(idx), "max_num_neighbors", self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        cov_matrices = scatter(
            edge_vec.unsqueeze(-1) * edge_vec.unsqueeze(-2),
            edge_index[1],
            dim=0,
        )

        # compute the PCA:
        _, eigenvectors = torch.linalg.eigh(cov_matrices)

        # check how many neighbors each point has:
        num_neighbors = scatter(
            torch.ones_like(edge_index[0]), edge_index[1], dim=0, reduce="sum"
        ).float()
        no_neighbors_mask = num_neighbors <= 1
        if self.exceptional_choice == "random":
            random_lframes = RandomLFrames()(pos[no_neighbors_mask]).matrices
            eigenvectors[no_neighbors_mask] = random_lframes
        elif self.exceptional_choice == "zero":
            eigenvectors[no_neighbors_mask] = 0.0
        else:
            assert (
                NotImplementedError
            ), f"exceptional_choice {self.exceptional_choice} not implemented"

        return LFrames(eigenvectors.transpose(-1, -2))


class ThreeNNLFrames(LFramesPredictionModule):
    """Computes local frames using the 3-nearest neighbors.

    The Frames are O(3) equivariant.
    """

    def __init__(self) -> None:
        """Initializes an instance of the ThreeNNLFrames class."""
        super().__init__()

    def forward(
        self, pos: Tensor, idx: Union[Tensor, None] = None, batch: Union[Tensor, None] = None
    ) -> LFrames:
        """Forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor of shape (N, D) representing the positions of N points in D-dimensional space.
            idx (Tensor, optional): The indices of the points to consider. If None, all points are considered. Defaults to None.
            batch (Tensor, optional): The batch indices of the points. If None, a batch of zeros is used. Defaults to None.

        Returns:
            LFrames: The computed local frames as an instance of the LFrames class.
        """

        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)

        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

        # convert idx to bool tensor:
        if idx.dtype != torch.bool:
            idx = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device).scatter_(
                0, idx, True
            )

        # find 2 closest neighbors:
        row, col = knn(pos, pos[idx], k=4, batch_x=batch, batch_y=batch[idx])
        mask_self_loops = torch.arange(pos.shape[0], dtype=int, device=idx.device)[idx][row] == col
        assert (
            mask_self_loops.sum() == idx.sum()
        ), f"every center should have a self loop, {mask_self_loops.sum()} != {idx.sum()}"
        row = row[~mask_self_loops]
        col = col[~mask_self_loops].reshape((-1, 3))  # remove self loops

        assert torch.all(row[1:] >= row[:-1]), "row must be sorted"

        # compute the local frames:
        row = torch.arange(col.shape[0], device=pos.device)
        x_axis = pos[col[:, 0]] - pos[row]
        y_axis = pos[col[:, 1]] - pos[row]
        z_axis = pos[col[:, 2]] - pos[row]

        matrices = gram_schmidt(x_axis, y_axis, z_axis)

        return LFrames(matrices)


class RandomLFrames(LFramesPredictionModule):
    """Randomly generates local frames for each node."""

    def __init__(self, flip_probability: float = 0.5) -> None:
        """Initialize an instance of the RandomLFrames class.

        Args:
            flip_probability (float, optional): The probability of flipping the frames. Defaults to 0.5.
        """
        super().__init__()
        self.flip_probability = flip_probability

    def forward(
        self, pos: Tensor, idx: Union[Tensor, None] = None, batch: Union[Tensor, None] = None
    ) -> LFrames:
        """Forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor representing the positions.
            idx (Tensor, optional): The indices to select from the input tensor. Defaults to None.
            batch (Tensor, optional): The batch tensor. Defaults to None.

        Returns:
            LFrames: The output LFrames.
        """
        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        lframes = rand_matrix(pos[idx].shape[0], device=pos.device)
        if self.flip_probability > 0:
            flip_mask = torch.rand(lframes.shape[0], device=lframes.device) < self.flip_probability
            # flip the x-axis
            lframes[flip_mask, 0] = -lframes[flip_mask, 0]
        return LFrames(lframes)


class RandomGlobalLFrames(LFramesPredictionModule):
    """Randomly generates a global frame."""

    def __init__(self, flip_probability: float = 0.5) -> None:
        """Initializes an instance of the RandomGlobalLFrames class.

        Args:
            flip_probability (float, optional): The probability of flipping the frames. Defaults to 0.5.
        """
        super().__init__()
        self.flip_probability = flip_probability

    def forward(
        self, pos: Tensor, idx: Union[Tensor, None] = None, batch: Union[Tensor, None] = None
    ) -> LFrames:
        """Applies forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor representing the positions.
            idx (Tensor, optional): The indices tensor. Defaults to None.
            batch (Tensor, optional): The batch tensor. Defaults to None.

        Returns:
            LFrames: The output LFrames tensor.
        """
        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

        # randomly generate one local frame
        matrix = rand_matrix(1, device=pos.device)

        # if random number is less than 0.5, flip the x-axis
        if torch.rand(1, device=pos.device) < self.flip_probability:
            matrix[0] = -matrix[0]

        return LFrames(matrix.repeat(pos[idx].shape[0], 1, 1))


class IdentityLFrames(LFramesPredictionModule):
    """Identity local frames."""

    def __init__(self) -> None:
        """Initializes an instance of the ClassicalLFrames class."""
        super().__init__()

    def forward(
        self, pos: Tensor, idx: Union[Tensor, None] = None, batch: Union[Tensor, None] = None
    ) -> LFrames:
        """Forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor of shape (N, 3) representing the positions.
            idx (Tensor): The index tensor of shape (N,) representing the indices to select from `pos`.
                If None, all indices are selected.
            batch (Tensor): The batch tensor of shape (N,) representing the batch indices.

        Returns:
            LFrames: The output LFrames object.
        """
        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

        return LFrames(torch.eye(3, device=pos.device).repeat(pos[idx].shape[0], 1, 1))
