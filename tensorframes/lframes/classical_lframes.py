import torch
from torch import Tensor
from torch_geometric.nn import knn

from tensorframes.lframes.gram_schmidt import gram_schmidt
from tensorframes.lframes.lframes import LFrames


class ThreeNNLFrames(torch.nn.Module):
    """Computes local frames using the 3-nearest neighbors.

    The Frames are O(3) equivariant.
    """

    def __init__(self) -> None:
        """Initializes an instance of the ThreeNNLFrames class."""
        super().__init__()

    def forward(self, pos: Tensor, idx: Tensor | None = None, batch: Tensor | None = None):
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
