from typing import Tuple, Union

import e3nn.o3 as o3
import torch
from torch import Tensor

from tensorframes.nn.embedding.radial import compute_edge_vec


class AngularEmbedding(torch.nn.Module):
    """Angular Embedding module."""

    def __init__(self):
        """Init AngularEmbedding module."""
        super().__init__()
        self.out_dim = None  # should be set in the subclass

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Computes the embedding for the given edge vector.

        Args:
            edge_vec (Tensor): The input edge vector.

        Returns:
            Tensor: The computed embedding.
        """
        raise NotImplementedError

    def forward(
        self, pos: Union[Tensor, Tuple], edge_index: Tensor, edge_vec: Tensor = None
    ) -> Tensor:
        """Forward pass of the AngularEmbedding module.

        Args:
            pos (Union[Tensor, Tuple]): The position tensor or tuple.
            edge_index (Tensor): The edge index tensor.
            edge_vec (Tensor, optional): The edge vector tensor. Defaults to None.

        Returns:
            Tensor: The computed embedding.
        """
        if edge_vec is None:
            edge_vec = compute_edge_vec(pos, edge_index)

        return self.compute_embedding(edge_vec)


class TrivialAngularEmbedding(AngularEmbedding):
    """A trivial implementation of the AngularEmbedding class."""

    def __init__(self, normalize: bool = True):
        """Init TrivialAngularEmbedding module.

        Args:
            normalize (bool, optional): Indicates whether to normalize the computed embeddings. Defaults to True.

        Attributes:
            normalize (bool): Indicates whether to normalize the computed embeddings.
            out_dim (int): The output dimension of the embeddings.
        """
        super().__init__()
        self.normalize = normalize
        self.out_dim = 3

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Computes the embedding for the given edge vector.

        Args:
            edge_vec (Tensor): The input edge vector.

        Returns:
            Tensor: The computed embedding.
        """
        if self.normalize:
            return edge_vec / torch.clamp(
                torch.linalg.norm(edge_vec, dim=-1, keepdim=True), min=1e-9
            )
        else:
            return edge_vec


class SphericalHarmonicsEmbedding(torch.nn.Module):
    """Spherical Harmonics Embedding module."""

    def __init__(self, lmax: int = 2, normalization: str = "norm"):
        """Init Spherical Harmonics Embedding module.

        Args:
            lmax (int): Maximum degree of the spherical harmonics expansion. Default is 2.
            normalization (str): Normalization method for the spherical harmonics. Choose from ["component", "norm", "integral"].
                Default is "norm".
        """
        super().__init__()
        self.normalization = normalization
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.out_dim = self.irreps_sh.dim - 1  # remove the constant function

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Compute the spherical harmonics embedding for the given edge vectors.

        Args:
            edge_vec (Tensor): Tensor of shape (batch_size, num_edges, 3) representing the edge vectors.

        Returns:
            Tensor: Tensor of shape (batch_size, num_edges, out_dim) representing the computed embeddings.
        """
        return o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization=self.normalization
        )[..., 1:]
