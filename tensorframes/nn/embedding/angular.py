from typing import Tuple, Union

import e3nn.o3 as o3
import numpy as np
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


class SphericalHarmonicsEmbedding(AngularEmbedding):
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


def fibonacci_sphere(samples=1000):
    """Generate points on a sphere using the Fibonacci sphere algorithm.

    Args:
        samples (int): The number of points to generate on the sphere. Default is 1000.

    Returns:
        list: A list of tuples representing the points on the sphere. Each tuple contains the x, y, and z coordinates.
    """
    points = []
    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points


class GaussianOnSphereEmbedding(AngularEmbedding):
    """A class representing a Gaussian embedding on a sphere.

    Args:
        num_angular (int): The number of equi-spaced points on the sphere.
        normalized (bool, optional): Whether to normalize the gaussians. Defaults to True.
        is_learnable (bool, optional): Whether the gaussian widths are learnable parameters. Defaults to True.
    """

    def __init__(self, num_angular: int, normalized: bool = True, is_learnable: bool = True):
        super().__init__()
        self.normalized = normalized

        # get num_angular equi spaced points on the sphere:
        points = torch.tensor(fibonacci_sphere(num_angular))

        # calculate mean of the difference of the points
        mean = torch.mean(torch.norm(points[1:] - points[:-1]), dim=0)
        widths = mean.repeat(num_angular).view(1, num_angular)

        # initialize the gaussian widths to be the mean of the difference of the points
        if is_learnable:
            self.centers = torch.nn.Parameter(data=points)
            self.widths = torch.nn.Parameter(data=widths)
        else:
            self.register_buffer("centers", points)
            self.register_buffer("widths", widths)

        self.out_dim = num_angular

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Compute the Gaussian embedding for the given edge vector.

        Args:
            edge_vec (Tensor): The input edge vector.

        Returns:
            Tensor: The computed Gaussian embedding.
        """
        edge_vec = edge_vec / torch.clamp(
            torch.linalg.norm(edge_vec, dim=-1, keepdim=True), min=1e-9
        )
        diff_norm = torch.linalg.norm(edge_vec.unsqueeze(1) - self.centers, dim=-1)

        # calculate the gaussian embedding
        exp = torch.exp(-0.5 * (diff_norm / self.widths) ** 2).to(edge_vec.dtype)

        # normalize the gaussians
        if self.normalized:
            exp = exp / torch.sqrt(self.widths * torch.pi).to(edge_vec.dtype)

        return exp
