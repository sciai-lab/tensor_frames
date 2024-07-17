from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from tensorframes.nn.envelope import Envelope


def compute_edge_vec(
    pos: Union[Tensor, Tuple], edge_index: Tensor, lframes: Union[Tensor, Tuple] = None
) -> Tensor:
    """Compute the edge vectors between node positions and rotates them into the local frames of
    the receiving nodes.

    Args:
        pos (Union[Tensor, Tuple]): The positions of the atoms. Can be a single tensor or a tuple of tensors.
        edge_index (Tensor): The edge index representing the connections between atoms.
        lframes (Union[Tensor, Tuple], optional): The local frames of the atoms. Can be a single tensor or a tuple of tensors. Defaults to None.

    Returns:
        Tensor: The computed edge vectors.
    """

    # get the position of the atom i and j from the edge index
    # j is the first row of the edge index and i is the second row
    if not isinstance(pos, tuple):
        pos_i = pos.index_select(0, edge_index[1])
        pos_j = pos.index_select(0, edge_index[0])
    else:
        pos_i = pos[1].index_select(0, edge_index[1])
        pos_j = pos[0].index_select(0, edge_index[0])

    if not isinstance(lframes, tuple):
        lframes = (lframes, lframes)

    edge_vec = pos_j - pos_i
    if lframes[1] is not None:
        lframes_i = lframes[1].index_select(0, edge_index[1]).reshape(-1, 3, 3)
        edge_vec = torch.matmul(lframes_i, edge_vec.unsqueeze(-1)).squeeze(-1)
    return edge_vec


class RadialEmbedding(Module):
    """RadialEmbedding module for computing embeddings based on radial distances.

    Methods:
        compute_embedding(norm: Tensor) -> Tensor:
            Computes the embedding based on the given norm.

        forward(pos: Union[Tensor, Tuple], edge_index: Tensor, edge_vec: Tensor = None) -> Tensor:
            Forward pass of the RadialEmbedding module.
    """

    def __init__(self):
        """Initialises the RadialEmbedding module."""
        super().__init__()
        self.out_dim = None  # should be set in the subclass

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Computes the embedding based on the given norm.

        Args:
            norm (Tensor): The norm of the distance vector. Shape: E x 1.

        Returns:
            Tensor: The computed embedding. of shape E x out_dim.
        """
        raise NotImplementedError

    def forward(
        self, pos: Union[Tensor, Tuple], edge_index: Tensor, edge_vec: Tensor = None
    ) -> Tensor:
        """Forward pass of the RadialEmbedding module.

        Args:
            pos (Union[Tensor, Tuple]): The position tensor or tuple.
            edge_index (Tensor): The edge index tensor.
            edge_vec (Tensor, optional): The edge vector tensor. Defaults to None.

        Returns:
            Tensor: The computed embedding.
        """
        if edge_vec is None:
            edge_vec = compute_edge_vec(pos, edge_index)

        # calculate the norm of the distance vector
        norm = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)  # Shape E x 1

        return self.compute_embedding(norm)


class BesselEmbedding(RadialEmbedding):
    """BesselEmbedding class represents a radial embedding using Bessel functions.

    Args:
        num_radial (int): The number of radial basis functions.
        cutoff (float): The cutoff radius for the radial basis functions.
        envelope_exponent (int): The exponent for the envelope function.

    Attributes:
        num_radial (int): The number of radial basis functions.
        inv_cutoff (float): The inverse of the cutoff radius.
        norm_const (float): The normalization constant.
        out_dim (int): The output dimension of the embedding.
        envelope (Envelope): The envelope function.
        frequencies (torch.nn.Parameter): The frequencies of the radial basis functions.

    Methods:
        compute_embedding(norm: Tensor) -> Tensor:
            Computes the embedding for the given norm.
    """

    def __init__(self, num_radial: int, cutoff: float, envelope_exponent: int):
        super().__init__()

        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5
        self.out_dim = num_radial

        self.envelope = Envelope(envelope_exponent)

        data = torch.pi * torch.arange(1, num_radial + 1)
        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=data,
            requires_grad=True,
        )

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Computes the embedding for the given norm.

        Args:
            norm (Tensor): The norm of the input.

        Returns:
            Tensor: The computed embedding.
        """
        norm_scaled = norm * self.inv_cutoff
        norm_env = self.envelope(norm_scaled)

        return (
            norm_env * self.norm_const * torch.sin(norm_scaled * self.frequencies) / (norm + 1e-9)
        )


class GaussianEmbedding(Module):
    """Module to calculate the edge attributes based on gaussian basis functions and the distance
    between nodes."""

    def __init__(
        self,
        num_gaussians: int = 10,
        normalized: bool = False,
        maximum_initial_radius: float = 5.0,
        is_learnable: bool = True,
        intersection: float = 0.5,
        gaussian_width: float = None,
    ) -> None:
        """Initialises the class. You can specify the number of gaussians and if the gaussians
        should be normalized. This function initialises the shift and scale parameters of the
        gaussians as learnable parameters.

        Args:
            num_gaussians (int, optional): Number of gaussian functions on which the edge features are calculated.
                Defaults to 10.
            normalized (bool, optional): Defines if the gaussians should be normalized. Defaults to False.
        """
        super().__init__()

        self.num_gaussians = num_gaussians
        self.out_dim = num_gaussians

        if gaussian_width is None:
            gaussian_width = self.get_gaussian_width(
                num_gaussians, maximum_initial_radius, intersection=intersection
            )
        if is_learnable:
            # use linspace to create the shifts of the gaussians in the range of 0 to 3
            # we use 3 as an initialisation because the bond length of c-c is 3 Bohr
            self.shift = torch.nn.Parameter(
                torch.linspace(0, maximum_initial_radius, num_gaussians)
            )
            # initialisation of the scale parameter as 1
            self.scale = torch.nn.Parameter(torch.ones(num_gaussians) * gaussian_width)
        else:
            self.register_buffer("scale", torch.ones(num_gaussians) * gaussian_width)
            self.register_buffer("shift", torch.linspace(0, maximum_initial_radius, num_gaussians))

        self.normalized = normalized

    def get_gaussian_width(
        self, num_gaussians, maximum_initial_radius, intersection: float = 0.5
    ) -> float:
        """Calculate the width of the gaussian functions based on the number of gaussians and the
        maximum initial radius of the gaussians such that neighboring gaussians have an specified
        intersection.

        Args:
            num_gaussians (int): Number of gaussian functions on which the edge features are calculated.
            maximum_initial_radius (float): The maximum initial radius of the gaussians.
            intersection (float, optional): The intersection of the gaussians. Defaults to 0.5.

        Returns:
            float: The width of the gaussian functions.
        """
        if num_gaussians == 1:
            return maximum_initial_radius / np.sqrt(-2 * np.log(intersection))
        else:
            diff = maximum_initial_radius / (num_gaussians - 1)
            return diff / np.sqrt(-8 * np.log(intersection))

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Calculates the embedding of the edge attributes based on the gaussian functions.

        .. math::
            e_{ij}^{k} = \\exp\\left(-\frac{1}{2} \\left(\frac{\\|r_i - r_j\\| - \\mu^k}
            {\\sigma^k} \right)^2\right).

        Args:
            pos (Tensor): positions of the atoms
            edge_index (Tensor): edge_index of the graph to calculate the edge attributes

        Returns:
            Tensor: The edge attributes. Shape: (E, num_gauss)
        """
        squared_diff = torch.square(norm - self.shift)
        squared_scale = torch.square(self.scale)

        # calculate the gaussian
        gaussian = torch.exp(-squared_diff / (2 * squared_scale))

        # if the gaussians should be normalized, divide by the normalization factor
        if self.normalized:
            gaussian = gaussian / (np.sqrt(2 * np.pi) * self.scale)

        return gaussian


class TrivialRadialEmbedding(RadialEmbedding):
    """A trivial radial embedding class that returns the norm of an edge vector as the
    embedding."""

    def __init__(self):
        """Initialises the class."""
        super().__init__()
        self.out_dim = 1

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Computes the embedding for the given input tensor.

        Args:
            norm (Tensor): The input tensor.

        Returns:
            Tensor: The computed embedding, which is the same as the input tensor.
        """
        return norm
