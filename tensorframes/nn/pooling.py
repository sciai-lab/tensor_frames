import torch
import torch_geometric
import torch_geometric.utils
from torch import Tensor
from torch_geometric.nn import global_add_pool

from tensorframes.reps.reps import Reps


class GlobalAttentionPooling(torch.nn.Module):
    """The GlobalAttentionPooling module.

    This module takes a tensor and aggregates the values in each batch using a global attention mechanism.
    TODO: at the moment this only works if one only wants a scalar output in the network.
    """

    def __init__(
        self,
        in_reps: Reps,
        out_reps: Reps,
        bias: bool = False,
    ) -> None:
        """Initialize the GlobalAttentionPooling module.

        Args:
            in_reps (list): List of input representations.
            out_reps (list): List of output representations.
            bias (bool, optional): Whether to include bias terms. Defaults to False.
        """
        super().__init__()
        self.in_reps = in_reps
        self.out_reps = out_reps

        self.query = torch.nn.Parameter(torch.randn(in_reps.dim))
        self.key = torch.nn.Linear(in_reps.dim, in_reps.dim, bias=bias)
        self.value = torch.nn.Linear(in_reps.dim, out_reps.dim, bias=bias)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """Applies the GlobalAttentionPooling module.

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        q = self.query
        k = self.key(x)
        v = self.value(x)

        softmax = torch_geometric.utils.softmax(q @ k.transpose(-1, -2), batch, dim=-1)

        x = torch.einsum("i,ij->ij", softmax, v)

        out = global_add_pool(x, batch)

        return out
