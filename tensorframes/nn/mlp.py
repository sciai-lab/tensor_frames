from typing import Union

import torch
from torch import Tensor
from torchvision.ops import MLP as TorchMLP

from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


class MLP(torch.nn.Module):
    """An MLP module which uses reps for the input and output dimensions."""

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        out_reps: Union[TensorReps, Irreps],
        hidden_layers,
        **mlp_kwargs
    ) -> None:
        """Initialize the MLP class.

        Args:
            in_reps (Union[TensorReps, Irreps]): The input representations.
            out_reps (Union[TensorReps, Irreps]): The output representations.
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            **mlp_kwargs: Additional keyword arguments to be passed to the MLP constructor.
        """

        super().__init__()
        self.in_reps = in_reps
        self.out_reps = out_reps
        self.hidden_layers = hidden_layers.copy()
        self.hidden_layers.append(out_reps.dim)

        self.mlp = TorchMLP(in_reps.dim, hidden_layers, **mlp_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the MLP network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the MLP network.
        """
        return self.mlp(x)
