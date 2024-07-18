from typing import Union

from torch import Tensor
from torch.nn import Module

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


class FromGlobalToLocalFrame(Module):
    """Transforms a tensor with a given representation from a global frame to a local frame."""

    def __init__(self, rep: Union[TensorReps, Irreps]) -> None:
        """Initialize the FromGlobalToLocalFrame Module.

        Args:
            rep (Union[TensorReps, Irreps]): The representation which is used to transform the features.
        """
        super().__init__()
        self.rep = rep
        self.trafo_class = rep.get_transform_class()

    def forward(self, x: Tensor, lframes: LFrames) -> Tensor:
        """Transforms the features x from a global frame to a local frame.

        Args:
            x (Tensor): The input tensor.
            lframes (LFrames): The local frames.

        Returns:
            Tensor: The output tensor.
        """
        return self.trafo_class(x, lframes)


class FromLocalToGlobalFrame(Module):
    """Transforms a tensor with a given representation from a local frame to a global frame."""

    def __init__(self, rep: Union[TensorReps, Irreps]) -> None:
        """Initialize the FromLocalToGlobalFrame Module.

        Args:
            rep (Union[TensorReps, Irreps]): The representation which is used to transform the features.
        """
        super().__init__()
        self.rep = rep
        self.trafo_class = rep.get_transform_class()

    def forward(self, x: Tensor, lframes: LFrames) -> Tensor:
        """Transforms the features x from a local frame to a global frame.

        Args:
            x (Tensor): The input tensor.
            lframes (LFrames): The local frames.

        Returns:
            Tensor: The output tensor.
        """
        return self.trafo_class(x, lframes.inv)
