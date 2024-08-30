from torch import Tensor
from torch.nn import Module

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.reps import Reps


class FromGlobalToLocalFrame(Module):
    """Transforms a tensor with a given representation from a global frame to a local frame."""

    def __init__(self, reps: Reps) -> None:
        """Initialize the FromGlobalToLocalFrame Module.

        Args:
            reps (Reps): The representation which is used to transform the features.
        """
        super().__init__()
        self.reps = reps
        self.trafo_class = self.reps.get_transform_class()

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

    def __init__(self, reps: Reps) -> None:
        """Initialize the FromLocalToGlobalFrame Module.

        Args:
            reps (Reps): The representation which is used to transform the features.
        """
        super().__init__()
        self.reps = reps
        self.trafo_class = self.reps.get_transform_class()

    def forward(self, x: Tensor, lframes: LFrames) -> Tensor:
        """Transforms the features x from a local frame to a global frame.

        Args:
            x (Tensor): The input tensor.
            lframes (LFrames): The local frames.

        Returns:
            Tensor: The output tensor.
        """
        return self.trafo_class(x, lframes.inverse_lframes())
