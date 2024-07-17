import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


class Envelope(Module):
    """A class representing the Envelope function which creates a smooth cutoff on the function
    value at x=1."""

    def __init__(self, p, use_cosine: bool = False):
        """Initializes the Envelope function.

        Args:
            p (float): The power parameter for the envelope function.
            use_cosine (bool, optional): Whether to use the cosine envelope function.
                Defaults to False.
        """

        super().__init__()

        self.p = p

        self.c_1 = -(p + 1) * (p + 2) / 2
        self.c_2 = p * (p + 2)
        self.c_3 = -p * (p + 1) / 2

        self.use_cosine = use_cosine

    def forward(self, x: Tensor) -> Tensor:
        """Applies the envelope function to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the envelope function.
        """
        if not self.use_cosine:
            x_envelope = (
                1
                + self.c_1 * x**self.p
                + self.c_2 * x ** (self.p + 1)
                + self.c_3 * x ** (self.p + 2)
            )
        else:
            x_envelope = 0.5 * (1 + torch.cos(np.pi * x))
        return torch.where(x < 1, x_envelope, torch.zeros_like(x_envelope))
