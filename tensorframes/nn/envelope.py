import torch
from torch import Tensor
from torch.nn import Module


class EnvelopePoly(Module):
    """A class representing an envelope polynomial module."""

    def __init__(self, p):
        """Initializes an instance of the Envelope class.

        Args:
            p (int): The value of the highest exponent in the polynomial.
        """
        super().__init__()

        self.p = p

        self.c_1 = -(p + 1) * (p + 2) / 2
        self.c_2 = p * (p + 2)
        self.c_3 = -p * (p + 1) / 2

    def forward(self, x: Tensor):
        """Forward pass of the envelope polynomial module.

        Applies the envelope polynomial function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the envelope polynomial function.
        """
        x_envelope = (
            1
            + self.c_1 * x**self.p
            + self.c_2 * x ** (self.p + 1)
            + self.c_3 * x ** (self.p + 2)
        )
        return torch.where(x < 1, x_envelope, torch.zeros_like(x_envelope))


class EnvelopeCosine(Module):
    """A PyTorch module that applies an envelope function to the input tensor using the cosine
    function.

    The envelope function is defined as 0.5 * (1 + cos(pi * x)) for values of x less than 1, and 0
    for other values.
    """

    def __init__(self):
        """Initializes a new instance of the Envelope class."""
        super().__init__()

    def forward(self, x: Tensor):
        """Applies the envelope function to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The input tensor after applying the envelope function.
        """
        x_envelope = 0.5 * (1 + torch.cos(torch.pi * x))
        return torch.where(x < 1, x_envelope, torch.zeros_like(x_envelope))
