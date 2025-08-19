import torch
from torch import Tensor


class Swish(torch.nn.Module):
    """Swish activation function."""

    def __init__(self, beta: float = 1.0, beta_learnable: bool = True) -> None:
        """Initializes the Activation class.

        Args:
            beta (float, optional): The value of beta. Defaults to 1.0.
            beta_learnable (bool, optional): Whether beta is learnable or not. Defaults to True.
        """
        super().__init__()
        if beta_learnable:
            self.beta = torch.nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("beta", Tensor([beta]))

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the Swish activation function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the Swish activation function.
        """
        return x * torch.sigmoid(self.beta * x)


class ActGLU(torch.nn.Module):
    """Activation function gated linear unit (SwiGLU) activation function."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        include_bias: bool = True,
        activation_function: torch.nn.Module = Swish(),
    ) -> None:
        """Initialize the ActGLU activation layer.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            include_bias (bool, optional): Whether to include a bias term. Defaults to True.
            activation_function (torch.nn.Module, optional): The activation function to use. Defaults to Swish().
        """
        super().__init__()
        self.lin_1 = torch.nn.Linear(in_dim, out_dim, bias=include_bias)
        self.lin_2 = torch.nn.Linear(in_dim, out_dim, bias=include_bias)
        self.activation_function = activation_function

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the ActGLU activation function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the SwiGLU activation function.
        """
        return self.lin_1(x) * self.activation_function(self.lin_2(x))
