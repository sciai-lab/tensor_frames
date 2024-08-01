import torch


class VectorLinear(torch.nn.Module):
    """A linear layer for vectors.

    This layer reshapes the input tensor to a 3D tensor and applies a linear layer to each vector.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes a VectorLinear object.

        Args:
            in_channels (int): The number of input vectors.
            out_channels (int): The number of output vectors.
        """
        super().__init__()
        self.in_channels = in_channels
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the vector linear layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the vector linear layer.
        """
        return self.linear(x.reshape(-1, 3, self.in_channels)).reshape(
            -1, self.linear.out_features * 3
        )


class VectorReLU(torch.nn.Module):
    """A ReLU like layer for vectors."""

    def __init__(self, channels: int) -> None:
        """Initializes a VectorReLU object.

        Args:
            channels (int): The number of input channels.

        Returns:
            None
        """
        super().__init__()
        self.channels = channels
        self.U = VectorLinear(channels, channels)
        self.W = VectorLinear(channels, channels)

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the VectorReLU layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the VectorReLU layer.
        """
        k = self.U(x).reshape(-1, self.channels, 3)
        q = self.W(x).reshape(-1, self.channels, 3)

        # calculate scalar product of k and q
        dot_product = torch.sum(k * q, dim=-1)

        # calculate norm of k
        norm_k = torch.linalg.norm(k, dim=-1)

        out = torch.where(dot_product > 0, q, q - k * dot_product / norm_k**2)

        return out


class VectorNorm(torch.nn.Module):
    """A layer that normalizes vectors."""

    def __init__(self, channels: int) -> None:
        """Initializes a VectorNorm object.

        Args:
            channels (int): The number of channels in the input tensor.

        Returns:
            None
        """
        super().__init__()
        self.channels = channels
        self.layer_norm = torch.nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the forward pass of the VectorNorm layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, 3).

        Returns:
            torch.Tensor: The output tensor after applying the VectorNorm layer.
        """
        x = x.reshape(-1, self.channels, 3)

        norm_x = torch.linalg.norm(x, dim=-1)
        layer_norm = self.layer_norm(norm_x)
        out = x * layer_norm / norm_x

        return out


class VectorMLP(torch.nn.Module):
    """A multi-layer perceptron for vectors."""

    def __init__(self, in_channels: int, hidden_channels: list[int], out_channels: int) -> None:
        """Initializes a VectorMLP object.

        Args:
            in_channels (int): The number of input channels.
            hidden_channels (list[int]): A list of integers representing the number of hidden channels for each layer.
            out_channels (int): The number of output channels.

        Returns:
            None
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels.copy()
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()

        for i in range(len(hidden_channels)):
            self.layers.append(
                VectorLinear(
                    in_channels=self.in_channels if i == 0 else hidden_channels[i - 1],
                    out_channels=hidden_channels[i],
                )
            )
            self.layers.append(VectorReLU(channels=hidden_channels[i]))
            self.layers.append(VectorNorm(channels=hidden_channels[i]))
            in_channels = hidden_channels[i]
        self.layers.append(VectorLinear(hidden_channels[-1], out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through all the layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x
