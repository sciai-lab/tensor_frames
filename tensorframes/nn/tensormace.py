from typing import Union

import torch
from torch import Tensor
from torch_geometric.nn import LayerNorm, MessagePassing

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.linear import EdgeLinear
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


class TensorMACE(MessagePassing):
    """The TensorMACE model.

    TODO: Make it more verbose
    """

    def __init__(
        self,
        in_tensor_reps: Union[TensorReps, Irreps],
        out_tensor_reps: Union[TensorReps, Irreps],
        edge_emb_dim: int,
        hidden_dim: int,
        order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a TensorMace object.

        Args:
            in_tensor_reps (Union[TensorReps, Irreps]): The input tensor representations.
            out_tensor_reps (Union[TensorReps, Irreps]): The output tensor representations.
            edge_emb_dim (int): The dimension of the edge embeddings.
            hidden_dim (int): The dimension of the hidden layer.
            order (int): The message passing body order. Defaults to 3.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__(
            params_dict={
                "x": {"type": "local", "rep": in_tensor_reps},
                "edge_embedding": {"type": None, "rep": None},
            }
        )
        self.in_tensor_reps = in_tensor_reps
        self.out_tensor_reps = out_tensor_reps
        self.in_dim = in_tensor_reps.dim
        self.out_dim = out_tensor_reps.dim
        self.edge_emb_dim = edge_emb_dim
        self.order = order
        self.hidden_dim = hidden_dim

        self.lin_1 = EdgeLinear(self.in_dim, self.edge_emb_dim, self.hidden_dim)

        self.param_1 = torch.nn.Parameter(
            torch.randn(self.hidden_dim, self.order, self.hidden_dim)
        )
        self.bias_1 = torch.nn.Parameter(torch.randn(self.hidden_dim, self.order))

        self.param_2 = torch.nn.Parameter(torch.randn(self.out_dim, self.order, self.hidden_dim))
        self.bias_2 = torch.nn.Parameter(torch.randn(self.out_dim, self.order))

        self.lin_skip = torch.nn.Linear(self.in_dim, self.out_dim)

        self.layer_norm = LayerNorm(self.in_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_embedding: Tensor,
        lframes: LFrames,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the TensorMACE layer.

        Args:
            x (Tensor): Input node features of shape (num_nodes, input_dim).
            edge_index (Tensor): Graph edge indices of shape (2, num_edges).
            edge_embedding (Tensor): Edge embeddings of shape (num_edges, edge_dim).
            lframes (LFrames): LFrames object containing the local frames for each node.
            batch (Tensor, optional): Batch tensor of shape (num_nodes,). Defaults to None.

        Returns:
            Tensor: Output node features of shape (num_nodes, output_dim).
        """
        skip = x

        x = self.layer_norm(x, batch)

        # calculate the As
        A = self.propagate(edge_index, x=x, edge_embedding=edge_embedding, lframes=lframes)

        # calculate the Bs
        tmp = torch.einsum("ih, onh -> ion", A, self.param_1) + self.bias_1

        # TODO: Why is for loop faster than cumprod?
        B = torch.zeros((x.shape[0], self.hidden_dim, self.order), device=x.device, dtype=x.dtype)
        for i in range(self.order):
            B[:, :, i] = torch.prod(tmp[:, :, : i + 1], dim=-1)

        # B = torch.cumprod(tmp, dim=-1)

        # calculate the new node features
        x = (torch.einsum("ihn, onh -> ion", B, self.param_2) + self.bias_2).sum(dim=-1)

        return self.dropout(x) + self.lin_skip(skip)

    def message(self, x_j: Tensor, edge_embedding: Tensor) -> Tensor:
        """This method performs a message passing operation.

        Args:
            x_j (Tensor): The input tensor for node j.
            edge_embedding (Tensor): The edge embedding tensor.

        Returns:
            Tensor: The result of the message passing operation.
        """
        return self.lin_1(x_j, edge_embedding)


if __name__ == "__main__":
    # Test the TensorMACE model
    from tensorframes.lframes.lframes import LFrames
    from tensorframes.reps.irreps import Irreps
    from tensorframes.reps.tensorreps import TensorReps

    in_tensor_reps = TensorReps("10x0n+5x1n+2x2n")
    out_tensor_reps = Irreps("10x0n+5x1n+2x2n")
    edge_emb_dim = 32
    hidden_dim = 64
    order = 3
    dropout = 0.1

    model = TensorMACE(in_tensor_reps, out_tensor_reps, edge_emb_dim, hidden_dim, order, dropout)

    x = torch.randn(10, in_tensor_reps.dim)
    # create a big edge_index
    edge_index = torch.randint(0, 10, (2, 100))
    edge_embedding = torch.randn(100, edge_emb_dim)
    import e3nn

    lframes = LFrames(e3nn.o3.rand_matrix(10))

    out = model(x, edge_index, edge_embedding, lframes)
    print(out)
