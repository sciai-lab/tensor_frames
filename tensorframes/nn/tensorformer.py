from typing import Type, Union

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.nn import LayerNorm
from torch_geometric.utils import softmax
from torchvision.ops import MLP

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.envelope import EnvelopePoly
from tensorframes.nn.linear import EdgeLinear, HeadedLinear
from tensorframes.nn.tfmessage_passing import TFMessagePassing
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


class TensorFormer(TFMessagePassing):
    """The TensorFormer model.

    TODO: insert arxiv paper reference if we have one.
    """

    def __init__(
        self,
        tensor_reps: Union[TensorReps, Irreps],
        num_heads: int,
        hidden_layers: list[int],
        hidden_value_dim: int,
        hidden_scalar_dim: int,
        hidden_activation: Type[Module] = torch.nn.SiLU,
        edge_embedding_dim: int = 0,
        scalar_activation_function: Module = torch.nn.LeakyReLU(),
        value_activation_function: Module = torch.nn.SiLU(),
        dropout_attention: float = 0.0,
        dropout_mlp: float = 0.0,
        stochastic_depth: float = 0.0,
        radial_cutoff: float = 5.0,
        envelope: Module = EnvelopePoly(5),
        softmax: bool = False,
        attention_weight_dropout: float = 0.0,
    ) -> None:
        """Initialize the TensorFormer model.

        Args:
            tensor_reps (Union[TensorReps, Irreps]): The representation of the features.
            num_heads (int): The number of attention heads.
            hidden_layers (list[int]): The sizes of the hidden layers in the MLP.
            hidden_value_dim (int): The dimension of the hidden value vectors.
            hidden_scalar_dim (int): The dimension of the hidden scalar vectors.
            hidden_activation (Module, optional): The activation function for the hidden layers. Defaults to torch.nn.SiLU().
            edge_embedding_dim (int, optional): The dimension of the edge embeddings. Defaults to 0.
            scalar_activation_function (Module, optional): The activation function for the scalar vectors. Defaults to torch.nn.LeakyReLU().
            dropout_attention (float, optional): The dropout rate for the values after attention. Defaults to 0.0.
            dropout_mlp (float, optional): The dropout rate for MLP layers. Defaults to 0.0.
            stochastic_depth (float, optional): The stochastic depth rate. Must be between 0 and 1. Defaults to 0.0.
            radial_cutoff (float, optional): The radial cutoff distance. Defaults to 5.0.
            envelope (Module, optional): The envelope function for radial basis functions. Defaults to EnvelopePoly(5).
            softmax (bool, optional): Whether to apply softmax to attention weights if not uses SiLU. Defaults to False.
            attention_weight_dropout (float, optional): The dropout rate for attention weights. Defaults to 0.0.
        """

        super().__init__(
            params_dict={
                "x": {"type": "local", "rep": tensor_reps},
                "pos": {"type": "global", "rep": TensorReps("1x1n")},
                "edge_embedding": {"type": None, "rep": None},
            }
        )

        self.tensor_reps = tensor_reps
        self.dim = tensor_reps.dim

        self.num_heads = num_heads

        self.hidden_value_dim = hidden_value_dim
        self.hidden_scalar_dim = hidden_scalar_dim

        self.lin_1 = EdgeLinear(
            self.dim * 2, edge_embedding_dim, (hidden_value_dim + hidden_scalar_dim) * num_heads
        )

        self.scalar_norm = LayerNorm(hidden_scalar_dim)
        self.act_scalar = scalar_activation_function
        self.lin_scalar = HeadedLinear(hidden_scalar_dim, 1, num_heads)

        self.act_value = value_activation_function
        self.lin_value = EdgeLinear(self.hidden_value_dim, edge_embedding_dim, hidden_value_dim)

        self.lin_out = Linear(self.hidden_value_dim * num_heads, self.dim)

        mlp_hidden_layers = hidden_layers.copy()

        mlp_hidden_layers.append(self.dim)

        self.mlp = MLP(
            self.dim,
            mlp_hidden_layers,
            activation_layer=hidden_activation,
        )

        self.layer_norm_1 = LayerNorm(self.dim)
        self.layer_norm_2 = LayerNorm(self.dim)

        self.dropout_attention = torch.nn.Dropout(dropout_attention)
        self.dropout_mlp = torch.nn.Dropout(dropout_mlp)

        self.envelope = envelope
        self.radial_cutoff = radial_cutoff
        self.softmax = softmax
        self.silu = torch.nn.SiLU()

        assert 0 <= stochastic_depth < 1, "Stochastic depth must be between 0 and 1"

        self.stochastic_depth = stochastic_depth
        self.attention_weight_dropout = attention_weight_dropout
        self.attention_weight_dropout_layer = torch.nn.Dropout(attention_weight_dropout)

    def forward(
        self,
        x: Tensor,
        lframes: LFrames,
        edge_index: Tensor,
        pos: Tensor,
        edge_embedding: Tensor,
        batch: Tensor | None = None,
    ):
        """Forward pass of the TensorFormer module. TODO: insert arxiv paper reference if we have
        one.

        Args:
            x (Tensor): Input tensor.
            lframes (LFrames): LFrames object.
            edge_index (Tensor): Edge index tensor.
            pos (Tensor): Position tensor.
            edge_embedding (Tensor): Edge embedding tensor.
            batch (Tensor, optional): Batch tensor. Defaults to None.

        Returns:
            Tensor: Output tensor.
        """
        if self.stochastic_depth > 0.0:
            if self.training and torch.rand(1) > 1 - self.stochastic_depth:
                return x

        skip_x = x

        x = self.layer_norm_1(x, batch)

        attention_output = self.propagate(
            edge_index,
            x=x,
            lframes=lframes,
            pos=pos,
            edge_embedding=edge_embedding,
        )

        attention_output = self.lin_out(attention_output)

        attention_output = self.dropout_attention(attention_output)

        attention_output = attention_output + skip_x
        skip_x = attention_output

        attention_output = self.layer_norm_2(attention_output, batch)

        out = self.mlp(attention_output)

        out = self.dropout_mlp(out)

        out = out + skip_x

        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        index: Tensor,
        ptr: Tensor,
        size_i: int,
        edge_embedding: Tensor,
    ):
        """Calculates the message passing operation for the Tensorformer model.

        Args:
            x_i (Tensor): Input tensor for node i.
            x_j (Tensor): Input tensor for node j.
            pos_i (Tensor): Position tensor for node i.
            pos_j (Tensor): Position tensor for node j.
            index (Tensor): Index tensor.
            ptr (Tensor): Pointer tensor.
            size_i (int): Size of node i.
            edge_embedding (Tensor): Edge embedding tensor.

        Returns:
            Tensor: Output tensor after message passing operation.
        """
        # calculate envelope function
        norm_pos = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        envelope = self.envelope(norm_pos / self.radial_cutoff)

        x = torch.cat([x_i, x_j], dim=-1)

        x = self.lin_1(x, edge_embedding)

        tmp = x.view(-1, self.num_heads, self.hidden_value_dim + self.hidden_scalar_dim)

        values = tmp[:, :, : self.hidden_value_dim]
        scalars = tmp[:, :, self.hidden_value_dim :]

        # scalar path
        scalars = self.scalar_norm(scalars)
        scalars = self.act_scalar(scalars)
        scalars = self.lin_scalar(scalars)

        if self.softmax:
            alpha = softmax(scalars / (self.hidden_value_dim**0.5), index, ptr, size_i)
        else:
            alpha = self.silu(scalars)

        if self.attention_weight_dropout > 0.0:
            alpha = self.attention_weight_dropout_layer(alpha)

        # value path
        value = self.act_value(values)
        value = self.lin_value(value, edge_embedding)

        out = value * alpha.view(-1, self.num_heads, 1)
        out = out.contiguous().view(-1, self.num_heads * self.hidden_value_dim) * envelope

        return out


if __name__ == "__main__":
    # test the TensorFormer model
    tensor_reps = TensorReps("10x0n+5x1n+2x2n")
    num_heads = 4
    hidden_layers = [128, 64]
    hidden_value_dim = 64
    hidden_scalar_dim = 64
    edge_embedding_dim = 32

    model = TensorFormer(
        tensor_reps=tensor_reps,
        num_heads=num_heads,
        hidden_layers=hidden_layers,
        hidden_value_dim=hidden_value_dim,
        hidden_scalar_dim=hidden_scalar_dim,
        edge_embedding_dim=edge_embedding_dim,
    )

    # create test data
    x = torch.randn(10, tensor_reps.dim)
    pos = torch.randn(10, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    edge_embedding = torch.randn(10, edge_embedding_dim)
    import e3nn

    lframes_mat = e3nn.o3.rand_matrix(10)
    lframes = LFrames(lframes_mat)

    # forward pass
    out = model(x, lframes, edge_index, pos, edge_embedding)
    print(out.shape)
