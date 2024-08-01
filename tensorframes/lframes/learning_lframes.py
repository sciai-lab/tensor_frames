from typing import Union

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torchvision.ops import MLP

from tensorframes.lframes.gram_schmidt import gram_schmidt
from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.envelope import EnvelopePoly
from tensorframes.nn.vector_neuron import VectorMLP


class LearnedGramSchmidtLFrames(MessagePassing):
    """The LearnedGramSchmidtLFrames class is a message passing neural network that learns local
    frames from its neighborhood."""

    def __init__(
        self,
        even_scalar_input_dim: int,
        radial_dim: int,
        hidden_channels: list[int],
        predict_o3: bool = True,
        cutoff: float | None = None,
        even_scalar_edge_dim: int = 0,
        concat_receiver: bool = True,
        exceptional_choice: str = "random",
        anchor_z_axis: bool = False,
        envelope: Union[torch.nn.Module, None] = EnvelopePoly(5),
        use_vector_mlp: bool = False,
        vector_in_channels: int = 16,
        vector_hidden_channels: list[int] = [32],
        **mlp_kwargs: dict,
    ) -> None:
        """Initialize the LearnedGramSchmidtLFrames model.

        Args:
            even_scalar_input_dim (int): The dimension of the scalar input.
            radial_dim (int): The dimension of the radial input.
            hidden_channels (list[int]): A list of integers representing the hidden channels in the MLP.
            predict_o3 (bool, optional): Whether to predict O3. Defaults to True.
            cutoff (float | None, optional): The cutoff value. Defaults to None. If not None, the envelope module is used.
            even_scalar_edge_dim (int, optional): The dimension of the edge input. Defaults to 0.
            concat_receiver (bool, optional): Whether to concatenate the receiver input to the mlp input. Defaults to True.
            exceptional_choice (str, optional): The exceptional choice, which is used by gram schmidt. Defaults to "random".
            anchor_z_axis (bool, optional): Whether to anchor the z-axis. Defaults to False.
            envelope (Union[torch.nn.Module, None], optional): The envelope module. Defaults to EnvelopePoly(5).
            **mlp_kwargs (dict): Additional keyword arguments for the MLP.
        """
        super().__init__()
        self.even_scalar_input_dim = even_scalar_input_dim
        self.radial_dim = radial_dim

        self.hidden_channels = hidden_channels.copy()

        self.predict_o3 = predict_o3

        if not self.predict_o3 and anchor_z_axis:
            raise ValueError("anchor_z_axis only works with predict_o3")

        self.anchor_z_axis = anchor_z_axis

        if self.predict_o3 and anchor_z_axis:
            self.num_pred_vecs = 2
        elif self.predict_o3:
            self.num_pred_vecs = 3
        else:
            self.num_pred_vecs = 2

        if use_vector_mlp:
            self.vector_mlp = VectorMLP(
                in_channels=vector_in_channels,
                hidden_channels=vector_hidden_channels,
                out_channels=self.num_pred_vecs,
            )
            self.num_pred_vecs = vector_in_channels

        self.hidden_channels.append(self.num_pred_vecs)

        self.cutoff = cutoff
        self.concat_receiver = concat_receiver
        self.exceptional_choice = exceptional_choice

        if self.cutoff is not None:
            self.envelope = envelope

        if concat_receiver:
            in_channels = self.even_scalar_input_dim * 2 + self.radial_dim
        else:
            in_channels = self.even_scalar_input_dim + self.radial_dim

        self.even_scalar_edge_dim = even_scalar_edge_dim
        in_channels += even_scalar_edge_dim

        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            **mlp_kwargs,
        )

    def forward(
        self, x: Tensor, radial: Tensor, pos: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> LFrames:
        """Forward pass of the learning_lframes module.

        Args:
            x (Tensor): Input tensor, can only be even scalars for the layer to be equivariant
            radial (Tensor): Radial tensor.
            pos (Tensor): Position tensor.
            edge_index (Tensor): Edge index tensor.
            edge_attr (Tensor): Edge attribute tensor, can only be even scalars for the layer to be equivariant

        Returns:
            LFrames: The local frames object containing the local frames.
        """
        vecs = self.propagate(edge_index, x=x, radial=radial, pos=pos, edge_attr=edge_attr)

        if self.anchor_z_axis:
            vecs = vecs.reshape(-1, self.num_pred_vecs + 1, 3)
            z_axis = vecs[:, -1, :]
            vecs = vecs[:, :-1, :]
        else:
            vecs = vecs.reshape(-1, self.num_pred_vecs, 3)

        if hasattr(self, "vector_mlp"):
            vecs = self.vector_mlp(vecs)

        if self.predict_o3:
            local_frames = gram_schmidt(
                vecs[:, 0, :],
                vecs[:, 1, :],
                vecs[:, 2, :] if not self.anchor_z_axis else z_axis,
                exceptional_choice=self.exceptional_choice,
            )
        else:
            local_frames = gram_schmidt(
                vecs[:, 0, :], vecs[:, 1, :], exceptional_choice=self.exceptional_choice
            )

        return LFrames(local_frames)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        radial: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Computes the message passed between two nodes in the graph.

        Args:
            x_i (Tensor): The input features of node is.
            x_j (Tensor): The input features of node j.
            radial (Tensor): The radial input.
            pos_i (Tensor): The position of node i.
            pos_j (Tensor): The position of node j.
            edge_attr (Tensor): The attributes of the edge between node i and node j.

        Returns:
            Tensor: The computed message.
        """
        if self.even_scalar_input_dim == 0:
            inp = radial
        else:
            if self.concat_receiver:
                inp = torch.cat([x_i, x_j, radial], dim=-1)
            else:
                inp = torch.cat([x_i, radial], dim=-1)

        if self.even_scalar_edge_dim > 0:
            inp = torch.cat([inp, edge_attr], dim=-1)

        mlp_out = self.mlp(inp)

        relative_vec = pos_j - pos_i
        relative_norm = torch.clamp(torch.linalg.norm(relative_vec, dim=-1, keepdim=True), 1e-6)
        relative_vec = relative_vec / relative_norm

        out = torch.einsum("ij,ik->ijk", mlp_out, relative_vec).reshape(-1, self.num_pred_vecs * 3)

        if self.cutoff is not None and self.envelope is not None:
            scaled_r = relative_norm / self.cutoff
            envelope = self.envelope(scaled_r)
            out = out * envelope

        if self.anchor_z_axis:
            # anchor the z-axis to the cross product of the x and y axis
            if self.cutoff is not None and self.envelope is not None:
                weighted_relative_vec = relative_vec * envelope
            else:
                weighted_relative_vec = relative_vec
            out = torch.cat([out, weighted_relative_vec], dim=-1)

        return out
