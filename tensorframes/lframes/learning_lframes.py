from typing import Union

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torchvision.ops import MLP

from tensorframes.lframes.gram_schmidt import gram_schmidt
from tensorframes.nn.embedding.envelope import EnvelopePoly


class LearnedGramSchmidtLFrames(MessagePassing):
    def __init__(
        self,
        scalar_input_dim: int,
        radial_dim: int,
        hidden_channels: list[int],
        predict_o3: bool = True,
        cutoff: float | None = None,
        edge_dim: int = 0,
        concat_receiver: bool = True,
        exceptional_choice: str = "random",
        anchor_z_axis: bool = False,
        envelope: Union[torch.nn.Module, None] = EnvelopePoly(5),
        **mlp_kwargs: dict,
    ) -> None:
        """Initialize the LearningLFrames model.

        Args:
            scalar_input_dim (int): The dimension of the scalar input.
            radial_dim (int): The dimension of the radial input.
            hidden_channels (list[int]): A list of integers representing the hidden channels in the MLP.
            predict_o3 (bool, optional): Whether to predict O3. Defaults to True.
            cutoff (float | None, optional): The cutoff value. Defaults to None. If not None, the envelope module is used.
            edge_dim (int, optional): The dimension of the edge input. Defaults to 0.
            concat_receiver (bool, optional): Whether to concatenate the receiver input to the mlp input. Defaults to True.
            exceptional_choice (str, optional): The exceptional choice, which is used by gram schmidt. Defaults to "random".
            anchor_z_axis (bool, optional): Whether to anchor the z-axis. Defaults to False.
            envelope (Union[torch.nn.Module, None], optional): The envelope module. Defaults to EnvelopePoly(5).
            **mlp_kwargs (dict): Additional keyword arguments for the MLP.
        """
        super().__init__()
        self.scalar_input_dim = scalar_input_dim
        self.radial_dim = radial_dim

        self.hidden_channels = hidden_channels.copy()

        self.predict_o3 = predict_o3

        if self.predict_o3:
            self.num_pred_vecs = 3
        else:
            self.num_pred_vecs = 2

        self.anchor_z_axis = anchor_z_axis
        if self.anchor_z_axis:
            assert (
                self.predict_o3
            ), f"anchor_z_axis only works with predict_o3, predict_o3 = {predict_o3}"
            self.num_pred_vecs -= 1

        self.hidden_channels.append(self.num_pred_vecs)

        self.cutoff = cutoff
        self.concat_receiver = concat_receiver
        self.exceptional_choice = exceptional_choice

        if self.cutoff is not None:
            self.envelope = envelope

        if concat_receiver:
            in_channels = self.scalar_input_dim * 2 + self.radial_dim
        else:
            in_channels = self.scalar_input_dim + self.radial_dim

        self.edge_dim = edge_dim
        in_channels += edge_dim

        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            **mlp_kwargs,
        )

    def forward(
        self, x: Tensor, radial: Tensor, pos: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> torch.Tensor:
        """Forward pass of the learning_lframes module.

        Args:
            x (Tensor): Input tensor.
            radial (Tensor): Radial tensor.
            pos (Tensor): Position tensor.
            edge_index (Tensor): Edge index tensor.
            edge_attr (Tensor): Edge attribute tensor.

        Returns:
            torch.Tensor: Local frames tensor.
        """
        vecs = self.propagate(edge_index, x=x, radial=radial, pos=pos, edge_attr=edge_attr)

        # calculate the local frames
        if self.anchor_z_axis:
            vecs = vecs.reshape(-1, self.num_pred_vecs + 1, 3)
        else:
            vecs = vecs.reshape(-1, self.num_pred_vecs, 3)

        if self.predict_o3:
            local_frames = gram_schmidt(
                vecs[:, 0, :],
                vecs[:, 1, :],
                vecs[:, 2, :],
                exceptional_choice=self.exceptional_choice,
            )
        else:
            local_frames = gram_schmidt(
                vecs[:, 0, :], vecs[:, 1, :], exceptional_choice=self.exceptional_choice
            )

        return local_frames

    def message(self, x_i, x_j, radial, pos_i, pos_j, edge_attr):
        if self.scalar_input_dim == 0:
            inp = radial
        else:
            if self.concat_receiver:
                inp = torch.cat([x_i, x_j, radial], dim=-1)
            else:
                inp = torch.cat([x_i, radial], dim=-1)

        if self.edge_dim > 0:
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
