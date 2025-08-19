from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import MessagePassing, radius
from torch_geometric.typing import PairTensor

from tensorframes.lframes.classical_lframes import LFramesPredictionModule
from tensorframes.lframes.gram_schmidt import double_cross_orthogonalize, gram_schmidt, double_gradient_safe_norm
from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.embedding.radial import RadialEmbedding
from tensorframes.nn.envelope import EnvelopePoly
from tensorframes.nn.mlp import MLPWrapped
from tensorframes.reps import Irreps, TensorReps
from tensorframes.reps.utils import extract_even_scalar_mask_from_reps


class LearnedGramSchmidtLFrames(MessagePassing, LFramesPredictionModule):
    """The LearnedGramSchmidtLFrames class is a message passing neural network that learns local
    frames from its neighborhood."""

    def __init__(
        self,
        even_scalar_input_dim: int,
        radial_dim: int,
        hidden_channels: list[int],
        predict_o3: bool = True,
        cutoff: Union[float, None] = None,
        even_scalar_edge_dim: int = 0,
        concat_receiver: bool = True,
        exceptional_choice: str = "random",
        anchor_z_axis: bool = True,
        fix_gravitational_axis: bool = False,
        gravitational_axis_index: int = 1,
        envelope: Union[torch.nn.Module, None] = EnvelopePoly(5),
        use_double_cross_product: bool = False,
        eps: float = 1e-6,
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
            fix_gravitational_axis (bool, optional): Whether to fix the gravitational axis. Defaults to False.
            gravitational_axis_index (int, optional): The index of the gravitational axis. Defaults to 1.
            envelope (Union[torch.nn.Module, None], optional): The envelope module. Defaults to EnvelopePoly(5).
            use_double_cross_product (bool, optional): Whether to use the double cross product method to compute the third vector. Defaults to False.
            **mlp_kwargs (dict): Additional keyword arguments for the MLP.
        """
        super().__init__()
        self.eps = eps
        self.even_scalar_input_dim = even_scalar_input_dim
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

        if fix_gravitational_axis:
            gravitational_axis = torch.zeros(3)
            gravitational_axis[gravitational_axis_index] = 1.0
            self.register_buffer("gravitational_axis", gravitational_axis)
            self.num_pred_vecs -= 1

            # find the even permutation where index_order[gravitational_axis_index] is 0:
            index_order = [0, 1, 2]
            for i in range(3):
                current_index_order = np.roll(index_order, i)
                if current_index_order[gravitational_axis_index] == 0:
                    self.index_order = current_index_order.tolist()
        else:
            self.gravitational_axis = None
            self.index_order = None

        self.hidden_channels.append(self.num_pred_vecs)

        self.cutoff = cutoff
        self.concat_receiver = concat_receiver
        self.exceptional_choice = exceptional_choice
        self.use_double_cross_product = use_double_cross_product

        if self.cutoff is not None:
            self.envelope = envelope

        if concat_receiver:
            in_channels = self.even_scalar_input_dim * 2 + self.radial_dim
        else:
            in_channels = self.even_scalar_input_dim + self.radial_dim

        self.even_scalar_edge_dim = even_scalar_edge_dim
        in_channels += even_scalar_edge_dim

        self.mlp = MLPWrapped(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            **mlp_kwargs,
        )

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        radial: Tensor,
        pos: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: Union[Tensor, None] = None,
        batch: Union[Tensor, PairTensor, None] = None,
    ) -> LFrames:
        """Forward pass of the learning_lframes module.

        Args:
            x (Tensor, PairTensor): Input tensor, can only be even scalars for the layer to be equivariant
            radial (Tensor): Radial tensor.
            pos (Tensor, PairTensor): Position tensor.
            edge_index (Tensor): Edge index tensor.
            edge_attr (Tensor, None, optional): Edge attribute tensor, can only be even scalars for the layer to be equivariant. Defaults to None.
            batch (Tensor, PairTensor, None, optional): Batch tensor. Defaults to None.

        Returns:
            LFrames: The local frames object containing the local frames.
        """

        # needed for index-magic in message
        if isinstance(batch, tuple):
            batch = (
                None if batch[0] is None else batch[0].view(-1, 1),
                None if batch[1] is None else batch[1].view(-1, 1),
            )
        else:
            batch = None if batch is None else batch.view(-1, 1)

        vecs = self.propagate(
            edge_index, x=x, radial=radial, pos=pos, edge_attr=edge_attr, batch=batch
        )

        # calculate the local frames
        if self.anchor_z_axis:
            vecs = vecs.reshape(-1, self.num_pred_vecs + 1, 3)
        else:
            vecs = vecs.reshape(-1, self.num_pred_vecs, 3)

        if self.predict_o3:
            if self.gravitational_axis is None:
                vec1 = vecs[:, 0, :]
                vec2 = vecs[:, 1, :]
                vec3 = vecs[:, 2, :]
            else:
                vec1 = self.gravitational_axis[None, :].repeat(vecs.shape[0], 1)
                vec2 = vecs[:, 0, :]
                vec3 = vecs[:, 1, :]

            if self.use_double_cross_product:
                local_frames = double_cross_orthogonalize(
                    vec1, vec2, vec3, exceptional_choice=self.exceptional_choice, eps=self.eps
                )
            else:
                local_frames = gram_schmidt(
                    vec1,
                    vec2,
                    vec3,
                    exceptional_choice=self.exceptional_choice,
                    eps=self.eps,
                )
        else:
            if self.gravitational_axis is None:
                vec1 = vecs[:, 0, :]
                vec2 = vecs[:, 1, :]
            else:
                vec1 = self.gravitational_axis[None, :].repeat(vecs.shape[0], 1)
                vec2 = vecs[:, 0, :]

            if self.use_double_cross_product:
                local_frames = double_cross_orthogonalize(
                    vec1, vec2, exceptional_choice=self.exceptional_choice, eps=self.eps
                )
            else:
                local_frames = gram_schmidt(
                    vec1,
                    vec2,
                    exceptional_choice=self.exceptional_choice,
                    eps=self.eps,
                )

        # permute the axes to be at the correct index position:
        if self.index_order is not None:
            local_frames = local_frames[:, self.index_order]
        return LFrames(local_frames)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        radial: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        batch_j: Union[Tensor, None],
        edge_attr: Union[Tensor, None],
    ) -> Tensor:
        """Computes the message passed between two nodes in the graph.

        Args:
            x_i (Tensor): The input features of node is.
            x_j (Tensor): The input features of node j.
            radial (Tensor): The radial input.
            pos_i (Tensor): The position of node i.
            pos_j (Tensor): The position of node j.
            batch_j (Tensor, None): The batch index of node j.
            edge_attr (Tensor, None): The attributes of the edge between node i and node j.

        Returns:
            Tensor: The computed message.
        """
        if self.even_scalar_input_dim == 0:
            inp = radial
        else:
            if self.concat_receiver:
                inp = torch.cat([x_i, x_j, radial], dim=-1)
            else:
                inp = torch.cat([x_j, radial], dim=-1)

        if self.even_scalar_edge_dim > 0 and edge_attr is not None:
            inp = torch.cat([inp, edge_attr], dim=-1)

        mlp_out = self.mlp(x=inp, batch=None if batch_j is None else batch_j.flatten())

        relative_vec = pos_j - pos_i
        relative_norm = double_gradient_safe_norm(relative_vec, eps=self.eps)
        relative_vec = relative_vec / torch.clamp(relative_norm, min=self.eps)

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


class WrappedLearnedLFrames(torch.nn.Module):
    """The WrappedLearnedLocalFramesModule is a wrapper around the LearnedGramSchmidtLFrames
    module."""

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        hidden_channels: list[int],
        radial_module: RadialEmbedding,
        max_radius: Union[float, None] = None,
        edge_attr_tensor_reps: Union[TensorReps, Irreps, None] = None,
        max_num_neighbors: int = 64,
        transform_into_lframes: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the WrappedLearnedLocalFramesModule.

        Args:
            in_reps (Union[TensorReps, Irreps]): The input representations.
            hidden_channels (list[int]): The hidden channels for the LearnedGramSchmidtLFrames module.
            max_radius (float, optional): The maximum radius for the neighbor search. Defaults to None.
            radial_module (torch.nn.Module, optional): The radial module for the radial embedding. Defaults to None.
            edge_attr_tensor_reps (Union[TensorReps, Irreps], optional): The edge attribute tensor representations. Defaults to None.
            max_num_neighbors (int, optional): The maximum number of neighbors for the radius-graph neighbor search. Defaults to 64.
            **kwargs: Additional keyword arguments of the LearnedGramSchmidtLFrames module.
        """
        super().__init__()
        self.in_reps = in_reps
        self.scalar_x_mask = extract_even_scalar_mask_from_reps(self.in_reps)
        self.scalar_x_dim = torch.sum(self.scalar_x_mask).item()
        self.scalar_edge_attr_mask = (
            None
            if edge_attr_tensor_reps is None
            else extract_even_scalar_mask_from_reps(edge_attr_tensor_reps)
        )
        self.scalar_edge_attr_dim = (
            0
            if self.scalar_edge_attr_mask is None
            else torch.sum(self.scalar_edge_attr_mask).item()
        )

        self.radial_module = radial_module
        self.max_radius = max_radius
        self.max_num_neighbors = max_num_neighbors
        if max_radius is not None:
            # use max radius also as the cutoff.
            kwargs["cutoff"] = max_radius

        self.lframes_module = LearnedGramSchmidtLFrames(
            even_scalar_input_dim=self.scalar_x_dim,
            even_scalar_edge_dim=self.scalar_edge_attr_dim,
            radial_dim=self.radial_module.out_dim,
            hidden_channels=hidden_channels,
            **kwargs,
        )

        if transform_into_lframes:
            self.transform_class = self.in_reps.get_transform_class()
        else:
            self.transform_class = None

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        batch: Union[Tensor, PairTensor],
        edge_attr: Union[Tensor, None] = None,
        edge_index: Union[Tensor, None] = None,
    ) -> tuple[LFrames, Tensor]:
        """Performs the forward pass of the WrappedLearnedLocalFramesModule. Works even if x, pos,
        and batch are tuples.

        Args:
            x (Union[Tensor, PairTensor]): The input tensor or tuple of tensors.
            pos (Union[Tensor, PairTensor]): The position tensor or tuple of tensors.
            batch (Union[Tensor, PairTensor]): The batch tensor or tuple of tensors.
            edge_attr (Union[Tensor, None]): The edge attribute tensor or None.
            edge_index (Union[Tensor, None]): The edge index tensor or None.

        Returns:
            LFrames: The output local frames.
            Tensor: The transformed feature tensor.
        """
        if edge_index is None:
            assert self.max_radius is not None, "need to provide edge_index if max_radius is None"
            row, col = radius(
                pos, pos, self.max_radius, batch, batch, max_num_neighbors=self.max_num_neighbors
            )
            edge_index = torch.stack([col, row], dim=0)
        else:
            assert self.max_radius is None, "max_radius should be None if edge_index is provided"
        radial = self.radial_module(pos, edge_index)
        if edge_attr is not None:
            edge_attr = edge_attr[:, self.scalar_edge_attr_mask]

        if not isinstance(x, tuple):
            x = (x, x)
        if not isinstance(pos, tuple):
            pos = (pos, pos)
        if not isinstance(batch, tuple):
            batch = (batch, batch)

        x_scalar = (
            None if x[0] is None else x[0][:, self.scalar_x_mask],
            None if x[1] is None else x[1][:, self.scalar_x_mask],
        )
        lframes = self.lframes_module(
            x=x_scalar,
            pos=pos,
            batch=batch,
            radial=radial,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # transform the features from the global frame into the new local frame:
        if self.transform_class is None:
            return lframes
        else:
            x_transformed = self.transform_class(x[1], lframes)
            return x_transformed, lframes
